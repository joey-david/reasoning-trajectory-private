"""Dimension, layer, view, and patch-scope sweeps for object extraction."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from tqdm.auto import tqdm

from src.models.activation_capture import SelectedLayerCapture
from src.models.introspection import get_decoder_layers, resolve_layer_indices

from .decoders import decode_representation
from .features import load_activation_model
from .patching import (
    answer_token,
    distribution_entropy,
    pair_condition,
    select_causal_pairs,
    token_probability,
)
from .projections import (
    fit_group_projection,
    project,
    random_projection,
)
from .retrieval import evaluate_by_split
from .rsa import run_rsa
from .storage import (
    load_experiment_config,
    output_dir,
    read_jsonl,
    write_json,
    write_npz,
)


EVAL_SPLITS = (
    "validation",
    "template_validation",
    "heldout_vocab",
    "heldout_template",
)


def load_feature_views(
    feature_path: Path,
) -> tuple[dict[str, np.ndarray], list[int], np.ndarray]:
    """Load aligned residual-space feature views."""
    with np.load(feature_path) as data:
        pool = data["h_pool"].astype(np.float32)
        endpoint = data["h_last"].astype(np.float32)
        prefix = data["h_text_mean"].astype(np.float32)
        views = {
            "anchor_prefix": 0.5 * (pool + prefix),
            "anchor": pool,
            "endpoint": endpoint,
            "last_two": data["h_last_two"].astype(np.float32),
            "delta": data["h_delta"].astype(np.float32),
            "prefix": prefix,
        }
        return (
            views,
            data["layers"].astype(int).tolist(),
            data["record_ids"].astype(str),
        )


def run_retrieval_sweep(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Sweep readout dimensions, layers, and residual feature views."""
    loaded = load_experiment_config(run_path)
    cfg = loaded["experiment"]["improvement"]
    out = output_dir(run_path) / "improvement"
    out.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(run_path / "dataset.jsonl")
    selection_records = with_template_validation(records)
    views, captured_layers, record_ids = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    expected_ids = np.asarray([row["record_id"] for row in records], dtype=str)
    if not np.array_equal(record_ids, expected_ids):
        raise ValueError("Improvement sweep features do not match the bank")
    dimensions = [
        int(value)
        for value in (
            cfg.get("local_dimensions") if local else cfg["dimensions"]
        )
    ]
    requested_layers = [
        int(value)
        for value in (cfg.get("local_layers") if local else cfg["layers"])
    ]
    layers = [layer for layer in requested_layers if layer in captured_layers]
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"],
        dtype=int,
    )
    train_labels = np.asarray(
        [records[index]["canonical_graph_id"] for index in train_indices],
        dtype=str,
    )
    evaluation_indices = np.asarray(
        [
            index
            for index, row in enumerate(records)
            if row["split"] in {"heldout_vocab", "heldout_template"}
        ],
        dtype=int,
    )
    rows = []
    bases: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    tasks = [(layer, view) for layer in layers for view in views]
    for layer, view_name in tqdm(
        tasks, desc="retrieval layer/view sweep", unit="fit"
    ):
        layer_col = captured_layers.index(layer)
        vectors = views[view_name][:, layer_col]
        mean, full_basis = fit_group_projection(
            vectors[train_indices], train_labels, max_dim=max(dimensions)
        )
        bases[(layer, view_name)] = (mean, full_basis)
        for dimension in dimensions:
            basis = full_basis[: min(dimension, len(full_basis))]
            z = project(vectors, mean, basis)
            split_reports, _ = evaluate_by_split(
                train_vectors=z[train_indices],
                all_vectors=z,
                records=selection_records,
                train_indices=train_indices,
                splits=EVAL_SPLITS,
            )
            rsa = run_rsa(
                z[evaluation_indices],
                [records[index] for index in evaluation_indices],
            )
            heldout_indices = np.asarray(
                [
                    index
                    for index, row in enumerate(records)
                    if row["split"] == "heldout_vocab"
                ],
                dtype=int,
            )
            decoder = decode_representation(
                z[train_indices],
                z[heldout_indices],
                [records[index] for index in train_indices],
                [records[index] for index in heldout_indices],
            )
            rows.append(
                {
                    "layer": layer,
                    "view": view_name,
                    "requested_dimension": dimension,
                    "effective_dimension": int(basis.shape[0]),
                    "retrieval": split_reports,
                    "rsa": rsa,
                    "operation_macro_f1": decoder["operation"]["macro_f1"],
                    "edit_macro_f1": decoder["edit_type"]["macro_f1"],
                    "lexical_probe_accuracy": nuisance_probe_accuracy(
                        z[train_indices],
                        [
                            records[index]["surface"]["lexical_family"]
                            for index in train_indices
                        ],
                    ),
                    "template_probe_accuracy": nuisance_probe_accuracy(
                        z[train_indices],
                        [
                            records[index]["surface"]["template_id"]
                            for index in train_indices
                        ],
                    ),
                }
            )
    selected = max(
        rows,
        key=lambda row: (
            min(
                row["retrieval"]["validation"]["top1"],
                row["retrieval"]["template_validation"]["top1"],
            ),
            (
                row["retrieval"]["validation"]["top1"]
                + row["retrieval"]["template_validation"]["top1"]
            )
            / 2,
            row["retrieval"]["validation"]["mean_retrieval_margin"] or -1e9,
            -row["lexical_probe_accuracy"],
            -row["effective_dimension"],
        ),
    )
    selected_mean, selected_full_basis = bases[
        (int(selected["layer"]), str(selected["view"]))
    ]
    selected_basis = selected_full_basis[: int(selected["effective_dimension"])]
    write_npz(
        out / "retrieval_sweep_projection.npz",
        object_mean=selected_mean,
        object_basis=selected_basis,
    )
    report = {
        "selection_rule": (
            "maximin vocabulary/template-validation top1, mean validation top1, "
            "validation margin, lower lexical probe, lower dimension"
        ),
        "local": local,
        "dimensions": dimensions,
        "layers": layers,
        "views": sorted(views),
        "results": rows,
        "selected": selected,
    }
    write_json(out / "retrieval_sweep.json", report)
    return report


def with_template_validation(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split held-out-template surfaces into selection and untouched test halves."""
    output = []
    for row in records:
        copied = dict(row)
        if (
            row["split"] == "heldout_template"
            and row["surface"]["lexical_family"] == "stationery"
        ):
            copied["split"] = "template_validation"
        output.append(copied)
    return output


def nuisance_probe_accuracy(vectors: np.ndarray, labels: list[str]) -> float:
    """Estimate linearly decodable surface information in train representations."""
    y = np.asarray(labels, dtype=str)
    if len(set(y)) < 2:
        return 1.0
    folds = min(4, min(CounterLike(y).values()))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    )
    return float(
        cross_val_score(
            model,
            vectors,
            y,
            cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
        ).mean()
    )


def CounterLike(values: np.ndarray) -> dict[str, int]:
    """Return counts without importing a broader collection helper."""
    counts: defaultdict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def run_causal_sweep(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Sweep linear object patch rank, layer, and token scope."""
    loaded = load_experiment_config(run_path)
    cfg = loaded["experiment"]["improvement"]
    records = read_jsonl(run_path / "dataset.jsonl")
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    retrieval_report = read_json(
        output_dir(run_path) / "improvement" / "retrieval_sweep.json"
    )
    selected_view = str(retrieval_report["selected"]["view"])
    leakage_lookup = {
        (
            int(row["layer"]),
            str(row["view"]),
            int(row["effective_dimension"]),
        ): float(row["lexical_probe_accuracy"])
        for row in retrieval_report["results"]
    }
    dimensions = [
        int(value)
        for value in (
            cfg.get("local_dimensions") if local else cfg["dimensions"]
        )
    ]
    requested_layers = [
        int(value)
        for value in (cfg.get("local_layers") if local else cfg["layers"])
    ]
    layers = [layer for layer in requested_layers if layer in captured_layers]
    scopes = [
        str(value)
        for value in cfg.get(
            "patch_scopes", ["final_token", "last_2_tokens", "operation_interval"]
        )
    ]
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"],
        dtype=int,
    )
    train_labels = np.asarray(
        [records[index]["canonical_graph_id"] for index in train_indices],
        dtype=str,
    )
    lexical_labels = np.asarray(
        [records[index]["surface"]["lexical_family"] for index in train_indices],
        dtype=str,
    )
    bases = {}
    lexical_bases = {}
    for layer in layers:
        layer_col = captured_layers.index(layer)
        bases[layer] = fit_group_projection(
            views[selected_view][train_indices, layer_col],
            train_labels,
            max_dim=max(dimensions),
        )
        lexical_bases[layer] = fit_group_projection(
            views[selected_view][train_indices, layer_col],
            lexical_labels,
            max_dim=max(len(set(lexical_labels)) - 1, 1),
        )[1]
    model, tokenizer, _ = load_activation_model(loaded["run"]["model"])
    pairs = select_causal_pairs(
        records,
        max_pairs_per_condition=2 if local else int(
            loaded["experiment"]["causal"]["max_pairs_per_condition"]
        ),
    )
    rows = []
    tasks = []
    for layer in layers:
        seen_dimensions = set()
        for requested_dimension in dimensions:
            effective = min(requested_dimension, len(bases[layer][1]))
            if effective in seen_dimensions:
                continue
            seen_dimensions.add(effective)
            tasks.extend(
                (layer, requested_dimension, scope) for scope in scopes
            )
    for layer, requested_dimension, scope in tqdm(
        tasks, desc="causal rank/layer/scope sweep", unit="cell"
    ):
        patch_layers = (
            neighboring_layers(layers, layer) if scope == "multi_layer" else [layer]
        )
        object_bases = {
            patch_layer: bases[patch_layer][1][
                : min(requested_dimension, len(bases[patch_layer][1]))
            ]
            for patch_layer in patch_layers
        }
        random_bases = {
            patch_layer: random_projection(
                basis.shape[1], basis.shape[0], seed=42 + patch_layer
            )
            for patch_layer, basis in object_bases.items()
        }
        cell = evaluate_patch_cell(
            model=model,
            tokenizer=tokenizer,
            pairs=pairs,
            layers=patch_layers,
            object_bases=object_bases,
            random_bases=random_bases,
            lexical_bases={
                patch_layer: lexical_bases[patch_layer]
                for patch_layer in patch_layers
            },
            scope=scope,
        )
        rows.append(
            {
                "layer": layer,
                "patch_layers": patch_layers,
                "requested_dimension": requested_dimension,
                "dimension": int(object_bases[layer].shape[0]),
                "view": selected_view,
                "scope": scope,
                "lexical_probe_accuracy": leakage_lookup[
                    (
                        layer,
                        selected_view,
                        int(object_bases[layer].shape[0]),
                    )
                ],
                **cell,
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["object_minus_strongest_control_donor_delta"],
            row["object_minus_random_donor_delta"],
            row["target_probability_drop"],
            -abs(row["object_entropy_change"]),
        ),
    )
    report = {
        "local": local,
        "pairs": len(pairs),
        "selected_view_from_retrieval": selected_view,
        "results": rows,
        "pareto_frontier": causal_leakage_pareto(rows),
        "selected": selected,
        "primary_metric": "object_minus_random_donor_delta",
    }
    write_json(
        output_dir(run_path) / "improvement" / "causal_sweep.json", report
    )
    return report


def causal_leakage_pareto(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return nondominated causal-strength versus lexical-leakage cells."""
    frontier = []
    for row in rows:
        strength = row["object_minus_strongest_control_donor_delta"]
        leakage = row["lexical_probe_accuracy"]
        dominated = any(
            other["object_minus_strongest_control_donor_delta"] >= strength
            and other["lexical_probe_accuracy"] <= leakage
            and (
                other["object_minus_strongest_control_donor_delta"] > strength
                or other["lexical_probe_accuracy"] < leakage
            )
            for other in rows
        )
        if not dominated:
            frontier.append(
                {
                    "layer": row["layer"],
                    "dimension": row["dimension"],
                    "scope": row["scope"],
                    "patch_layers": row["patch_layers"],
                    "causal_strength": strength,
                    "lexical_probe_accuracy": leakage,
                }
            )
    return sorted(
        frontier,
        key=lambda row: (
            row["lexical_probe_accuracy"],
            -row["causal_strength"],
        ),
    )


def evaluate_patch_cell(
    *,
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    layers: list[int],
    object_bases: dict[int, np.ndarray],
    random_bases: dict[int, np.ndarray],
    lexical_bases: dict[int, np.ndarray],
    scope: str,
) -> dict[str, float]:
    """Evaluate object and matched random swaps for one sweep cell."""
    object_deltas = []
    random_deltas = []
    lexical_deltas = []
    target_drops = []
    entropy_changes = []
    condition_rows: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    for pair in pairs:
        target_by_layer = {}
        donor_by_layer = {}
        baseline = None
        for layer in layers:
            target_by_layer[layer], layer_baseline = forward_sequence_states(
                model, tokenizer, pair["target"]["causal_prefix"], layer
            )
            donor_by_layer[layer], _ = forward_sequence_states(
                model, tokenizer, pair["donor"]["causal_prefix"], layer
            )
            baseline = layer_baseline if baseline is None else baseline
        assert baseline is not None
        count = scope_token_count(tokenizer, pair["target"], scope)
        donor_token = answer_token(tokenizer, pair["donor"]["causal_result"])
        target_token = answer_token(tokenizer, pair["target"]["causal_result"])
        baseline_donor = token_probability(baseline, donor_token)
        baseline_target = token_probability(baseline, target_token)
        patch_deltas: dict[str, dict[int, np.ndarray]] = {
            "object": {},
            "random": {},
            "lexical": {},
        }
        for layer in layers:
            target_slice = target_by_layer[layer][-count:]
            donor_slice = aligned_tail(donor_by_layer[layer], count)
            delta = donor_slice - target_slice
            for name, basis in (
                ("object", object_bases[layer]),
                ("random", random_bases[layer]),
                ("lexical", lexical_bases[layer]),
            ):
                patch_deltas[name][layer] = (delta @ basis.T) @ basis
        object_logits = forward_with_layer_deltas(
            model,
            tokenizer,
            pair["target"]["causal_prefix"],
            patch_deltas["object"],
        )
        random_logits = forward_with_layer_deltas(
            model,
            tokenizer,
            pair["target"]["causal_prefix"],
            patch_deltas["random"],
        )
        lexical_logits = forward_with_layer_deltas(
            model,
            tokenizer,
            pair["target"]["causal_prefix"],
            patch_deltas["lexical"],
        )
        object_deltas.append(
            token_probability(object_logits, donor_token) - baseline_donor
        )
        random_deltas.append(
            token_probability(random_logits, donor_token) - baseline_donor
        )
        lexical_deltas.append(
            token_probability(lexical_logits, donor_token) - baseline_donor
        )
        target_drops.append(
            baseline_target - token_probability(object_logits, target_token)
        )
        entropy_changes.append(
            distribution_entropy(object_logits) - distribution_entropy(baseline)
        )
        condition_rows[str(pair["condition"])].append(
            {
                "object": object_deltas[-1],
                "random": random_deltas[-1],
                "lexical": lexical_deltas[-1],
                "target_drop": target_drops[-1],
            }
        )
    object_delta = float(np.mean(object_deltas))
    random_delta = float(np.mean(random_deltas))
    lexical_delta = float(np.mean(lexical_deltas))
    return {
        "object_donor_probability_delta": object_delta,
        "random_donor_probability_delta": random_delta,
        "lexical_donor_probability_delta": lexical_delta,
        "object_minus_random_donor_delta": object_delta - random_delta,
        "object_minus_lexical_donor_delta": object_delta - lexical_delta,
        "object_minus_strongest_control_donor_delta": object_delta
        - max(random_delta, lexical_delta),
        "target_probability_drop": float(np.mean(target_drops)),
        "object_entropy_change": float(np.mean(entropy_changes)),
        "by_condition": {
            condition: {
                key: float(np.mean([row[key] for row in values]))
                for key in ("object", "random", "lexical", "target_drop")
            }
            for condition, values in sorted(condition_rows.items())
        },
    }


def neighboring_layers(layers: list[int], center: int) -> list[int]:
    """Return the center and its nearest swept layer on either side."""
    index = layers.index(center)
    return layers[max(0, index - 1) : min(len(layers), index + 2)]


def forward_with_layer_deltas(
    model: Any,
    tokenizer: Any,
    text: str,
    deltas: dict[int, np.ndarray],
) -> torch.Tensor:
    """Add aligned donor deltas to the live stream at one or more layers."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    decoder_layers = get_decoder_layers(model)
    handles = []
    for requested, delta in deltas.items():
        resolved = resolve_layer_indices([requested], len(decoder_layers))[0]

        def hook(
            _module: Any,
            _inputs: Any,
            output: Any,
            *,
            layer_delta: np.ndarray = delta,
        ) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            replaced = hidden.clone()
            count = min(len(layer_delta), hidden.shape[1])
            replaced[0, -count:] = hidden[0, -count:] + torch.as_tensor(
                layer_delta[-count:],
                device=hidden.device,
                dtype=hidden.dtype,
            )
            return (
                (replaced, *output[1:])
                if isinstance(output, tuple)
                else replaced
            )

        handles.append(decoder_layers[resolved].register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return output.logits[0, -1].float().cpu()


def forward_sequence_states(
    model: Any, tokenizer: Any, text: str, layer: int
) -> tuple[np.ndarray, torch.Tensor]:
    """Capture all prompt states at one layer and next-token logits."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = get_decoder_layers(model)
    resolved = resolve_layer_indices([layer], len(layers))
    with (
        torch.inference_mode(),
        SelectedLayerCapture(
            decoder_layers=layers,
            requested_layers=[layer],
            resolved_layers=resolved,
        ) as capture,
    ):
        output = model(**encoded, use_cache=False, return_dict=True)
    return (
        capture.outputs[layer][0].float().cpu().numpy(),
        output.logits[0, -1].float().cpu(),
    )


def forward_with_tail_patch(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
    states: np.ndarray,
) -> torch.Tensor:
    """Patch a tail interval at one decoder layer."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = get_decoder_layers(model)
    resolved = resolve_layer_indices([layer], len(layers))[0]

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        replaced = hidden.clone()
        count = min(len(states), hidden.shape[1])
        replaced[0, -count:] = torch.as_tensor(
            states[-count:], device=hidden.device, dtype=hidden.dtype
        )
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    handle = layers[resolved].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return output.logits[0, -1].float().cpu()


def scope_token_count(tokenizer: Any, row: dict[str, Any], scope: str) -> int:
    """Resolve a patch scope to a target-tail token count."""
    if scope == "final_token":
        return 1
    if scope == "last_2_tokens":
        return 2
    if scope == "operation_interval":
        expression = row["causal_prefix"].rsplit("The final computation is ", 1)[-1]
        return max(1, len(tokenizer.encode(expression, add_special_tokens=False)))
    if scope == "multi_layer":
        return 1
    raise ValueError(f"Unsupported local patch scope: {scope}")


def aligned_tail(states: np.ndarray, count: int) -> np.ndarray:
    """Return exactly ``count`` donor tail states, left-padding when necessary."""
    if len(states) >= count:
        return states[-count:]
    padding = np.repeat(states[[0]], count - len(states), axis=0)
    return np.concatenate([padding, states], axis=0)


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON report."""
    import json

    return json.loads(path.read_text(encoding="utf-8"))
