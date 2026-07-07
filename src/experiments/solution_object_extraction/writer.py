"""Learn a separate residual writer for transferring encoded object deltas."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.models.introspection import get_decoder_layers, resolve_layer_indices

from .features import load_activation_model
from .nonlinear import (
    ObjectEncoder,
    build_labels,
    build_multiview_input,
    encode_all,
)
from .patching import (
    answer_token,
    distribution_entropy,
    select_causal_pairs,
    token_probability,
)
from .projections import fit_group_projection, project, random_projection
from .storage import (
    load_experiment_config,
    output_dir,
    read_jsonl,
    write_json,
)
from .sweeps import (
    forward_sequence_states,
    forward_with_tail_patch,
    load_feature_views,
    read_json,
)


class CausalWriter(torch.nn.Module):
    """Map donor-target latent differences and target context to a residual edit."""

    def __init__(self, latent_dim: int, hidden_dim: int, residual_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, residual_dim),
        )
        torch.nn.init.zeros_(self.network[-1].weight)
        torch.nn.init.zeros_(self.network[-1].bias)

    def forward(
        self, donor_delta: torch.Tensor, target_context: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat([donor_delta, target_context], dim=-1))


def run_writer_experiment(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Train a causal writer and compare it with linear/full/random patching."""
    loaded = load_experiment_config(run_path)
    experiment = loaded["experiment"]
    cfg = experiment["improvement"]["writer"]
    records = read_jsonl(run_path / "dataset.jsonl")
    z, nonlinear_manifest = load_nonlinear_latents(run_path, records, local=local)
    record_index = {
        str(row["record_id"]): index for index, row in enumerate(records)
    }
    causal_sweep = read_json(
        output_dir(run_path) / "improvement" / "causal_sweep.json"
    )
    layer = int(causal_sweep["selected"]["layer"])
    dimension = int(causal_sweep["selected"]["dimension"])
    view = str(causal_sweep["selected"]["view"])
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"]
    )
    graph_labels = np.asarray(
        [records[index]["canonical_graph_id"] for index in train_indices]
    )
    lexical_labels = np.asarray(
        [records[index]["surface"]["lexical_family"] for index in train_indices]
    )
    layer_col = captured_layers.index(layer)
    projection_mean, full_basis = fit_group_projection(
        views[view][train_indices, layer_col],
        graph_labels,
        max_dim=dimension,
    )
    linear_basis = full_basis[:dimension]
    _lexical_mean, lexical_basis = fit_group_projection(
        views[view][train_indices, layer_col],
        lexical_labels,
        max_dim=min(dimension, max(len(set(lexical_labels)) - 1, 1)),
    )
    random_basis = random_projection(
        linear_basis.shape[1], linear_basis.shape[0], seed=42
    )
    train_pairs, validation_pairs, pair_split = split_writer_pairs(
        records,
        train_limit=int(cfg["train_pairs"]),
        validation_limit=int(cfg["validation_pairs"]),
    )
    model, tokenizer, _ = load_activation_model(loaded["run"]["model"])
    model.requires_grad_(False)
    device = model.get_input_embeddings().weight.device
    writer = CausalWriter(
        z.shape[1],
        hidden_dim=128 if local else 256,
        residual_dim=linear_basis.shape[0],
    ).to(device)
    basis_tensor = torch.from_numpy(linear_basis).to(device=device)
    optimizer = torch.optim.AdamW(writer.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs_cfg = cfg.get("epochs", 2)
    epoch_targets = sorted(
        {
            int(value)
            for value in (
                epochs_cfg if isinstance(epochs_cfg, list) else [epochs_cfg]
            )
        }
    )
    epochs = max(epoch_targets)
    alphas = [float(value) for value in cfg["alpha"]]
    train_alpha = 1.0 if 1.0 in alphas else alphas[0]
    losses = []
    writer_results = []
    writer_checkpoints = {}
    for _epoch in tqdm(range(epochs), desc="causal writer training", unit="epoch"):
        epoch_losses = []
        for pair in train_pairs:
            target_z = torch.from_numpy(
                z[record_index[pair["target"]["record_id"]]]
            ).to(device)
            donor_z = torch.from_numpy(
                z[record_index[pair["donor"]["record_id"]]]
            ).to(device)
            coefficients = writer(donor_z - target_z, target_z)
            delta = coefficients.to(basis_tensor.dtype) @ basis_tensor
            logits = differentiable_patch_logits(
                model,
                tokenizer,
                pair["target"]["causal_prefix"],
                layer,
                train_alpha * delta,
            )
            donor_token = answer_token(tokenizer, pair["donor"]["causal_result"])
            target_token = answer_token(tokenizer, pair["target"]["causal_result"])
            log_probabilities = torch.log_softmax(logits, dim=-1)
            with torch.no_grad():
                _states, baseline_logits = forward_sequence_states(
                    model, tokenizer, pair["target"]["causal_prefix"], layer
                )
            surface_kl = finite_masked_kl_divergence(
                baseline_logits.to(device),
                logits,
                excluded_tokens=(donor_token, target_token),
            )
            loss = (
                -log_probabilities[donor_token]
                + 0.1 * log_probabilities[target_token]
                + 0.2 * surface_kl
                + 1e-3 * torch.mean(delta.float() ** 2)
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite writer loss for "
                    f"{pair['target']['record_id']} <- "
                    f"{pair['donor']['record_id']}"
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
        epoch = _epoch + 1
        if epoch in epoch_targets:
            for alpha in alphas:
                result = evaluate_writer(
                    model=model,
                    tokenizer=tokenizer,
                    writer=writer,
                    writer_basis=linear_basis,
                    z=z,
                    record_index=record_index,
                    pairs=validation_pairs,
                    layer=layer,
                    alpha=alpha,
                )
                result["epoch"] = epoch
                writer_results.append(result)
            writer_checkpoints[epoch] = {
                key: value.detach().cpu().clone()
                for key, value in writer.state_dict().items()
            }
    controls = evaluate_controls(
        model=model,
        tokenizer=tokenizer,
        pairs=validation_pairs,
        layer=layer,
        object_basis=linear_basis,
        random_basis=random_basis,
        lexical_basis=lexical_basis,
    )
    selected = max(
        writer_results,
        key=lambda row: (
            row["donor_probability_delta"],
            row["target_probability_drop"],
            -abs(row["entropy_change"]),
        ),
    )
    linear_control = controls["linear_subspace"]
    writer_beats_linear = bool(
        selected["donor_probability_delta"]
        > linear_control["donor_probability_delta"]
        and selected["surface_js_divergence"]
        <= linear_control["surface_js_divergence"]
    )
    writer.load_state_dict(writer_checkpoints[int(selected["epoch"])])
    out = output_dir(run_path) / "improvement"
    torch.save(
        {
            "state_dict": {
                key: value.detach().cpu()
                for key, value in writer.state_dict().items()
            },
            "layer": layer,
            "latent_dim": int(z.shape[1]),
            "residual_dim": int(linear_basis.shape[0]),
            "patch_dimension": int(linear_basis.shape[1]),
            "selected": selected,
            "nonlinear_manifest": nonlinear_manifest,
        },
        out / "causal_writer.pt",
    )
    report = {
        "local": local,
        "layer": layer,
        "linear_dimension": dimension,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "pair_split": pair_split,
        "epoch_sweep": epoch_targets,
        "losses": losses,
        "writer_results": writer_results,
        "controls": controls,
        "selected": selected,
        "writer_beats_linear": writer_beats_linear,
        "recommended_method": (
            "learned_writer" if writer_beats_linear else "linear_subspace"
        ),
        "primary_metric": "donor_probability_delta",
    }
    write_json(out / "writer_report.json", report)
    return report


def load_nonlinear_latents(
    run_path: Path,
    records: list[dict[str, Any]],
    *,
    local: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rebuild inputs and load the selected nonlinear encoder."""
    checkpoint = torch.load(
        output_dir(run_path) / "improvement" / "nonlinear_encoder.pt",
        map_location="cpu",
        weights_only=True,
    )
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    layers = sorted(
        {int(item["layer"]) for item in checkpoint["input_manifest"]}
    )
    per_view_dim = int(checkpoint["input_manifest"][0]["dimensions"])
    retrieval = read_json(
        output_dir(run_path) / "improvement" / "retrieval_sweep.json"
    )
    selected_linear = retrieval["selected"]
    with np.load(
        output_dir(run_path) / "improvement" / "retrieval_sweep_projection.npz"
    ) as projection_data:
        base_values = project(
            views[str(selected_linear["view"])][
                :, captured_layers.index(int(selected_linear["layer"]))
            ],
            projection_data["object_mean"].astype(np.float32),
            projection_data["object_basis"].astype(np.float32),
        )
    x, manifest = build_multiview_input(
        views,
        captured_layers,
        layers,
        per_view_dim=per_view_dim,
        base_values=base_values,
    )
    selected = checkpoint["selected"]
    labels = build_labels(records)
    model = ObjectEncoder(
        int(checkpoint["input_dim"]),
        int(selected["hidden_dimension"]),
        int(selected["latent_dimension"]),
        {name: len(values) for name, values in labels.classes.items()},
        base_dim=int(checkpoint["base_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    if torch.cuda.is_available():
        model.to("cuda")
    elif torch.backends.mps.is_available():
        model.to("mps")
    return encode_all(model, x), {"input_manifest": manifest, "selected": selected}


def differentiable_patch_logits(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Add a differentiable writer edit at one final-token residual state."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = get_decoder_layers(model)
    resolved = resolve_layer_indices([layer], len(layers))[0]

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        replaced = hidden.clone()
        replaced[0, -1] = hidden[0, -1] + delta.to(hidden.dtype)
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    handle = layers[resolved].register_forward_hook(hook)
    try:
        output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return output.logits[0, -1].float()


def finite_masked_kl_divergence(
    baseline_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    *,
    excluded_tokens: tuple[int, int],
) -> torch.Tensor:
    """KL over the finite non-answer vocabulary used by the writer loss."""
    baseline = baseline_logits.detach().float()
    patched = patched_logits.float()
    mask = torch.ones_like(baseline, dtype=torch.bool)
    for token in set(excluded_tokens):
        mask[token] = False
    baseline_probabilities = torch.softmax(baseline[mask], dim=-1)
    patched_log_probabilities = torch.log_softmax(patched[mask], dim=-1)
    value = torch.nn.functional.kl_div(
        patched_log_probabilities,
        baseline_probabilities,
        reduction="sum",
    )
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite masked surface KL")
    return value.clamp_min(0)


def evaluate_writer(
    *,
    model: Any,
    tokenizer: Any,
    writer: CausalWriter,
    writer_basis: np.ndarray,
    z: np.ndarray,
    record_index: dict[str, int],
    pairs: list[dict[str, Any]],
    layer: int,
    alpha: float,
) -> dict[str, float]:
    """Evaluate one writer scale on held-out pairs."""
    device = next(writer.parameters()).device
    basis = torch.from_numpy(writer_basis).to(device)
    metrics = []
    writer.eval()
    for pair in pairs:
        target_z = torch.from_numpy(
            z[record_index[pair["target"]["record_id"]]]
        ).to(device)
        donor_z = torch.from_numpy(
            z[record_index[pair["donor"]["record_id"]]]
        ).to(device)
        with torch.no_grad():
            coefficients = writer(donor_z - target_z, target_z)
            delta = coefficients.to(basis.dtype) @ basis
            _baseline_states, baseline = forward_sequence_states(
                model, tokenizer, pair["target"]["causal_prefix"], layer
            )
            logits = differentiable_patch_logits(
                model,
                tokenizer,
                pair["target"]["causal_prefix"],
                layer,
                alpha * delta,
            )
        metrics.append(
            patch_metrics(tokenizer, pair, baseline, logits)
        )
    return aggregate_metrics(metrics, alpha=alpha)


def evaluate_controls(
    *,
    model: Any,
    tokenizer: Any,
    pairs: list[dict[str, Any]],
    layer: int,
    object_basis: np.ndarray,
    random_basis: np.ndarray,
    lexical_basis: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Evaluate linear, random, and full-vector final-token swaps."""
    rows: dict[str, list[dict[str, float]]] = {
        "linear_subspace": [],
        "random_subspace": [],
        "lexical_subspace": [],
        "full_vector": [],
    }
    for pair in pairs:
        target, baseline = forward_sequence_states(
            model, tokenizer, pair["target"]["causal_prefix"], layer
        )
        donor, _ = forward_sequence_states(
            model, tokenizer, pair["donor"]["causal_prefix"], layer
        )
        target_state = target[-1]
        donor_state = donor[-1]
        states = {
            "linear_subspace": target_state
            + ((donor_state - target_state) @ object_basis.T) @ object_basis,
            "random_subspace": target_state
            + ((donor_state - target_state) @ random_basis.T) @ random_basis,
            "lexical_subspace": target_state
            + ((donor_state - target_state) @ lexical_basis.T) @ lexical_basis,
            "full_vector": donor_state,
        }
        for name, state in states.items():
            logits = forward_with_tail_patch(
                model,
                tokenizer,
                pair["target"]["causal_prefix"],
                layer,
                state[None],
            )
            rows[name].append(patch_metrics(tokenizer, pair, baseline, logits))
    return {
        name: aggregate_metrics(values, alpha=1.0)
        for name, values in rows.items()
    }


def split_writer_pairs(
    records: list[dict[str, Any]],
    *,
    train_limit: int,
    validation_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build question-disjoint train and validation donor-target pairs."""
    questions = sorted(
        {
            str(row["question_id"])
            for row in records
            if row["edit_type"] == "OPERATE"
            and row["is_correct"]
            and row["split"]
            in {"validation", "heldout_vocab", "heldout_template"}
        }
    )
    validation_question_count = max(2, len(questions) // 4)
    validation_questions = set(questions[-validation_question_count:])
    train_questions = set(questions) - validation_questions

    def pairs_for(allowed: set[str], limit: int) -> list[dict[str, Any]]:
        subset = [
            row
            for row in records
            if str(row["question_id"]) in allowed
        ]
        candidates = [
            pair
            for pair in select_causal_pairs(
                subset, max_pairs_per_condition=10_000
            )
            if pair["target"]["causal_result"]
            != pair["donor"]["causal_result"]
        ]
        by_question: dict[str, list[dict[str, Any]]] = {
            question: [] for question in sorted(allowed)
        }
        for pair in candidates:
            for target, donor in (
                (pair["target"], pair["donor"]),
                (pair["donor"], pair["target"]),
            ):
                question = str(target["question_id"])
                if question not in by_question:
                    continue
                by_question[question].append(
                    {
                        **pair,
                        "target": target,
                        "donor": donor,
                    }
                )
        selected = []
        offsets = {question: 0 for question in by_question}
        while len(selected) < limit:
            added = False
            for question, question_pairs in by_question.items():
                offset = offsets[question]
                if offset >= len(question_pairs):
                    continue
                selected.append(question_pairs[offset])
                offsets[question] += 1
                added = True
                if len(selected) == limit:
                    break
            if not added:
                break
        return selected

    train_pairs = pairs_for(train_questions, train_limit)
    validation_pairs = pairs_for(validation_questions, validation_limit)
    if not train_pairs or not validation_pairs:
        raise ValueError("Could not construct question-disjoint writer pairs")
    return train_pairs, validation_pairs, {
        "unit": "question_id",
        "train_questions": sorted(train_questions),
        "validation_questions": sorted(validation_questions),
        "overlap": sorted(train_questions & validation_questions),
    }


def patch_metrics(
    tokenizer: Any,
    pair: dict[str, Any],
    baseline: torch.Tensor,
    patched: torch.Tensor,
) -> dict[str, float]:
    """Compute type-corrected writer metrics for one pair."""
    donor = answer_token(tokenizer, pair["donor"]["causal_result"])
    target = answer_token(tokenizer, pair["target"]["causal_result"])
    donor_delta = token_probability(patched, donor) - token_probability(
        baseline, donor
    )
    target_drop = token_probability(baseline, target) - token_probability(
        patched, target
    )
    return {
        "donor_delta": donor_delta,
        "target_drop": target_drop,
        "flip": float(
            token_probability(patched, donor)
            > token_probability(patched, target)
        ),
        "greedy_donor": float(int(torch.argmax(patched).item()) == donor),
        "entropy_change": distribution_entropy(patched)
        - distribution_entropy(baseline),
        "surface_js": surface_js_divergence(
            baseline, patched, excluded_tokens=(donor, target)
        ),
    }


def aggregate_metrics(
    rows: list[dict[str, float]], *, alpha: float
) -> dict[str, float]:
    """Aggregate pair-level patch metrics."""
    return {
        "alpha": alpha,
        "donor_probability_delta": float(
            np.mean([row["donor_delta"] for row in rows])
        ),
        "target_probability_drop": float(
            np.mean([row["target_drop"] for row in rows])
        ),
        "flip_rate": float(np.mean([row["flip"] for row in rows])),
        "greedy_donor_rate": float(
            np.mean([row["greedy_donor"] for row in rows])
        ),
        "entropy_change": float(
            np.mean([row["entropy_change"] for row in rows])
        ),
        "surface_js_divergence": float(
            np.mean([row["surface_js"] for row in rows])
        ),
    }


def surface_js_divergence(
    baseline: torch.Tensor,
    patched: torch.Tensor,
    *,
    excluded_tokens: tuple[int, int],
) -> float:
    """Measure non-answer distribution drift as a surface-preservation proxy."""
    p = torch.softmax(baseline.detach().float().cpu(), dim=-1)
    q = torch.softmax(patched.detach().float().cpu(), dim=-1)
    p = p.clone()
    q = q.clone()
    for token in set(excluded_tokens):
        p[token] = 0
        q[token] = 0
    p /= p.sum().clamp_min(1e-12)
    q /= q.sum().clamp_min(1e-12)
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (
        torch.sum(torch.special.xlogy(p, p / midpoint.clamp_min(1e-12)))
        + torch.sum(torch.special.xlogy(q, q / midpoint.clamp_min(1e-12)))
    )
    return float(divergence)
