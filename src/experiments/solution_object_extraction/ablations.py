"""Matched subspace ablations for the improved solution-object readout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from .features import load_activation_model
from .patching import answer_token, distribution_entropy, token_probability
from .projections import (
    fit_group_projection,
    fit_pca_projection,
    random_projection,
)
from .storage import load_experiment_config, output_dir, read_jsonl, write_json
from .sweeps import (
    forward_sequence_states,
    forward_with_tail_patch,
    load_feature_views,
    read_json,
    scope_token_count,
)
from .writer import surface_js_divergence


def run_ablation_sweep(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Compare object ablation with semantic and matched-rank controls."""
    loaded = load_experiment_config(run_path)
    records = read_jsonl(run_path / "dataset.jsonl")
    causal = read_json(
        output_dir(run_path) / "improvement" / "causal_sweep.json"
    )["selected"]
    layer = int(causal["layer"])
    dimension = int(causal["dimension"])
    view_name = str(causal["view"])
    selected_patch_scope = str(causal["scope"])
    scope = (
        "final_token"
        if selected_patch_scope == "multi_layer"
        else selected_patch_scope
    )
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    controls = build_ablation_controls(
        records=records,
        views=views,
        captured_layers=captured_layers,
        layer=layer,
        view_name=view_name,
        dimension=dimension,
    )
    candidates = [
        row
        for row in records
        if row["edit_type"] == "OPERATE"
        and row["is_correct"]
        and row["split"]
        in {"validation", "heldout_vocab", "heldout_template"}
    ]
    limit = 12 if local else int(
        loaded["experiment"]["improvement"]
        .get("ablation", {})
        .get("max_prompts", 64)
    )
    candidates = balanced_questions(candidates, limit=limit)
    model, tokenizer, _ = load_activation_model(loaded["run"]["model"])
    rows = []
    for row in tqdm(candidates, desc="matched subspace ablations", unit="prompt"):
        rows.extend(
            evaluate_ablation_prompt(
                model=model,
                tokenizer=tokenizer,
                row=row,
                layer=layer,
                controls=controls,
                scope=scope,
            )
        )
    summary = {
        mode: summarize([row for row in rows if row["mode"] == mode])
        for mode in controls
    }
    report = {
        "local": local,
        "layer": layer,
        "view": view_name,
        "scope": scope,
        "selected_patch_scope": selected_patch_scope,
        "prompts": len(candidates),
        "summary": summary,
        "object_minus_strongest_control_probability_drop": summary["object"][
            "correct_probability_drop"
        ]
        - max(
            summary[name]["correct_probability_drop"]
            for name in ("random", "lexical", "answer", "compression")
        ),
        "details": rows,
    }
    retrieval = read_json(
        output_dir(run_path) / "improvement" / "retrieval_sweep.json"
    )["selected"]
    nonlinear = read_json(
        output_dir(run_path) / "improvement" / "nonlinear_sweep.json"
    )["selected"]
    gate = {
        "status": "deferred",
        "reason": (
            "Object ablation does not exceed the matched-rank compression "
            "control; do not fit a real-trace object predictor yet."
        ),
        "checks": {
            "heldout_vocab_retrieval_at_least_baseline": bool(
                retrieval["retrieval"]["heldout_vocab"]["top1"] >= 0.7125
            ),
            "heldout_template_retrieval_at_least_baseline": bool(
                retrieval["retrieval"]["heldout_template"]["top1"] >= 0.51875
            ),
            "causal_patch_exceeds_surface_controls": bool(
                causal["object_minus_strongest_control_donor_delta"] > 0
            ),
            "object_ablation_exceeds_all_controls": bool(
                report[
                    "object_minus_strongest_control_probability_drop"
                ]
                > 0
            ),
            "nonlinear_training_improves_checkpoint": bool(
                nonlinear["selected_epoch"] > 0
            ),
        },
    }
    report["trajectory_gate"] = gate
    write_json(
        output_dir(run_path) / "improvement" / "ablation_sweep.json", report
    )
    write_json(
        output_dir(run_path) / "improvement" / "trajectory_gate.json", gate
    )
    return report


def run_ablation_grid(run_path: Path, *, local: bool) -> dict[str, Any]:
    """Run targeted ablations on low-leakage causal cells."""
    loaded = load_experiment_config(run_path)
    records = read_jsonl(run_path / "dataset.jsonl")
    cfg = loaded["experiment"]["improvement"]
    grid_cfg = cfg.get("ablation_grid", {})
    max_lexical_probe = float(grid_cfg.get("max_lexical_probe", 0.85))
    min_causal_strength = float(grid_cfg.get("min_causal_strength", 0.0))
    dimensions = {
        int(value) for value in grid_cfg.get("dimensions", [16, 32])
    }
    scopes = {
        str(value)
        for value in grid_cfg.get(
            "scopes",
            ["final_token", "last_2_tokens", "operation_interval", "multi_layer"],
        )
    }
    max_cells = int(grid_cfg.get("max_cells", 4 if local else 8))
    causal_report = read_json(
        output_dir(run_path) / "improvement" / "causal_sweep.json"
    )
    candidates = select_ablation_grid_cells(
        causal_report["results"],
        dimensions=dimensions,
        scopes=scopes,
        max_lexical_probe=max_lexical_probe,
        min_causal_strength=min_causal_strength,
        limit=max_cells,
    )
    prompt_rows = [
        row
        for row in records
        if row["edit_type"] == "OPERATE"
        and row["is_correct"]
        and row["split"]
        in {"validation", "heldout_vocab", "heldout_template"}
    ]
    prompt_limit = 8 if local else int(
        cfg.get("ablation", {}).get("max_prompts", 64)
    )
    prompt_rows = balanced_questions(prompt_rows, limit=prompt_limit)
    views, captured_layers, _ = load_feature_views(
        output_dir(run_path) / "captured_features.npz"
    )
    model, tokenizer, _ = load_activation_model(loaded["run"]["model"])
    cells = []
    improvement_dir = output_dir(run_path) / "improvement"
    for cell in tqdm(candidates, desc="targeted ablation grid", unit="cell"):
        scope = (
            "final_token"
            if str(cell["scope"]) == "multi_layer"
            else str(cell["scope"])
        )
        controls = build_ablation_controls(
            records=records,
            views=views,
            captured_layers=captured_layers,
            layer=int(cell["layer"]),
            view_name=str(cell["view"]),
            dimension=int(cell["dimension"]),
        )
        details = []
        for row in tqdm(
            prompt_rows,
            desc=f"ablate L{cell['layer']} d{cell['dimension']} {scope}",
            unit="prompt",
            leave=False,
        ):
            details.extend(
                evaluate_ablation_prompt(
                    model=model,
                    tokenizer=tokenizer,
                    row=row,
                    layer=int(cell["layer"]),
                    controls=controls,
                    scope=scope,
                )
            )
        summary = summarize_by_mode(details)
        cells.append(
            {
                "layer": int(cell["layer"]),
                "patch_layers": [int(value) for value in cell["patch_layers"]],
                "view": str(cell["view"]),
                "dimension": int(cell["dimension"]),
                "causal_scope": str(cell["scope"]),
                "ablation_scope": scope,
                "lexical_probe_accuracy": float(
                    cell["lexical_probe_accuracy"]
                ),
                "causal_strength": float(
                    cell["object_minus_strongest_control_donor_delta"]
                ),
                "prompts": len(prompt_rows),
                "summary": summary,
                "object_minus_strongest_control_probability_drop": (
                    object_minus_strongest_control(summary)
                ),
                "details": details,
            }
        )
        write_json(
            improvement_dir / "ablation_grid.partial.json",
            {
                "local": local,
                "candidate_cells": len(candidates),
                "completed_cells": len(cells),
                "cells": cells,
            },
        )
    best = (
        max(
            cells,
            key=lambda row: (
                row["object_minus_strongest_control_probability_drop"],
                row["causal_strength"],
                -row["lexical_probe_accuracy"],
            ),
        )
        if cells
        else None
    )
    retrieval = read_json(
        output_dir(run_path) / "improvement" / "retrieval_sweep.json"
    )["selected"]
    nonlinear = read_json(
        output_dir(run_path) / "improvement" / "nonlinear_sweep.json"
    )["selected"]
    gate = trajectory_gate(
        retrieval=retrieval,
        nonlinear=nonlinear,
        causal=best or causal_report["selected"],
        ablation_margin=(
            best["object_minus_strongest_control_probability_drop"]
            if best
            else float("-inf")
        ),
        max_lexical_probe=max_lexical_probe,
        source="ablation_grid",
    )
    report = {
        "local": local,
        "selection_rule": (
            "low lexical probe, positive causal strength, target dimensions "
            "and scopes, then strongest causal cells"
        ),
        "filters": {
            "dimensions": sorted(dimensions),
            "scopes": sorted(scopes),
            "max_lexical_probe": max_lexical_probe,
            "min_causal_strength": min_causal_strength,
            "max_cells": max_cells,
        },
        "candidate_cells": len(candidates),
        "cells": cells,
        "selected": best,
        "trajectory_gate": gate,
    }
    write_json(improvement_dir / "ablation_grid.json", report)
    write_json(improvement_dir / "trajectory_gate.json", gate)
    return report


def select_ablation_grid_cells(
    rows: list[dict[str, Any]],
    *,
    dimensions: set[int],
    scopes: set[str],
    max_lexical_probe: float,
    min_causal_strength: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Select the bounded low-leakage causal cells for targeted ablation."""
    candidates = [
        row
        for row in rows
        if int(row["dimension"]) in dimensions
        and str(row["scope"]) in scopes
        and float(row["lexical_probe_accuracy"]) <= max_lexical_probe
        and float(row["object_minus_strongest_control_donor_delta"])
        > min_causal_strength
    ]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["object_minus_strongest_control_donor_delta"]),
            float(row["lexical_probe_accuracy"]),
            int(row["dimension"]),
            str(row["scope"]),
            int(row["layer"]),
        ),
    )[:limit]


def build_ablation_controls(
    *,
    records: list[dict[str, Any]],
    views: dict[str, np.ndarray],
    captured_layers: list[int],
    layer: int,
    view_name: str,
    dimension: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit object and matched controls for one ablation cell."""
    values = views[view_name][:, captured_layers.index(layer)]
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"]
    )
    train = values[train_indices]
    label_sets = {
        "object": np.asarray(
            [records[index]["canonical_graph_id"] for index in train_indices]
        ),
        "lexical": np.asarray(
            [
                records[index]["surface"]["lexical_family"]
                for index in train_indices
            ]
        ),
        "answer": np.asarray(
            [str(records[index]["causal_result"]) for index in train_indices]
        ),
    }
    controls: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, labels in label_sets.items():
        requested = (
            dimension
            if name == "object"
            else min(dimension, max(len(set(labels)) - 1, 1))
        )
        controls[name] = fit_group_projection(
            train, labels, max_dim=requested
        )
    controls["compression"] = fit_pca_projection(train, max_dim=dimension)
    controls["random"] = (
        train.mean(axis=0).astype(np.float32),
        random_projection(train.shape[1], dimension, seed=42),
    )
    return controls


def evaluate_ablation_prompt(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    layer: int,
    controls: dict[str, tuple[np.ndarray, np.ndarray]],
    scope: str,
) -> list[dict[str, Any]]:
    """Evaluate all ablation controls for one prompt."""
    states, baseline = forward_sequence_states(
        model, tokenizer, row["causal_prefix"], layer
    )
    count = scope_token_count(tokenizer, row, scope)
    state_slice = states[-count:]
    correct = answer_token(tokenizer, row["causal_result"])
    baseline_probability = token_probability(baseline, correct)
    rows = []
    for name, (mean, basis) in controls.items():
        centered = state_slice - mean
        ablated = state_slice - (centered @ basis.T) @ basis
        logits = forward_with_tail_patch(
            model,
            tokenizer,
            row["causal_prefix"],
            layer,
            ablated,
        )
        greedy_token = int(torch.argmax(logits).item())
        greedy_text = tokenizer.decode(
            [greedy_token], clean_up_tokenization_spaces=False
        ).strip()
        rows.append(
            {
                "record_id": row["record_id"],
                "question_id": row["question_id"],
                "mode": name,
                "rank": int(basis.shape[0]),
                "correct_probability_drop": baseline_probability
                - token_probability(logits, correct),
                "entropy_increase": distribution_entropy(logits)
                - distribution_entropy(baseline),
                "greedy_correct": float(greedy_token == correct),
                "valid_numeric_update": float(
                    re.fullmatch(r"[-+]?\d+(?:\.\d+)?", greedy_text)
                    is not None
                ),
                "surface_js_divergence": surface_js_divergence(
                    baseline,
                    logits,
                    excluded_tokens=(correct, correct),
                ),
            }
        )
    return rows


def summarize_by_mode(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Summarize all control modes in an ablation detail table."""
    return {
        mode: summarize([row for row in rows if row["mode"] == mode])
        for mode in sorted({row["mode"] for row in rows})
    }


def object_minus_strongest_control(
    summary: dict[str, dict[str, float]]
) -> float:
    """Return object ablation drop minus the strongest non-object control."""
    return summary["object"]["correct_probability_drop"] - max(
        summary[name]["correct_probability_drop"]
        for name in ("random", "lexical", "answer", "compression")
        if name in summary
    )


def trajectory_gate(
    *,
    retrieval: dict[str, Any],
    nonlinear: dict[str, Any],
    causal: dict[str, Any],
    ablation_margin: float,
    max_lexical_probe: float | None = None,
    source: str,
) -> dict[str, Any]:
    """Decide whether the real-trace trajectory predictor should run."""
    causal_strength = float(
        causal.get(
            "object_minus_strongest_control_donor_delta",
            causal.get("causal_strength", float("-inf")),
        )
    )
    checks = {
        "heldout_vocab_retrieval_at_least_baseline": bool(
            retrieval["retrieval"]["heldout_vocab"]["top1"] >= 0.7125
        ),
        "heldout_template_retrieval_at_least_baseline": bool(
            retrieval["retrieval"]["heldout_template"]["top1"] >= 0.51875
        ),
        "causal_patch_exceeds_surface_controls": bool(causal_strength > 0),
        "object_ablation_exceeds_all_controls": bool(ablation_margin > 0),
        "nonlinear_training_improves_checkpoint": bool(
            nonlinear["selected_epoch"] > 0
        ),
    }
    if max_lexical_probe is not None:
        checks["causal_cell_low_lexical_probe"] = bool(
            causal["lexical_probe_accuracy"] <= max_lexical_probe
        )
    passed = all(checks.values())
    return {
        "status": "ready_for_real_trajectory" if passed else "deferred",
        "reason": (
            "Object ablation clears matched controls on a low-leakage causal "
            "cell; run the real-trace object predictor next."
            if passed
            else "Object ablation does not exceed all matched controls on the "
            "tested causal cells; do not fit a real-trace object predictor yet."
        ),
        "source": source,
        "checks": checks,
    }


def balanced_questions(
    rows: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    """Select prompts round-robin across questions and surface conditions."""
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_question.setdefault(str(row["question_id"]), []).append(row)
    selected = []
    offset = 0
    while len(selected) < limit:
        added = False
        for question in sorted(by_question):
            candidates = by_question[question]
            if offset < len(candidates):
                selected.append(candidates[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Average ablation metrics for one control."""
    keys = (
        "correct_probability_drop",
        "entropy_increase",
        "greedy_correct",
        "valid_numeric_update",
        "surface_js_divergence",
    )
    return {
        "rank": int(rows[0]["rank"]),
        **{
            key: float(np.mean([row[key] for row in rows]))
            for key in keys
        },
    }
