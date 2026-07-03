"""Held-out evaluation of objective-relative token boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.experiments.token_segmentation.data import (
    TraceMeta,
    boundary_snippet,
    load_gold_targets,
    load_states,
    prepare_traces,
)
from src.experiments.token_segmentation.signals import (
    OBJECTIVES,
    boundary_budget,
    fit_boundary_models,
    fit_correctness_model,
    fit_projection,
    top_boundaries,
    transition_signals,
)


def run_token_segmentation(
    run_path: Path,
    *,
    gold_run: Path,
    updates_path: Path | None = None,
    layer: int = -1,
    min_segment_tokens: int = 4,
) -> Path:
    """Run the complete token-boundary comparison on existing artifacts."""
    updates_path = updates_path or (
        run_path
        / "analysis"
        / "experiments"
        / "h2_localized_updates"
        / "updates.jsonl"
    )
    traces = prepare_traces(run_path, updates_path)
    projection = fit_projection(run_path, traces, layer=layer)
    gold_targets = load_gold_targets(gold_run, layer)
    correctness = fit_correctness_model(run_path, traces, projection, layer=layer)
    boundary_models = fit_boundary_models(
        run_path,
        traces,
        projection,
        gold_targets,
        correctness,
        layer=layer,
        min_segment_tokens=min_segment_tokens,
    )
    result = evaluate_test_traces(
        run_path,
        traces,
        projection,
        gold_targets,
        correctness,
        boundary_models,
        layer,
        min_segment_tokens,
    )
    output_name = (
        "token_segmentation"
        if min_segment_tokens == 4
        else f"token_segmentation_gap{min_segment_tokens}"
    )
    out_dir = run_path / "analysis" / "experiments" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, result)
    return out_dir / "report.json"


def evaluate_test_traces(
    run_path: Path,
    traces: list[TraceMeta],
    projection: Any,
    gold_targets: dict[str, np.ndarray],
    correctness_model: Any,
    boundary_models: dict[str, Any],
    layer: int,
    min_segment_tokens: int,
) -> dict[str, Any]:
    """Score token segmenters against every held-out objective."""
    methods = [
        "random",
        "latent_magnitude",
        "latent_cosine",
        "latent_curvature",
        "sentence_boundaries",
        *[f"oracle_{name}" for name in OBJECTIVES],
        *[f"learned_{name}" for name in OBJECTIVES],
    ]
    utilities: dict[str, dict[str, list[tuple[str, float]]]] = {
        method: {objective: [] for objective in OBJECTIVES} for method in methods
    }
    agreements: dict[str, dict[str, list[float]]] = {
        left: {right: [] for right in OBJECTIVES} for left in OBJECTIVES
    }
    transfer_scores: dict[str, dict[str, list[float]]] = {
        source: {target: [] for target in OBJECTIVES} for source in OBJECTIVES
    }
    boundary_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    evaluated = 0
    evaluated_tokens = 0
    for trace in traces:
        if trace.train or len(trace.object_boundaries) < 2:
            continue
        states = load_states(run_path, trace, layer)
        signals, features = transition_signals(
            states,
            projection,
            gold_target=gold_targets.get(trace.sample_id),
            correctness_model=correctness_model,
            object_boundaries=trace.object_boundaries,
        )
        budget = boundary_budget(trace, len(features), min_segment_tokens)
        objective_boundaries = {
            name: top_boundaries(
                signals[name], budget, min_segment_tokens=min_segment_tokens
            )
            for name in OBJECTIVES
        }
        selected = {
            name: top_boundaries(
                signals[name], budget, min_segment_tokens=min_segment_tokens
            )
            for name in ("latent_magnitude", "latent_cosine", "latent_curvature")
        }
        rng = np.random.default_rng(trace.seed)
        selected["random"] = top_boundaries(
            rng.random(len(features)),
            budget,
            min_segment_tokens=min_segment_tokens,
        )
        selected["sentence_boundaries"] = matched_sentence_boundaries(
            trace.sentence_boundaries,
            len(features),
            budget,
            min_segment_tokens,
        )
        selected.update(
            {f"oracle_{name}": indices for name, indices in objective_boundaries.items()}
        )
        learned_scores: dict[str, np.ndarray] = {}
        for name, model in boundary_models.items():
            learned_scores[name] = model.predict_proba(features)[:, 1]
            selected[f"learned_{name}"] = top_boundaries(
                learned_scores[name],
                budget,
                min_segment_tokens=min_segment_tokens,
            )

        for method, indices in selected.items():
            for objective in OBJECTIVES:
                value = normalized_utility(
                    signals[objective], indices, objective_boundaries[objective]
                )
                if np.isfinite(value):
                    utilities[method][objective].append((trace.sample_id, value))
        for left in OBJECTIVES:
            for right in OBJECTIVES:
                agreements[left][right].append(
                    boundary_f1(
                        objective_boundaries[left],
                        objective_boundaries[right],
                        tolerance=4,
                    )
                )
        for source, scores in learned_scores.items():
            for target, labels_at in objective_boundaries.items():
                labels = np.zeros(len(scores), dtype=np.int8)
                labels[labels_at] = 1
                if len(np.unique(labels)) == 2:
                    transfer_scores[source][target].append(
                        float(roc_auc_score(labels, scores))
                    )
        boundary_rows.append(
            {
                "sample_id": trace.sample_id,
                "seed": trace.seed,
                "is_correct": trace.is_correct,
                "tokens": len(states),
                "budget": budget,
                "boundaries": {
                    name: values.tolist() for name, values in selected.items()
                },
            }
        )
        if len(examples) < 40:
            append_examples(examples, trace, selected, signals, budget)
        evaluated += 1
        evaluated_tokens += len(states)

    matrix = grouped_mean_nested(utilities)
    matrix_intervals = grouped_bootstrap_intervals(utilities)
    agreement_matrix = mean_nested(agreements)
    transfer_matrix = mean_nested(transfer_scores)
    return {
        "report": {
            "experiment": "objective_relative_token_segmentation",
            "unit": "every generated-token transition",
            "boundary_budget": "verified symbolic update count, capped at 64",
            "minimum_segment_tokens": min_segment_tokens,
            "split": "question-disjoint; every fifth sorted question held out",
            "layer": layer,
            "traces_total": len(traces),
            "traces_train": sum(trace.train for trace in traces),
            "traces_test_evaluated": evaluated,
            "test_tokens": evaluated_tokens,
            "pca_dimensions": int(projection.n_components_),
            "correctness_probe_available": correctness_model is not None,
            "correct_train": sum(trace.train and trace.is_correct for trace in traces),
            "incorrect_train": sum(
                trace.train and not trace.is_correct for trace in traces
            ),
            "correct_test": sum(
                not trace.train and trace.is_correct for trace in traces
            ),
            "incorrect_test": sum(
                not trace.train and not trace.is_correct for trace in traces
            ),
            "objective_definitions": {
                "answer": "positive tokenwise increase in cosine alignment with the mean teacher-forced gold-solution state",
                "object": "completion token of a deterministically verified symbolic update",
                "correctness": "absolute tokenwise change in a train-only linear final-correctness probe",
                "compression": "difference between local left/right means in train-only PCA space",
            },
            "objective_matrix": matrix,
            "objective_matrix_question_bootstrap_95ci": matrix_intervals,
            "oracle_boundary_f1_tolerance_4": agreement_matrix,
            "supervised_transfer_auc": transfer_matrix,
            "best_maximin_method": best_maximin(matrix),
        },
        "matrix": matrix,
        "agreements": agreement_matrix,
        "transfer": transfer_matrix,
        "boundaries": boundary_rows,
        "examples": examples,
    }


def normalized_utility(
    scores: np.ndarray,
    selected: np.ndarray,
    oracle: np.ndarray,
) -> float:
    """Normalize selected score mass between random expectation and the oracle."""
    scores = np.asarray(scores, dtype=np.float64)
    if not len(scores) or not len(selected):
        return float("nan")
    random_expected = len(selected) * float(np.mean(scores))
    optimum = float(np.sum(scores[oracle]))
    denominator = optimum - random_expected
    if abs(denominator) < 1e-10:
        return float("nan")
    return (float(np.sum(scores[selected])) - random_expected) / denominator


def matched_sentence_boundaries(
    boundaries: np.ndarray,
    transition_count: int,
    budget: int,
    min_segment_tokens: int,
) -> np.ndarray:
    """Match sentence-parser boundaries to the common token-boundary budget."""
    valid = boundaries[(boundaries >= 0) & (boundaries < transition_count)]
    primary = (
        valid[
            np.unique(
                np.linspace(
                    0,
                    len(valid) - 1,
                    min(budget, len(valid)),
                    dtype=int,
                )
            )
        ]
        if len(valid)
        else valid
    )
    ordered = np.r_[primary, valid]
    selected: list[int] = []
    for boundary in ordered:
        value = int(boundary)
        if value < min_segment_tokens - 1:
            continue
        if value > transition_count - min_segment_tokens:
            continue
        if all(abs(value - other) >= min_segment_tokens for other in selected):
            selected.append(value)
            if len(selected) == budget:
                break
    return np.asarray(sorted(selected), dtype=np.int32)


def boundary_f1(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: int,
) -> float:
    """Greedily match two boundary sets within a token tolerance."""
    remaining = set(map(int, target))
    matches = 0
    for boundary in map(int, predicted):
        candidates = [value for value in remaining if abs(value - boundary) <= tolerance]
        if not candidates:
            continue
        closest = min(candidates, key=lambda value: abs(value - boundary))
        remaining.remove(closest)
        matches += 1
    if not len(predicted) or not len(target):
        return 0.0
    precision = matches / len(predicted)
    recall = matches / len(target)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def append_examples(
    output: list[dict[str, Any]],
    trace: TraceMeta,
    selected: dict[str, np.ndarray],
    signals: dict[str, np.ndarray],
    budget: int,
) -> None:
    """Save a few high-scoring token boundaries with surrounding text."""
    for method in ("oracle_answer", "oracle_object", "latent_magnitude"):
        for boundary in selected[method][-min(2, budget) :]:
            output.append(
                {
                    "sample_id": trace.sample_id,
                    "seed": trace.seed,
                    "method": method,
                    "token_boundary": int(boundary),
                    "answer_score": float(signals["answer"][boundary]),
                    "object_score": float(signals["object"][boundary]),
                    "snippet": boundary_snippet(trace, int(boundary)),
                }
            )


def mean_nested(
    values: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, float | None]]:
    """Average a nested collection while preserving unavailable cells."""
    return {
        row: {
            column: float(np.mean(cell)) if cell else None
            for column, cell in columns.items()
        }
        for row, columns in values.items()
    }


def grouped_mean_nested(
    values: dict[str, dict[str, list[tuple[str, float]]]],
) -> dict[str, dict[str, float | None]]:
    """Average traces within questions before averaging across questions."""
    output: dict[str, dict[str, float | None]] = {}
    for row, columns in values.items():
        output[row] = {}
        for column, records in columns.items():
            grouped: dict[str, list[float]] = {}
            for question, value in records:
                grouped.setdefault(question, []).append(value)
            output[row][column] = (
                float(np.mean([np.mean(items) for items in grouped.values()]))
                if grouped
                else None
            )
    return output


def grouped_bootstrap_intervals(
    values: dict[str, dict[str, list[tuple[str, float]]]],
    *,
    samples: int = 2000,
) -> dict[str, dict[str, list[float] | None]]:
    """Bootstrap question means for primary utility uncertainty."""
    rng = np.random.default_rng(0)
    output: dict[str, dict[str, list[float] | None]] = {}
    for row, columns in values.items():
        output[row] = {}
        for column, records in columns.items():
            grouped: dict[str, list[float]] = {}
            for question, value in records:
                grouped.setdefault(question, []).append(value)
            means = np.asarray(
                [np.mean(items) for items in grouped.values()], dtype=np.float64
            )
            if not len(means):
                output[row][column] = None
                continue
            draws = rng.choice(means, size=(samples, len(means)), replace=True).mean(
                axis=1
            )
            output[row][column] = [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ]
    return output


def best_maximin(
    matrix: dict[str, dict[str, float | None]],
) -> dict[str, Any] | None:
    """Return the method with the strongest worst available objective utility."""
    candidates = []
    for method, scores in matrix.items():
        finite = [value for value in scores.values() if value is not None]
        if len(finite) == len(OBJECTIVES):
            candidates.append((min(finite), method))
    if not candidates:
        return None
    score, method = max(candidates)
    return {"method": method, "utility": score}


def write_outputs(out_dir: Path, result: dict[str, Any]) -> None:
    """Persist the compact report, matrices, boundaries, and examples."""
    (out_dir / "report.json").write_text(
        json.dumps(result["report"], indent=2) + "\n", encoding="utf-8"
    )
    write_matrix(out_dir / "objective_matrix.csv", result["matrix"])
    write_matrix(out_dir / "oracle_agreement.csv", result["agreements"])
    write_matrix(out_dir / "supervised_transfer.csv", result["transfer"])
    write_jsonl(out_dir / "boundaries.jsonl", result["boundaries"])
    write_jsonl(out_dir / "boundary_examples.jsonl", result["examples"])


def write_matrix(path: Path, matrix: dict[str, dict[str, Any]]) -> None:
    """Write one nested mapping as a rectangular CSV matrix."""
    columns = list(next(iter(matrix.values()))) if matrix else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *columns])
        writer.writeheader()
        for method, values in matrix.items():
            writer.writerow({"method": method, **values})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write records as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
