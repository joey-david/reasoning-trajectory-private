"""Question-disjoint evaluation of human-semantic token units."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.experiments.token_segmentation.data import (
    TraceKey,
    boundary_snippet,
    load_gold_targets,
    load_states,
    prepare_traces,
)
from src.experiments.token_segmentation.evaluation import (
    boundary_f1,
    matched_sentence_boundaries,
    normalized_utility,
    write_jsonl,
    write_matrix,
)
from src.experiments.token_segmentation.semantic_labels import (
    SemanticSpan,
    SemanticTrace,
    load_semantic_traces,
    semantic_rows,
)
from src.experiments.token_segmentation.semantic_decoding import (
    evaluate_semantic_labels,
)
from src.experiments.token_segmentation.signals import (
    OBJECTIVES,
    boundary_budget,
    fit_correctness_model,
    fit_projection,
    top_boundaries,
    transition_signals,
)
from src.experiments.token_segmentation.text_boundary import (
    compare_latent_text_auc,
    evaluate_text_boundary_baselines,
)


SEMANTIC_OBJECTIVES = (*OBJECTIVES, "semantic")


def run_semantic_token_segmentation(
    run_path: Path,
    *,
    labels_run: Path,
    gold_run: Path,
    updates_path: Path,
    layer: int = -1,
    min_segment_tokens: int = 4,
) -> Path:
    """Audit silver labels and evaluate their latent token structure."""
    semantic, audit = load_semantic_traces(
        labels_run / "token_windows.jsonl",
        labels_run / "labels" / "silver_labels.jsonl",
    )
    traces = prepare_traces(run_path, updates_path)
    trace_map = {trace.key: trace for trace in traces}
    missing = sorted(set(semantic) - set(trace_map))
    if missing:
        raise ValueError(f"{len(missing)} labeled traces absent from activation run")

    projection = fit_projection(run_path, traces, layer=layer)
    correctness = fit_correctness_model(run_path, traces, projection, layer=layer)
    gold_targets = load_gold_targets(gold_run, layer)
    boundary_model = fit_semantic_boundary_model(
        run_path, trace_map, semantic, projection, gold_targets, correctness, layer
    )
    residual_boundary_model = fit_semantic_boundary_model(
        run_path,
        trace_map,
        semantic,
        projection,
        gold_targets,
        correctness,
        layer,
        sentence_residual=True,
    )
    result = evaluate_semantic_boundaries(
        run_path,
        trace_map,
        semantic,
        projection,
        gold_targets,
        correctness,
        boundary_model,
        residual_boundary_model,
        layer,
        min_segment_tokens,
    )
    text_baselines = evaluate_text_boundary_baselines(trace_map, semantic)
    result["report"]["text_boundary_baselines"] = text_baselines
    result["report"]["latent_vs_text_boundary_auc"] = compare_latent_text_auc(
        result["report"], text_baselines
    )
    label_result = evaluate_semantic_labels(
        run_path, trace_map, semantic, projection, layer
    )
    output_name = (
        "semantic_token_segmentation"
        if min_segment_tokens == 4
        else f"semantic_token_segmentation_gap{min_segment_tokens}"
    )
    out_dir = run_path / "analysis" / "experiments" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "human_semantic_token_segmentation",
        "layer": layer,
        "split": "question-disjoint; every fifth sorted question held out",
        "minimum_segment_tokens": min_segment_tokens,
        "label_audit": audit,
        "sentence_relation": sentence_relation_summary(trace_map, semantic),
        "boundary_evaluation": result["report"],
        "semantic_label_decoding": label_result,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_matrix(out_dir / "objective_matrix.csv", result["matrix"])
    write_matrix(out_dir / "oracle_agreement.csv", result["agreements"])
    write_jsonl(out_dir / "reconciled_spans.jsonl", semantic_rows(semantic))
    write_jsonl(out_dir / "boundary_examples.jsonl", result["examples"])
    return out_dir / "report.json"


def fit_semantic_boundary_model(
    run_path: Path,
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    projection: Any,
    gold_targets: dict[str, np.ndarray],
    correctness_model: Any,
    layer: int,
    *,
    sentence_residual: bool = False,
) -> LogisticRegression:
    """Fit a balanced linear probe for semantic end boundaries."""
    features_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    rng = np.random.default_rng(0)
    for key, annotation in semantic.items():
        trace = trace_map[key]
        if not trace.train:
            continue
        states = load_states(run_path, trace, layer)
        signals, features = transition_signals(
            states,
            projection,
            gold_target=gold_targets.get(trace.sample_id),
            correctness_model=correctness_model,
            object_boundaries=trace.object_boundaries,
        )
        del signals
        boundaries = (
            _far_from(annotation.boundaries, trace.sentence_boundaries, tolerance=4)
            if sentence_residual
            else annotation.boundaries
        )
        labels = _boundary_labels(len(features), boundaries)
        positives = np.flatnonzero(labels)
        negatives = np.flatnonzero(labels == 0)
        negatives = rng.choice(
            negatives,
            size=min(len(negatives), 4 * len(positives)),
            replace=False,
        )
        keep = np.r_[positives, negatives]
        features_all.append(features[keep])
        labels_all.append(labels[keep])
    return LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=0
    ).fit(np.concatenate(features_all), np.concatenate(labels_all))


def evaluate_semantic_boundaries(
    run_path: Path,
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    projection: Any,
    gold_targets: dict[str, np.ndarray],
    correctness_model: Any,
    boundary_model: LogisticRegression,
    residual_boundary_model: LogisticRegression,
    layer: int,
    min_segment_tokens: int,
) -> dict[str, Any]:
    """Compare semantic boundaries with all existing token objectives."""
    methods = [
        "random",
        "latent_magnitude",
        "latent_cosine",
        "latent_curvature",
        "sentence_boundaries",
        *[f"oracle_{name}" for name in SEMANTIC_OBJECTIVES],
        "learned_semantic",
    ]
    utilities = {
        method: {objective: [] for objective in SEMANTIC_OBJECTIVES}
        for method in methods
    }
    agreements = {
        left: {right: [] for right in SEMANTIC_OBJECTIVES}
        for left in SEMANTIC_OBJECTIVES
    }
    aucs: dict[str, list[float]] = defaultdict(list)
    residual_aucs: dict[str, list[float]] = defaultdict(list)
    latent_auc_by_question: dict[str, float] = {}
    residual_auc_by_question: dict[str, float] = {}
    examples: list[dict[str, Any]] = []
    test_tokens = 0
    test_traces = 0
    for key, annotation in semantic.items():
        trace = trace_map[key]
        if trace.train or len(annotation.boundaries) < 2:
            continue
        states = load_states(run_path, trace, layer)
        signals, features = transition_signals(
            states,
            projection,
            gold_target=gold_targets.get(trace.sample_id),
            correctness_model=correctness_model,
            object_boundaries=trace.object_boundaries,
        )
        signals["semantic"] = _boundary_labels(
            len(features), annotation.boundaries
        ).astype(np.float32)
        budget = boundary_budget(trace, len(features), min_segment_tokens)
        oracles = {
            name: top_boundaries(
                signals[name], budget, min_segment_tokens=min_segment_tokens
            )
            for name in SEMANTIC_OBJECTIVES
        }
        learned_scores = boundary_model.predict_proba(features)[:, 1]
        residual_scores = residual_boundary_model.predict_proba(features)[:, 1]
        selected = {
            "random": top_boundaries(
                np.random.default_rng(trace.seed).random(len(features)),
                budget,
                min_segment_tokens=min_segment_tokens,
            ),
            "latent_magnitude": top_boundaries(
                signals["latent_magnitude"],
                budget,
                min_segment_tokens=min_segment_tokens,
            ),
            "latent_cosine": top_boundaries(
                signals["latent_cosine"],
                budget,
                min_segment_tokens=min_segment_tokens,
            ),
            "latent_curvature": top_boundaries(
                signals["latent_curvature"],
                budget,
                min_segment_tokens=min_segment_tokens,
            ),
            "sentence_boundaries": matched_sentence_boundaries(
                trace.sentence_boundaries,
                len(features),
                budget,
                min_segment_tokens,
            ),
            "learned_semantic": top_boundaries(
                learned_scores,
                budget,
                min_segment_tokens=min_segment_tokens,
            ),
            **{f"oracle_{name}": values for name, values in oracles.items()},
        }
        labels = signals["semantic"].astype(np.int8)
        if len(np.unique(labels)) == 2:
            latent_auc = float(roc_auc_score(labels, learned_scores))
            aucs["learned_semantic"].append(latent_auc)
            latent_auc_by_question[trace.sample_id] = latent_auc
            for name in ("latent_magnitude", "latent_cosine", "latent_curvature"):
                aucs[name].append(roc_auc_score(labels, signals[name]))
        residual_labels = _boundary_labels(
            len(features),
            _far_from(
                annotation.boundaries, trace.sentence_boundaries, tolerance=4
            ),
        )
        if len(np.unique(residual_labels)) == 2:
            residual_auc = float(roc_auc_score(residual_labels, residual_scores))
            residual_aucs["learned_semantic"].append(residual_auc)
            residual_auc_by_question[trace.sample_id] = residual_auc
            for name in ("latent_magnitude", "latent_cosine", "latent_curvature"):
                residual_aucs[name].append(
                    roc_auc_score(residual_labels, signals[name])
                )
        for method, boundaries in selected.items():
            for objective in SEMANTIC_OBJECTIVES:
                score = normalized_utility(
                    signals[objective], boundaries, oracles[objective]
                )
                if np.isfinite(score):
                    utilities[method][objective].append(float(score))
        for left in SEMANTIC_OBJECTIVES:
            for right in SEMANTIC_OBJECTIVES:
                agreements[left][right].append(
                    boundary_f1(oracles[left], oracles[right], tolerance=4)
                )
        if len(examples) < 30:
            positive_examples = [
                boundary
                for boundary in oracles["semantic"]
                if signals["semantic"][boundary] > 0
            ][:3]
            for boundary in positive_examples:
                examples.append(
                    {
                        "sample_id": trace.sample_id,
                        "seed": trace.seed,
                        "token_boundary": int(boundary),
                        "semantic_labels": _labels_near(
                            annotation.spans, int(boundary)
                        ),
                        "snippet": boundary_snippet(trace, int(boundary), radius=100),
                    }
                )
        test_tokens += len(states)
        test_traces += 1
    matrix = _mean_nested(utilities)
    rng = np.random.default_rng(0)
    return {
        "report": {
            "test_traces": test_traces,
            "test_tokens": test_tokens,
            "common_boundary_budget": "verified symbolic update count, capped at 64",
            "semantic_boundary_auc": {
                name: float(np.mean(values)) for name, values in aucs.items()
            },
            "semantic_boundary_auc_by_question": latent_auc_by_question,
            "semantic_boundary_auc_question_bootstrap_95ci": {
                name: _bootstrap_interval(values, rng) for name, values in aucs.items()
            },
            "non_sentence_semantic_boundary_auc": {
                name: float(np.mean(values))
                for name, values in residual_aucs.items()
            },
            "non_sentence_semantic_boundary_auc_by_question": residual_auc_by_question,
            "non_sentence_semantic_boundary_auc_question_bootstrap_95ci": {
                name: _bootstrap_interval(values, rng)
                for name, values in residual_aucs.items()
            },
            "objective_matrix": matrix,
            "objective_matrix_question_bootstrap_95ci": {
                method: {
                    objective: _bootstrap_interval(values, rng)
                    for objective, values in columns.items()
                }
                for method, columns in utilities.items()
            },
            "oracle_boundary_f1_tolerance_4": _mean_nested(agreements),
            "best_maximin_method": _best_maximin(matrix),
        },
        "matrix": matrix,
        "agreements": _mean_nested(agreements),
        "examples": examples,
    }


def sentence_relation_summary(
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    *,
    tolerance: int = 4,
) -> dict[str, Any]:
    """Quantify how much the semantic ontology departs from sentence parsing."""
    output: dict[str, Any] = {}
    for split_name, train_value in (("all", None), ("held_out", False)):
        semantic_total = semantic_near = sentence_total = sentence_near = 0
        crossing = total_spans = 0
        trace_f1: list[float] = []
        for key, annotation in semantic.items():
            trace = trace_map[key]
            if train_value is not None and trace.train != train_value:
                continue
            semantic_total += len(annotation.boundaries)
            sentence_total += len(trace.sentence_boundaries)
            semantic_near += len(
                annotation.boundaries[
                    _distance_to(annotation.boundaries, trace.sentence_boundaries)
                    <= tolerance
                ]
            )
            sentence_near += len(
                trace.sentence_boundaries[
                    _distance_to(trace.sentence_boundaries, annotation.boundaries)
                    <= tolerance
                ]
            )
            crossing += sum(
                np.any(
                    (trace.sentence_boundaries >= span.token_start)
                    & (trace.sentence_boundaries < span.token_end)
                )
                for span in annotation.spans
            )
            total_spans += len(annotation.spans)
            trace_f1.append(
                boundary_f1(
                    annotation.boundaries,
                    trace.sentence_boundaries,
                    tolerance=tolerance,
                )
            )
        output[split_name] = {
            "semantic_boundaries": semantic_total,
            "sentence_boundaries": sentence_total,
            "semantic_boundary_sentence_aligned": semantic_near
            / max(semantic_total, 1),
            "sentence_boundary_semantic_aligned": sentence_near
            / max(sentence_total, 1),
            "mean_trace_boundary_f1": float(np.mean(trace_f1)),
            "spans_crossing_sentence_boundaries": crossing / max(total_spans, 1),
        }
    return output


def _boundary_labels(length: int, boundaries: np.ndarray) -> np.ndarray:
    """Create a binary transition target from end-token boundaries."""
    labels = np.zeros(length, dtype=np.int8)
    valid = boundaries[(boundaries >= 0) & (boundaries < length)]
    labels[valid] = 1
    return labels


def _far_from(
    values: np.ndarray,
    references: np.ndarray,
    *,
    tolerance: int,
) -> np.ndarray:
    """Return boundaries farther than a tolerance from every reference."""
    return values[_distance_to(values, references) > tolerance]


def _distance_to(values: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Return each boundary's distance to its nearest reference."""
    if not len(values):
        return np.empty(0, dtype=np.float32)
    if not len(references):
        return np.full(len(values), np.inf, dtype=np.float32)
    return np.min(np.abs(values[:, None] - references[None, :]), axis=1)


def _labels_near(spans: list[SemanticSpan], boundary: int) -> list[str]:
    """Return semantic labels ending near one boundary."""
    return sorted(
        {span.label for span in spans if abs(span.token_end - boundary) <= 2}
    )


def _mean_nested(values: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
    """Average each cell in a nested metric mapping."""
    return {
        row: {
            column: float(np.mean(cell)) if cell else None
            for column, cell in columns.items()
        }
        for row, columns in values.items()
    }


def _bootstrap_interval(
    values: list[float],
    rng: np.random.Generator,
    *,
    samples: int = 2000,
) -> list[float] | None:
    """Bootstrap a mean over held-out questions."""
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _best_maximin(matrix: dict[str, dict[str, float | None]]) -> dict[str, Any]:
    """Return the method with the largest worst-objective utility."""
    values = [
        (min(score for score in row.values() if score is not None), method)
        for method, row in matrix.items()
        if all(score is not None for score in row.values())
    ]
    score, method = max(values)
    return {"method": method, "utility": score}
