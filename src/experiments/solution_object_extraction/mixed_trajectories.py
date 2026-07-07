"""Real mixed-success GSM-Symbolic trajectory analysis and reranking."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from reasoning_trajectory.token_alignment import build_token_spans
from src.experiments.common import balanced_generation_rows
from src.experiments.symbolic import extract_symbolic_updates
from src.runtime.artifact_store import load_hidden_states_npz

from .projections import project
from .retrieval import normalize_rows


CHECKPOINTS = (0.25, 0.5, 0.75, 1.0)


def analyze_mixed_trajectories(
    *,
    source_run: Path,
    projection_mean: np.ndarray,
    projection_basis: np.ndarray,
    layer: int,
    per_sample: int,
    max_questions: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze real correct/incorrect trajectories and rerank their answers."""
    rows = balanced_generation_rows(source_run, per_sample=per_sample)
    by_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[str(row["sample_id"])].append(row)
    mixed_questions = sorted(
        question
        for question, question_rows in by_question.items()
        if {bool(row["is_correct"]) for row in question_rows} == {False, True}
    )[:max_questions]
    selected = [
        row for question in mixed_questions for row in by_question[question]
    ]
    spans = build_token_spans(source_run, selected)
    traces = []
    for row, token_spans in tqdm(
        zip(selected, spans, strict=True),
        total=len(selected),
        desc="real object trajectories",
        unit="trace",
    ):
        states, layers = load_hidden_states_npz(
            source_run / str(row["hidden_states_file"])
        )
        layer_col = resolve_layer_column(layers, layer)
        count = min(len(row.get("generated_token_ids", [])), states.shape[0])
        updates = extract_symbolic_updates(
            str(row.get("produced_text", "")),
            token_spans,
            token_count=count,
        )
        raw_sequence = []
        for update in updates:
            state_index = min(max(int(update.token_end) + 1, 0), count - 1)
            raw_sequence.append(states[state_index, layer_col].astype(np.float32))
        if raw_sequence:
            z_sequence = project(
                np.stack(raw_sequence), projection_mean, projection_basis
            )
        else:
            z_sequence = np.empty(
                (0, projection_basis.shape[0]), dtype=np.float32
            )
        traces.append(
            {
                "sample_id": str(row["sample_id"]),
                "seed": int(row["seed"]),
                "is_correct": bool(row["is_correct"]),
                "produced_answer": row.get("produced_answer"),
                "updates": updates,
                "z_sequence": z_sequence,
            }
        )
    trajectory = correctness_report(traces, projection_basis.shape[0])
    reranking = reranking_report(traces)
    trajectory["source_run"] = source_run.as_posix()
    trajectory["per_sample_cap"] = per_sample
    trajectory["questions"] = len(mixed_questions)
    reranking["source_run"] = source_run.as_posix()
    return trajectory, reranking


def resolve_layer_column(layers: list[int], requested: int) -> int:
    """Resolve an exact saved layer or the final saved layer for ``-1``."""
    if requested in layers:
        return layers.index(requested)
    if requested == -1:
        return len(layers) - 1
    raise ValueError(f"Layer {requested} is absent from captured layers {layers}")


def correctness_report(
    traces: list[dict[str, Any]],
    dimension: int,
) -> dict[str, Any]:
    """Compute question-grouped probe and gold-prototype divergence metrics."""
    groups = np.asarray([trace["sample_id"] for trace in traces], dtype=str)
    labels = np.asarray([int(trace["is_correct"]) for trace in traces], dtype=int)
    results = []
    for fraction in CHECKPOINTS:
        features = np.stack(
            [trajectory_features(trace, dimension, fraction) for trace in traces]
        )
        folds = min(5, len(set(groups)))
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ),
        )
        probabilities = cross_val_predict(
            model,
            features,
            labels,
            groups=groups,
            cv=GroupKFold(n_splits=folds),
            method="predict_proba",
        )[:, 1]
        checkpoint_vectors = np.stack(
            [checkpoint_state(trace, dimension, fraction) for trace in traces]
        )
        consensus_scores = unsupervised_consensus_scores(
            checkpoint_vectors, groups
        )
        results.append(
            {
                "checkpoint": fraction,
                "question_grouped_correctness_auc": float(
                    roc_auc_score(labels, probabilities)
                ),
                "unsupervised_object_consensus_auc": float(
                    roc_auc_score(labels, consensus_scores)
                ),
                "feature_dim": int(features.shape[1]),
            }
        )
    divergence = prototype_divergence(traces, dimension)
    earliest = next(
        (
            result["checkpoint"]
            for result in results
            if result["unsupervised_object_consensus_auc"] >= 0.65
        ),
        None,
    )
    return {
        "traces": len(traces),
        "class_counts": dict(Counter(labels.tolist())),
        "checkpoint_results": results,
        "earliest_unsupervised_consensus_checkpoint_at_auc_0_65": earliest,
        "gold_prototype_divergence": divergence,
        "failure_modes": failure_mode_counts(traces),
    }


def trajectory_features(
    trace: dict[str, Any],
    dimension: int,
    fraction: float,
) -> np.ndarray:
    """Build fixed-width prefix features from projected object edits."""
    sequence = trace["z_sequence"]
    if len(sequence):
        keep = max(1, int(np.ceil(len(sequence) * fraction)))
        prefix = sequence[:keep]
        mean = prefix.mean(axis=0)
        last = prefix[-1]
        variance = prefix.var(axis=0)
    else:
        mean = last = variance = np.zeros(dimension, dtype=np.float32)
        keep = 0
    updates = trace["updates"][:keep]
    operations = [update.operation_signature for update in updates]
    extracts = [update for update in updates if update.operator == "EXTRACT"]
    produced = numeric_answer(trace.get("produced_answer"))
    extract_match = bool(
        extracts
        and produced is not None
        and np.isclose(extracts[-1].value, produced)
    )
    scalars = np.asarray(
        [
            keep / max(len(trace["updates"]), 1),
            len(set(operations)) / max(len(operations), 1),
            float(extract_match),
            float(bool(updates)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([mean, last, variance, scalars])


def prototype_divergence(
    traces: list[dict[str, Any]],
    dimension: int,
) -> list[dict[str, Any]]:
    """Describe distance to same-question correct object prototypes."""
    output = []
    for fraction in CHECKPOINTS:
        vectors = np.stack(
            [checkpoint_state(trace, dimension, fraction) for trace in traces]
        )
        normalized = normalize_rows(vectors)
        prototypes = {}
        for question in sorted({trace["sample_id"] for trace in traces}):
            indices = [
                index
                for index, trace in enumerate(traces)
                if trace["sample_id"] == question and trace["is_correct"]
            ]
            prototypes[question] = normalize_rows(
                normalized[indices].mean(axis=0, keepdims=True)
            )[0]
        distances = np.asarray(
            [
                1.0 - normalized[index] @ prototypes[trace["sample_id"]]
                for index, trace in enumerate(traces)
            ]
        )
        labels = np.asarray([int(trace["is_correct"]) for trace in traces])
        output.append(
            {
                "checkpoint": fraction,
                "correct_mean_distance": float(distances[labels == 1].mean()),
                "incorrect_mean_distance": float(distances[labels == 0].mean()),
                "descriptive_correctness_auc": float(
                    roc_auc_score(labels, -distances)
                ),
                "note": "descriptive; correct prototypes use labels",
            }
        )
    return output


def unsupervised_consensus_scores(
    vectors: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    """Score proximity to each question's leave-one-out object centroid."""
    normalized = normalize_rows(vectors)
    scores = np.zeros(len(vectors), dtype=np.float32)
    for question in sorted(set(groups)):
        indices = np.flatnonzero(groups == question)
        for index in indices:
            others = indices[indices != index]
            if not len(others):
                continue
            centroid = normalize_rows(
                normalized[others].mean(axis=0, keepdims=True)
            )[0]
            scores[index] = float(normalized[index] @ centroid)
    return scores


def checkpoint_state(
    trace: dict[str, Any], dimension: int, fraction: float
) -> np.ndarray:
    """Return the final projected edit at a normalized trajectory checkpoint."""
    sequence = trace["z_sequence"]
    if not len(sequence):
        return np.zeros(dimension, dtype=np.float32)
    index = min(int(np.ceil(len(sequence) * fraction)) - 1, len(sequence) - 1)
    return sequence[index]


def failure_mode_counts(traces: list[dict[str, Any]]) -> dict[str, int]:
    """Count deterministic object-trajectory failure signatures."""
    counts: Counter[str] = Counter()
    for trace in traces:
        if trace["is_correct"]:
            continue
        updates = trace["updates"]
        if not updates:
            counts["no_verified_object_edits"] += 1
            continue
        if not any(update.operator == "EXTRACT" for update in updates):
            counts["missing_extract"] += 1
        produced = numeric_answer(trace.get("produced_answer"))
        if produced is not None and not any(
            np.isclose(update.value, produced) for update in updates
        ):
            counts["answer_not_in_verified_graph"] += 1
        if len({update.graph_signature for update in updates}) < len(updates) / 2:
            counts["repeated_or_nonmutating_edits"] += 1
    return dict(counts)


def reranking_report(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare answer majority with unsupervised object/graph reranking."""
    by_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_question[trace["sample_id"]].append(trace)
    methods: defaultdict[str, list[bool]] = defaultdict(list)
    for question_traces in by_question.values():
        answer_counts = Counter(
            str(trace["produced_answer"])
            for trace in question_traces
            if trace["produced_answer"] is not None
        )
        if not answer_counts:
            continue
        majority = answer_counts.most_common(1)[0][0]
        answer_groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, trace in enumerate(question_traces):
            answer_groups[str(trace["produced_answer"])].append(index)
        majority_candidates = [question_traces[index] for index in answer_groups[majority]]
        methods["answer_majority"].append(
            any(trace["is_correct"] for trace in majority_candidates)
        )
        final_vectors = np.stack(
            [
                checkpoint_state(
                    trace,
                    trace["z_sequence"].shape[1]
                    if trace["z_sequence"].ndim == 2
                    else 1,
                    1.0,
                )
                for trace in question_traces
            ]
        )
        normalized = normalize_rows(final_vectors)
        similarity = normalized @ normalized.T
        validity = np.asarray(
            [graph_validity(trace) for trace in question_traces], dtype=float
        )
        answer_scores: defaultdict[str, dict[str, float]] = defaultdict(dict)
        for answer, indices in answer_groups.items():
            if len(indices) > 1:
                submatrix = similarity[np.ix_(indices, indices)]
                cohesion = float(
                    (submatrix.sum() - len(indices))
                    / (len(indices) * (len(indices) - 1))
                )
            else:
                cohesion = -1.0
            frequency = len(indices) / len(question_traces)
            graph_score = float(validity[indices].mean())
            answer_scores[answer] = {
                "object_cluster_cohesion": cohesion,
                "graph_validity": graph_score,
                "hybrid_answer_object": (
                    frequency + 0.25 * cohesion + 0.25 * graph_score
                ),
            }
        for name in (
            "object_cluster_cohesion",
            "graph_validity",
            "hybrid_answer_object",
        ):
            winning_answer = max(
                answer_scores,
                key=lambda answer: (
                    answer_scores[answer][name],
                    len(answer_groups[answer]),
                    answer,
                ),
            )
            methods[name].append(
                any(
                    question_traces[index]["is_correct"]
                    for index in answer_groups[winning_answer]
                )
            )
    return {
        "questions": len(by_question),
        "accuracy": {
            method: float(np.mean(values)) for method, values in methods.items()
        },
        "selection": (
            "answers are clusters; object score is within-answer trajectory "
            "cohesion, graph validity uses deterministic update/answer agreement, "
            "and the fixed hybrid is frequency + 0.25*cohesion + 0.25*validity"
        ),
    }


def graph_validity(trace: dict[str, Any]) -> float:
    """Score whether verified edits support the emitted answer."""
    updates = trace["updates"]
    if not updates:
        return 0.0
    produced = numeric_answer(trace.get("produced_answer"))
    support = produced is not None and any(
        np.isclose(update.value, produced) for update in updates
    )
    has_operation = any(update.operator in {"OPERATE", "VERIFY"} for update in updates)
    return float(support) + 0.5 * float(has_operation)


def numeric_answer(value: Any) -> float | None:
    """Parse a stored numeric answer conservatively."""
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
