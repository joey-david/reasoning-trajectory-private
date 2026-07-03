"""Token-transition signals and compact supervised boundary models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from src.experiments.token_segmentation.data import TraceMeta, load_states


OBJECTIVES = ("answer", "object", "correctness", "compression")


@dataclass(slots=True)
class SignalModels:
    """Training-only transforms used to score held-out token boundaries."""

    projection: PCA
    correctness: LogisticRegression | None
    boundary_models: dict[str, LogisticRegression]


def fit_projection(
    run_path: Path,
    traces: list[TraceMeta],
    *,
    layer: int = -1,
    dimensions: int = 32,
    samples_per_trace: int = 24,
) -> PCA:
    """Fit a question-disjoint PCA from evenly sampled training tokens."""
    samples: list[np.ndarray] = []
    for trace in traces:
        if not trace.train:
            continue
        states = load_states(run_path, trace, layer)
        indices = np.linspace(
            0, len(states) - 1, min(samples_per_trace, len(states)), dtype=int
        )
        samples.append(states[indices])
    matrix = np.concatenate(samples)
    dimensions = min(dimensions, matrix.shape[0] - 1, matrix.shape[1])
    return PCA(n_components=dimensions, whiten=True, random_state=0).fit(matrix)


def fit_correctness_model(
    run_path: Path,
    traces: list[TraceMeta],
    projection: PCA,
    *,
    layer: int = -1,
    samples_per_trace: int = 24,
) -> LogisticRegression | None:
    """Fit a linear token-state probe for final trace correctness."""
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for trace in traces:
        if not trace.train:
            continue
        states = load_states(run_path, trace, layer)
        indices = np.linspace(
            0, len(states) - 1, min(samples_per_trace, len(states)), dtype=int
        )
        features.append(projection.transform(states[indices]))
        labels.append(np.full(len(indices), int(trace.is_correct), dtype=np.int8))
    target = np.concatenate(labels)
    if len(np.unique(target)) < 2:
        return None
    return LogisticRegression(
        class_weight="balanced", max_iter=500, random_state=0
    ).fit(np.concatenate(features), target)


def transition_signals(
    states: np.ndarray,
    projection: PCA,
    *,
    gold_target: np.ndarray | None,
    correctness_model: LogisticRegression | None,
    object_boundaries: np.ndarray,
    window: int = 8,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute objective curves and model features at every token transition."""
    projected = projection.transform(states)
    delta = np.diff(states, axis=0)
    projected_delta = np.diff(projected, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    state_cosine = cosine_distance(states[:-1], states[1:])
    curvature = np.zeros(len(delta), dtype=np.float32)
    if len(delta) > 1:
        curvature[1:] = cosine_distance(delta[:-1], delta[1:])
    compression = local_mean_shift(projected, window)

    answer = np.zeros(len(delta), dtype=np.float32)
    if gold_target is not None and gold_target.shape[-1] == states.shape[-1]:
        alignment = cosine_similarity(states, gold_target[None, :])
        answer = np.maximum(np.diff(alignment), 0.0)

    correctness = np.zeros(len(delta), dtype=np.float32)
    if correctness_model is not None:
        probability = correctness_model.predict_proba(projected)[:, 1]
        correctness = np.abs(np.diff(probability)).astype(np.float32)

    object_score = np.zeros(len(delta), dtype=np.float32)
    valid = object_boundaries[
        (object_boundaries >= 0) & (object_boundaries < len(object_score))
    ]
    object_score[valid] = 1.0
    position = np.linspace(0.0, 1.0, len(delta), dtype=np.float32)
    features = np.column_stack(
        [
            projected[:-1],
            projected_delta,
            robust_scale(delta_norm),
            robust_scale(state_cosine),
            robust_scale(curvature),
            position,
        ]
    ).astype(np.float32)
    return (
        {
            "answer": answer,
            "object": object_score,
            "correctness": correctness,
            "compression": compression,
            "latent_magnitude": delta_norm,
            "latent_cosine": state_cosine,
            "latent_curvature": curvature,
        },
        features,
    )


def fit_boundary_models(
    run_path: Path,
    traces: list[TraceMeta],
    projection: PCA,
    gold_targets: dict[str, np.ndarray],
    correctness_model: LogisticRegression | None,
    *,
    layer: int = -1,
    negative_ratio: int = 4,
    min_segment_tokens: int = 4,
) -> dict[str, LogisticRegression]:
    """Train one strong linear boundary adversary per objective."""
    feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in OBJECTIVES}
    label_rows: dict[str, list[np.ndarray]] = {name: [] for name in OBJECTIVES}
    rng = np.random.default_rng(0)
    for trace in traces:
        if not trace.train or len(trace.object_boundaries) < 2:
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
        for objective in OBJECTIVES:
            positives = top_boundaries(
                signals[objective], budget, min_segment_tokens=min_segment_tokens
            )
            labels = np.zeros(len(features), dtype=np.int8)
            labels[positives] = 1
            negatives = np.flatnonzero(labels == 0)
            keep_negative = rng.choice(
                negatives,
                size=min(len(negatives), negative_ratio * len(positives)),
                replace=False,
            )
            keep = np.r_[positives, keep_negative]
            feature_rows[objective].append(features[keep])
            label_rows[objective].append(labels[keep])
    models: dict[str, LogisticRegression] = {}
    for objective in OBJECTIVES:
        labels = np.concatenate(label_rows[objective])
        models[objective] = LogisticRegression(
            class_weight="balanced", max_iter=500, random_state=0
        ).fit(np.concatenate(feature_rows[objective]), labels)
    return models


def boundary_budget(
    trace: TraceMeta,
    transition_count: int,
    min_segment_tokens: int = 4,
) -> int:
    """Match every method to the number of verified object-update completions."""
    feasible = max(1, (transition_count + 1) // min_segment_tokens - 1)
    return max(1, min(len(trace.object_boundaries), 64, feasible))


def top_boundaries(
    scores: np.ndarray,
    count: int,
    *,
    min_segment_tokens: int = 4,
) -> np.ndarray:
    """Return the exact fixed-budget optimum under a minimum segment length."""
    count = min(max(int(count), 0), len(scores))
    if not count:
        return np.empty(0, dtype=np.int32)
    lower = min_segment_tokens - 1
    upper = len(scores) - min_segment_tokens
    positions = np.arange(lower, upper + 1, dtype=np.int32)
    weights = np.nan_to_num(
        np.asarray(scores, dtype=np.float64)[positions],
        nan=-np.inf,
    )
    choices = np.zeros((count + 1, len(positions)), dtype=bool)
    previous = np.zeros(len(positions), dtype=np.float64)
    for selected_count in range(1, count + 1):
        shifted = np.full(len(positions), -np.inf, dtype=np.float64)
        if selected_count == 1:
            shifted[:] = 0.0
        elif min_segment_tokens < len(positions):
            shifted[min_segment_tokens:] = previous[:-min_segment_tokens]
        candidate = weights + shifted
        current = np.maximum.accumulate(candidate)
        prior_best = np.r_[-np.inf, current[:-1]]
        choices[selected_count] = candidate > prior_best
        previous = current

    selected: list[int] = []
    position_index = len(positions) - 1
    selected_count = count
    while selected_count and position_index >= 0:
        if choices[selected_count, position_index]:
            selected.append(int(positions[position_index]))
            position_index -= min_segment_tokens
            selected_count -= 1
        else:
            position_index -= 1
    if selected_count:
        raise ValueError("Boundary budget is infeasible for the segment constraint")
    return np.asarray(sorted(selected), dtype=np.int32)


def local_mean_shift(values: np.ndarray, window: int) -> np.ndarray:
    """Measure the mean-state difference across each candidate split."""
    count = len(values) - 1
    output = np.zeros(count, dtype=np.float32)
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    for boundary in range(count):
        left_start = max(0, boundary + 1 - window)
        right_end = min(len(values), boundary + 1 + window)
        left = (prefix[boundary + 1] - prefix[left_start]) / (
            boundary + 1 - left_start
        )
        right = (prefix[right_end] - prefix[boundary + 1]) / (
            right_end - boundary - 1
        )
        output[boundary] = np.linalg.norm(right - left)
    return output


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine distance."""
    return 1.0 - cosine_similarity(left, right)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine similarity with stable zero handling."""
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-8)


def robust_scale(values: np.ndarray) -> np.ndarray:
    """Median-center and MAD-scale one transition signal."""
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-6)
