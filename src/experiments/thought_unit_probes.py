"""Supervised boundary adversaries and projection-space coherence tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.objective_segmentation import append_objective_identity
from src.experiments.sentence_lattice import (
    optimal_partition,
    partition_cost,
    random_boundaries,
    squared_error_costs,
    top_boundaries,
)
from src.experiments.thought_unit_cache import trace_view
from src.experiments.thought_unit_types import (
    OBJECTIVES,
    ORACLE_NAMES,
    PRIMARY_FRACTION,
)


def boundary_features(values: np.ndarray) -> np.ndarray:
    """Describe each transition with signed and absolute coordinate changes.

    Args:
        values: Values to summarize or transform.

    Returns:
        The resulting numeric array or tensor.
    """
    delta = np.diff(values, axis=0)
    position = np.linspace(0.0, 1.0, len(delta), dtype=np.float32)[:, None]
    magnitude = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.concatenate([delta, np.abs(delta), magnitude, position], axis=1)


def evaluate_supervised_boundaries(
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Train objective-specific change-point adversaries and test transfer.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.
        primary_partitions: Reference objective-specific partitions keyed by trace.

    Returns:
        The resulting keyed records or metrics.
    """
    spaces = {
        "raw": "raw",
        "pca_whitened": "pca",
        "gram_spectrum": "gram",
        "h4_operation": "h4",
    }
    rows: list[dict[str, Any]] = []
    diagonal: dict[str, dict[str, float]] = defaultdict(dict)
    conditioned_diagonal: dict[str, dict[str, float]] = defaultdict(dict)
    transfer_ratios: dict[str, dict[str, float]] = defaultdict(dict)
    for space, attribute in spaces.items():
        train_x: list[np.ndarray] = []
        test_x: list[np.ndarray] = []
        train_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        test_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        test_slices: list[tuple[int, int, int]] = []
        cursor = 0
        for index in selected_indices:
            trace = trace_view(cache, index)
            features = boundary_features(getattr(trace, attribute))
            labels = {
                objective: boundary_labels(
                    len(features), primary_partitions[index][ORACLE_NAMES[objective]]
                )
                for objective in OBJECTIVES
            }
            if trace.train:
                train_x.append(features)
                for objective, values in labels.items():
                    train_labels[objective].append(values)
            else:
                test_x.append(features)
                for objective, values in labels.items():
                    test_labels[objective].append(values)
                positives = len(primary_partitions[index][ORACLE_NAMES["answer"]])
                test_slices.append((cursor, cursor + len(features), positives))
                cursor += len(features)
        x_train = np.concatenate(train_x).astype(np.float32)
        x_test = np.concatenate(test_x).astype(np.float32)
        y_train = {
            objective: np.concatenate(values)
            for objective, values in train_labels.items()
        }
        y_test = {
            objective: np.concatenate(values)
            for objective, values in test_labels.items()
        }
        for trained_on in OBJECTIVES:
            model = fit_boundary_model(
                x_train,
                y_train[trained_on],
                nonlinear=space != "raw",
            )
            probabilities = model.predict_proba(x_test)[:, 1]
            for evaluated_on in OBJECTIVES:
                expected = y_test[evaluated_on]
                predicted = matched_probability_labels(probabilities, test_slices)
                auc = float(roc_auc_score(expected, probabilities))
                average_precision = float(
                    average_precision_score(expected, probabilities)
                )
                f1 = float(f1_score(expected, predicted))
                rows.append(
                    {
                        "model": "objective_specific",
                        "space": space,
                        "trained_on": trained_on,
                        "evaluated_on": evaluated_on,
                        "roc_auc": auc,
                        "average_precision": average_precision,
                        "matched_budget_f1": f1,
                    }
                )
                if trained_on == evaluated_on:
                    diagonal[space][trained_on] = auc
            own_auc = diagonal[space][trained_on]
            cross = [
                row["roc_auc"]
                for row in rows
                if row["space"] == space
                and row["trained_on"] == trained_on
                and row["evaluated_on"] != trained_on
            ]
            transfer_ratios[space][trained_on] = float(np.mean(cross) / own_auc)

        objective_names = list(OBJECTIVES)
        if space == "raw":
            conditioned_model = fit_conditioned_linear_model(
                x_train,
                np.column_stack([y_train[objective] for objective in objective_names]),
            )
        else:
            conditioned_x = np.concatenate(
                [
                    append_objective_identity(
                        x_train,
                        objective_index,
                        len(objective_names),
                    )
                    for objective_index in range(len(objective_names))
                ]
            )
            conditioned_y = np.concatenate(
                [y_train[objective] for objective in objective_names]
            )
            conditioned_model = fit_boundary_model(
                conditioned_x,
                conditioned_y,
                nonlinear=True,
            )
        for objective_index, objective in enumerate(objective_names):
            if space == "raw":
                probabilities = conditioned_model.predict_proba(
                    x_test,
                    objective_index,
                )
            else:
                probabilities = conditioned_model.predict_proba(
                    append_objective_identity(
                        x_test,
                        objective_index,
                        len(objective_names),
                    )
                )[:, 1]
            expected = y_test[objective]
            predicted = matched_probability_labels(probabilities, test_slices)
            auc = float(roc_auc_score(expected, probabilities))
            conditioned_diagonal[space][objective] = auc
            rows.append(
                {
                    "model": "objective_conditioned",
                    "space": space,
                    "trained_on": "all_with_objective_id",
                    "evaluated_on": objective,
                    "roc_auc": auc,
                    "average_precision": float(
                        average_precision_score(expected, probabilities)
                    ),
                    "matched_budget_f1": float(f1_score(expected, predicted)),
                }
            )
    return {
        "model": (
            "objective-specific baselines plus conditioned models; raw uses a "
            "jointly optimized multi-head linear model, projected spaces use one "
            "histogram-gradient-boosting model with an objective ID"
        ),
        "conditioning_scope": {
            "raw": (
                "shared normalization with one linear head per objective; this "
                "tests one callable conditional model but not parameter sharing"
            ),
            "projected_spaces": (
                "one shared nonlinear estimator conditioned by objective one-hot"
            ),
        },
        "primary_boundary_fraction": PRIMARY_FRACTION,
        "diagonal_roc_auc": dict(diagonal),
        "objective_conditioned_diagonal_roc_auc": dict(conditioned_diagonal),
        "objective_conditioned_minus_specific_auc": {
            space: {
                objective: conditioned_diagonal[space][objective]
                - diagonal[space][objective]
                for objective in OBJECTIVES
            }
            for space in spaces
        },
        "mean_cross_to_in_domain_auc_ratio": dict(transfer_ratios),
        "rows": rows,
    }


@dataclass(slots=True)
class ConditionedLinearBoundaryModel:
    """Store one normalized linear boundary head per objective."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: np.ndarray

    def predict_proba(
        self,
        features: np.ndarray,
        objective_index: int,
        *,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Predict one objective's boundary probabilities in bounded batches.

        Args:
            features: Boundary feature matrix.
            objective_index: Objective head selected at inference.
            batch_size: Maximum rows transformed at once.

        Returns:
            Positive-boundary probability for each feature row.
        """
        output = np.empty(len(features), dtype=np.float32)
        for start in range(0, len(features), batch_size):
            batch = (
                features[start : start + batch_size].astype(np.float32) - self.mean
            ) / self.scale
            logits = batch @ self.weights[objective_index] + self.bias[objective_index]
            output[start : start + len(batch)] = 1.0 / (
                1.0 + np.exp(-np.clip(logits, -30.0, 30.0))
            )
        return output


def fit_conditioned_linear_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 12,
    batch_size: int = 256,
) -> ConditionedLinearBoundaryModel:
    """Fit all objective-specific linear heads in one batched optimization.

    Args:
        features: Shared raw boundary feature matrix.
        labels: Binary row-by-objective boundary labels.
        epochs: Number of shuffled gradient passes.
        batch_size: Maximum rows used by one gradient update.

    Returns:
        Joint objective-conditioned linear boundary model.
    """
    values = np.asarray(features, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    weights = np.zeros((targets.shape[1], values.shape[1]), dtype=np.float32)
    bias = np.zeros(targets.shape[1], dtype=np.float32)
    positives = targets.sum(axis=0)
    positive_weight = (len(targets) - positives) / np.maximum(positives, 1.0)
    rng = np.random.default_rng(42)

    for epoch in range(epochs):
        order = rng.permutation(len(values))
        learning_rate = 0.08 / np.sqrt(epoch + 1)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = (values[indices] - mean) / scale
            expected = targets[indices]
            logits = batch @ weights.T + bias
            predicted = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            sample_weight = np.where(expected > 0.5, positive_weight, 1.0)
            error = (predicted - expected) * sample_weight
            weights -= learning_rate * (error.T @ batch / len(batch) + 1e-4 * weights)
            bias -= learning_rate * error.mean(axis=0)
    return ConditionedLinearBoundaryModel(mean, scale, weights, bias)


def fit_boundary_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nonlinear: bool,
) -> Any:
    """Fit a nonlinear projected-space detector or scalable raw-space probe.

    Args:
        x: Input feature matrix.
        y: Target labels aligned with the feature rows.
        nonlinear: Whether to fit the nonlinear boundary adversary.

    Returns:
        A fitted linear or histogram-gradient-boosting classifier.
    """
    if nonlinear:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=42,
            ),
        )
    model.fit(x, y)
    return model


def boundary_labels(count: int, boundaries: np.ndarray) -> np.ndarray:
    """Convert selected boundary indices into a binary target vector.

    Args:
        count: Number of candidate sentence boundaries.
        boundaries: Sentence or token boundary indices.

    Returns:
        The resulting numeric array or tensor.
    """
    labels = np.zeros(count, dtype=np.int8)
    labels[np.asarray(boundaries, dtype=int)] = 1
    return labels


def matched_probability_labels(
    probabilities: np.ndarray,
    slices: list[tuple[int, int, int]],
) -> np.ndarray:
    """Threshold each trace at its exact oracle boundary budget.

    Args:
        probabilities: Predicted probabilities aligned with the labels.
        slices: Per-trace slices into the flattened probability vector.

    Returns:
        The resulting numeric array or tensor.
    """
    labels = np.zeros(len(probabilities), dtype=np.int8)
    for start, end, positives in slices:
        local = top_boundaries(probabilities[start:end], positives)
        labels[start + local] = 1
    return labels


def evaluate_projection_coherence(
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Measure whether each oracle also compresses each projection space.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.
        primary_partitions: Reference objective-specific partitions keyed by trace.

    Returns:
        The resulting keyed records or metrics.
    """
    spaces = {
        "raw": "raw",
        "pca_whitened": "pca",
        "gram_spectrum": "gram",
        "h4_operation": "h4",
    }
    rng = np.random.default_rng(904)
    scores: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in selected_indices:
        trace = trace_view(cache, index)
        if trace.train:
            continue
        for space, attribute in spaces.items():
            costs = squared_error_costs(getattr(trace, attribute))
            boundary_count = len(primary_partitions[index][ORACLE_NAMES["compression"]])
            oracle = optimal_partition(costs, boundary_count + 1)
            oracle_cost = partition_cost(costs, oracle)
            random_cost = float(
                np.mean(
                    [
                        partition_cost(
                            costs,
                            random_boundaries(len(trace.pca), boundary_count, rng),
                        )
                        for _ in range(12)
                    ]
                )
            )
            denominator = random_cost - oracle_cost
            if denominator <= 1e-10:
                continue
            for objective in OBJECTIVES:
                boundaries = primary_partitions[index][ORACLE_NAMES[objective]]
                utility = (
                    1.0
                    - (partition_cost(costs, boundaries) - oracle_cost) / denominator
                )
                scores[space][objective].append(float(utility))
    return {
        space: {
            objective: float(np.mean(values))
            for objective, values in objective_scores.items()
        }
        for space, objective_scores in scores.items()
    }
