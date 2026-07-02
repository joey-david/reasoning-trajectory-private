"""Objective-conditioned sentence partitions and oracle-relative evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.experiments.sentence_lattice import optimal_partition, partition_cost


def boundary_budget(sentence_count: int, fraction: float) -> int:
    """Convert a boundary fraction into a valid fixed boundary count.

    Args:
        sentence_count: Number of sentences in the trace.
        fraction: Requested fraction of available sentence boundaries.

    Returns:
        A boundary count between one and ``sentence_count - 1``.
    """
    if sentence_count < 2:
        return 0
    return min(
        max(int(round((sentence_count - 1) * fraction)), 1),
        sentence_count - 1,
    )


def objective_partition(costs: np.ndarray, boundary_count: int) -> np.ndarray:
    """Compute the exact fixed-budget partition for one additive objective.

    Args:
        costs: Matrix of half-open segment costs indexed by start and end.
        boundary_count: Exact number of internal boundaries to select.

    Returns:
        Sorted left-sentence indices of the optimal boundaries.
    """
    sentence_count = int(costs.shape[0])
    if costs.shape != (sentence_count, sentence_count + 1):
        raise ValueError("costs must have shape [sentences, sentences + 1]")
    boundary_count = min(max(int(boundary_count), 0), sentence_count - 1)
    return optimal_partition(costs, boundary_count + 1)


def typed_object_costs(edit_counts: np.ndarray) -> np.ndarray:
    """Build additive costs favoring one coherent typed edit per segment.

    Args:
        edit_counts: Sentence-by-edit-type count matrix.

    Returns:
        Half-open segment costs combining edit count and mixed-type penalties.
    """
    counts = np.asarray(edit_counts, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("edit_counts must have shape [sentences, edit types]")
    sentence_count = len(counts)
    prefix = np.vstack([np.zeros((1, counts.shape[1])), np.cumsum(counts, axis=0)])
    costs = np.full(
        (sentence_count, sentence_count + 1),
        np.inf,
        dtype=np.float64,
    )
    for end in range(1, sentence_count + 1):
        starts = np.arange(end)
        totals = prefix[end] - prefix[starts]
        edit_total = totals.sum(axis=1)
        active_types = np.count_nonzero(totals, axis=1)
        costs[starts, end] = (
            np.square(edit_total - 1.0)
            + np.maximum(active_types - 1, 0)
            + 0.25 * np.maximum(edit_total - 1.0, 0)
        )
    return costs


def normalized_regret(
    costs: np.ndarray,
    boundaries: np.ndarray,
    oracle_boundaries: np.ndarray,
    random_boundaries: Sequence[np.ndarray],
) -> float | None:
    """Measure candidate regret relative to oracle and matched random partitions.

    Args:
        costs: Matrix of additive segment costs for the evaluated objective.
        boundaries: Candidate partition boundaries.
        oracle_boundaries: Exact objective-optimal boundaries at the same budget.
        random_boundaries: Matched-budget random boundary samples.

    Returns:
        Zero for the oracle and one for mean random performance, or ``None``
        when random and oracle costs are indistinguishable.
    """
    if not random_boundaries:
        raise ValueError("at least one random partition is required")
    oracle_cost = partition_cost(costs, oracle_boundaries)
    random_cost = float(
        np.mean([partition_cost(costs, sample) for sample in random_boundaries])
    )
    scale = random_cost - oracle_cost
    if scale <= 1e-10:
        return None
    return float((partition_cost(costs, boundaries) - oracle_cost) / scale)


def regret_matrix(
    objective_costs: Mapping[str, np.ndarray],
    partitions: Mapping[str, np.ndarray],
    oracles: Mapping[str, np.ndarray],
    random_boundaries: Sequence[np.ndarray],
) -> dict[str, dict[str, float | None]]:
    """Evaluate every partition against every objective under one budget.

    Args:
        objective_costs: Additive cost matrix keyed by objective name.
        partitions: Candidate boundary sets keyed by method name.
        oracles: Objective-optimal boundary sets keyed by objective name.
        random_boundaries: Shared matched-budget random partitions.

    Returns:
        Nested method-by-objective normalized regrets.
    """
    if objective_costs.keys() != oracles.keys():
        raise ValueError("every objective must have exactly one oracle")
    return {
        method: {
            objective: normalized_regret(
                costs,
                boundaries,
                oracles[objective],
                random_boundaries,
            )
            for objective, costs in objective_costs.items()
        }
        for method, boundaries in partitions.items()
    }


def append_objective_identity(
    features: np.ndarray,
    objective_index: int,
    objective_count: int,
) -> np.ndarray:
    """Append a one-hot objective identity to each boundary feature row.

    Args:
        features: Boundary feature matrix.
        objective_index: Zero-based objective identity.
        objective_count: Total number of supported objectives.

    Returns:
        Feature rows augmented with the repeated objective one-hot vector.
    """
    values = np.asarray(features, dtype=np.float32)
    if not 0 <= objective_index < objective_count:
        raise ValueError("objective_index is outside the objective vocabulary")
    identity = np.zeros((len(values), objective_count), dtype=np.float32)
    identity[:, objective_index] = 1.0
    return np.concatenate([values, identity], axis=1)
