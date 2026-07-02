"""Exact matched-budget partitions and comparison metrics for sentence lattices."""

from __future__ import annotations

import math

import numpy as np


def squared_error_costs(values: np.ndarray) -> np.ndarray:
    """Return additive within-segment SSE costs for every half-open interval.

    Args:
        values: Values to summarize or transform.

    Returns:
        The resulting numeric array or tensor.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    count = len(x)
    prefix = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    squared = np.r_[0.0, np.cumsum(np.einsum("ij,ij->i", x, x))]
    costs = np.full((count, count + 1), np.inf, dtype=np.float64)
    for end in range(1, count + 1):
        starts = np.arange(end)
        lengths = (end - starts).astype(np.float64)
        sums = prefix[end] - prefix[starts]
        costs[starts, end] = (
            squared[end]
            - squared[starts]
            - np.einsum("ij,ij->i", sums, sums) / lengths
        )
    return np.maximum(costs, 0.0)


def object_update_costs(update_counts: np.ndarray) -> np.ndarray:
    """Penalize segments that contain anything other than one symbolic update.

    Args:
        update_counts: Symbolic-update counts for each sentence.

    Returns:
        The resulting numeric array or tensor.
    """
    updates = np.asarray(update_counts, dtype=np.float64)
    count = len(updates)
    prefix = np.r_[0.0, np.cumsum(updates)]
    costs = np.full((count, count + 1), np.inf, dtype=np.float64)
    for end in range(1, count + 1):
        starts = np.arange(end)
        totals = prefix[end] - prefix[starts]
        costs[starts, end] = np.square(totals - 1.0)
    return costs


def optimal_partition(costs: np.ndarray, segments: int) -> np.ndarray:
    """Find the exact minimum-cost partition with a fixed segment count.

    Args:
        costs: Precomputed interval costs indexed by endpoints.
        segments: Exact number of nonempty segments in the partition.

    Returns:
        The resulting numeric array or tensor.
    """
    count = costs.shape[0]
    segments = min(max(int(segments), 1), count)
    previous = np.full(count + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    backpointers = np.full((segments + 1, count + 1), -1, dtype=np.int32)

    # DP state (part, end) stores the cheapest partition of sentences [0, end)
    # into exactly `part` nonempty segments.
    for part in range(1, segments + 1):
        current = np.full(count + 1, np.inf, dtype=np.float64)
        for end in range(part, count + 1):
            starts = np.arange(part - 1, end)
            candidates = previous[starts] + costs[starts, end]
            best = int(np.argmin(candidates))
            current[end] = candidates[best]
            backpointers[part, end] = int(starts[best])
        previous = current

    boundaries: list[int] = []
    end = count
    # Convert half-open segment starts back to the left-sentence boundary
    # convention used throughout the sentence-lattice reports.
    for part in range(segments, 1, -1):
        start = int(backpointers[part, end])
        if start < 1:
            raise ValueError("Partition backtracking failed")
        boundaries.append(start - 1)
        end = start
    return np.asarray(sorted(boundaries), dtype=np.int32)


def partition_cost(costs: np.ndarray, boundaries: np.ndarray) -> float:
    """Sum interval costs for a boundary set indexed by its left sentence.

    Args:
        costs: Precomputed interval costs indexed by endpoints.
        boundaries: Sentence or token boundary indices.

    Returns:
        The computed scalar metric.
    """
    count = costs.shape[0]
    ends = [int(boundary) + 1 for boundary in boundaries] + [count]
    total = 0.0
    start = 0
    for end in ends:
        total += float(costs[start, end])
        start = end
    return total


def top_boundaries(scores: np.ndarray, boundary_count: int) -> np.ndarray:
    """Select the highest-scoring sentence transitions under a fixed budget.

    Args:
        scores: Method-by-objective scores.
        boundary_count: Number of boundaries to select.

    Returns:
        The resulting numeric array or tensor.
    """
    values = np.asarray(scores, dtype=np.float64)
    boundary_count = min(max(int(boundary_count), 0), len(values))
    if boundary_count == 0:
        return np.empty(0, dtype=np.int32)
    selected = np.argpartition(values, -boundary_count)[-boundary_count:]
    return np.sort(selected.astype(np.int32))


def fixed_boundaries(sentence_count: int, boundary_count: int) -> np.ndarray:
    """Place a fixed number of boundaries as evenly as possible.

    Args:
        sentence_count: Number of sentences in the trace.
        boundary_count: Number of boundaries to select.

    Returns:
        The resulting numeric array or tensor.
    """
    if boundary_count <= 0:
        return np.empty(0, dtype=np.int32)
    positions = np.linspace(1, sentence_count - 1, boundary_count + 2)[1:-1]
    return np.unique(np.rint(positions).astype(np.int32) - 1)


def random_boundaries(
    sentence_count: int,
    boundary_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample matched-budget boundaries without replacement.

    Args:
        sentence_count: Number of sentences in the trace.
        boundary_count: Number of boundaries to select.
        rng: Random-number generator used for reproducible sampling.

    Returns:
        The resulting numeric array or tensor.
    """
    boundary_count = min(max(boundary_count, 0), sentence_count - 1)
    if boundary_count == 0:
        return np.empty(0, dtype=np.int32)
    return np.sort(
        rng.choice(sentence_count - 1, size=boundary_count, replace=False).astype(
            np.int32
        )
    )


def boundary_f1(
    predicted: np.ndarray,
    expected: np.ndarray,
    *,
    tolerance: int = 0,
) -> float:
    """Compute one-to-one boundary F1 within an optional sentence tolerance.

    Args:
        predicted: Predicted boundary indices.
        expected: Reference boundary indices.
        tolerance: Maximum boundary-index distance counted as a match.

    Returns:
        The computed scalar metric.
    """
    predicted_set = [int(value) for value in predicted]
    remaining = set(int(value) for value in expected)
    matches = 0
    for value in predicted_set:
        # Remove each reference after matching so one expected boundary cannot
        # inflate recall by satisfying several nearby predictions.
        candidates = [
            candidate
            for candidate in remaining
            if abs(candidate - value) <= tolerance
        ]
        if candidates:
            match = min(candidates, key=lambda candidate: abs(candidate - value))
            remaining.remove(match)
            matches += 1
    precision = matches / max(len(predicted_set), 1)
    recall = matches / max(len(expected), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def boundary_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Compute exact Jaccard agreement between two boundary sets.

    Args:
        left: Left operand or comparison input.
        right: Right operand or comparison input.

    Returns:
        The computed scalar metric.
    """
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    return len(left_set & right_set) / max(len(union), 1)


def partition_variation_of_information(
    left: np.ndarray,
    right: np.ndarray,
    sentence_count: int,
) -> float:
    """Measure disagreement between two partitions in normalized information units.

    Args:
        left: Left operand or comparison input.
        right: Right operand or comparison input.
        sentence_count: Number of sentences in the trace.

    Returns:
        The computed scalar metric.
    """
    if sentence_count <= 1:
        return 0.0
    left_labels = partition_labels(left, sentence_count)
    right_labels = partition_labels(right, sentence_count)
    contingency = np.zeros(
        (left_labels.max() + 1, right_labels.max() + 1),
        dtype=np.float64,
    )
    np.add.at(contingency, (left_labels, right_labels), 1.0)
    joint = contingency / sentence_count
    left_prob = joint.sum(axis=1)
    right_prob = joint.sum(axis=0)
    left_entropy = entropy(left_prob)
    right_entropy = entropy(right_prob)
    denominator = left_prob[:, None] * right_prob[None, :]
    active = joint > 0
    mutual_information = float(
        np.sum(joint[active] * np.log(joint[active] / denominator[active]))
    )
    return (left_entropy + right_entropy - 2.0 * mutual_information) / math.log(
        sentence_count
    )


def partition_labels(boundaries: np.ndarray, sentence_count: int) -> np.ndarray:
    """Convert boundary indices into one integer segment label per sentence.

    Args:
        boundaries: Sentence or token boundary indices.
        sentence_count: Number of sentences in the trace.

    Returns:
        The resulting numeric array or tensor.
    """
    labels = np.zeros(sentence_count, dtype=np.int32)
    for segment, start in enumerate(
        [int(boundary) + 1 for boundary in boundaries], start=1
    ):
        labels[start:] = segment
    return labels


def entropy(probabilities: np.ndarray) -> float:
    """Return natural-log entropy for a discrete probability vector.

    Args:
        probabilities: Predicted probabilities aligned with the labels.

    Returns:
        The computed scalar metric.
    """
    active = probabilities > 0
    return float(-np.sum(probabilities[active] * np.log(probabilities[active])))


def pareto_front(scores: dict[str, dict[str, float]]) -> list[str]:
    """Return methods not strictly dominated across all supplied objectives.

    Args:
        scores: Method-by-objective scores.

    Returns:
        The resulting ordered records or values.
    """
    methods = sorted(scores)
    objectives = sorted(next(iter(scores.values()))) if scores else []
    front: list[str] = []
    for candidate in methods:
        dominated = False
        for other in methods:
            if other == candidate:
                continue
            weakly_better = all(
                scores[other][objective] >= scores[candidate][objective]
                for objective in objectives
            )
            strictly_better = any(
                scores[other][objective] > scores[candidate][objective]
                for objective in objectives
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front
