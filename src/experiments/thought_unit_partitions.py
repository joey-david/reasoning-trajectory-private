"""Objective costs, matched-budget partitions, and cross-objective regret."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.experiments.objective_segmentation import (
    boundary_budget,
    normalized_regret,
    objective_partition,
)
from src.experiments.sentence_lattice import (
    boundary_f1,
    boundary_jaccard,
    fixed_boundaries,
    object_update_costs,
    pareto_front,
    partition_variation_of_information,
    random_boundaries,
    squared_error_costs,
    top_boundaries,
)
from src.experiments.thought_unit_cache import trace_view
from src.experiments.thought_unit_types import (
    OBJECTIVES,
    ORACLE_NAMES,
    PRIMARY_FRACTION,
    TraceView,
)


def objective_costs(
    trace: TraceView,
    correctness_curve: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the four additive sentence-segment objective costs.

    Args:
        trace: Sentence-level activation and annotation view.
        correctness_curve: Per-boundary correctness-information scores.

    Returns:
        The resulting keyed records or metrics.
    """
    return {
        "answer": squared_error_costs(trace.answer_score),
        "object": object_update_costs(trace.update_count),
        "correctness": squared_error_costs(correctness_curve),
        "compression": squared_error_costs(trace.pca),
    }


def candidate_partitions(
    trace: TraceView,
    costs: dict[str, np.ndarray],
    *,
    fraction: float,
    rng: np.random.Generator,
    gram_transition_scores: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Build matched-budget heuristic and exact-oracle partitions.

    Args:
        trace: Sentence-level activation and annotation view.
        costs: Precomputed interval costs indexed by endpoints.
        fraction: Target fraction of available sentence boundaries.
        rng: Random-number generator used for reproducible sampling.
        gram_transition_scores: Per-boundary Gram-state transition scores.

    Returns:
        The computed aligned values described above.
    """
    count = len(trace.pca)
    boundary_count = boundary_budget(count, fraction)
    answer_change = np.abs(np.diff(trace.answer_score))
    h4_change = np.linalg.norm(np.diff(trace.h4, axis=0), axis=1)
    curvature = trace.raw_geometry[1:, 3]
    partitions = {
        "fixed_windows": fixed_boundaries(count, boundary_count),
        "raw_curvature": top_boundaries(curvature, boundary_count),
        "answer_peaks": top_boundaries(answer_change, boundary_count),
        "gram_transitions": top_boundaries(gram_transition_scores, boundary_count),
        "h4_transitions": top_boundaries(h4_change, boundary_count),
    }
    for objective, cost in costs.items():
        partitions[ORACLE_NAMES[objective]] = objective_partition(cost, boundary_count)
    random_samples = [random_boundaries(count, boundary_count, rng) for _ in range(12)]
    partitions["random"] = random_samples[0]
    return partitions, random_samples


def evaluate_partitions(
    cache: dict[str, Any],
    selected_indices: list[int],
    correctness_curves: dict[int, np.ndarray],
    gram_scores: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Evaluate all candidate partitions on held-out questions and budget sweeps.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.
        correctness_curves: Correctness-information curves keyed by trace.
        gram_scores: Gram-state transition scores keyed by trace.

    Returns:
        The resulting keyed records or metrics.
    """
    rng = np.random.default_rng(42)
    test_indices = [
        index
        for index in selected_indices
        if not bool(cache["records"][index]["train"])
    ]
    all_primary: dict[int, dict[str, np.ndarray]] = {}
    result: dict[str, Any] = {}
    for fraction in (0.1, 0.2, 0.3):
        utilities: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        utility_groups: defaultdict[str, defaultdict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        agreements: defaultdict[tuple[str, str], list[tuple[float, float, float]]] = (
            defaultdict(list)
        )
        for index in selected_indices:
            trace = trace_view(cache, index)
            costs = objective_costs(trace, correctness_curves[index])
            partitions, random_samples = candidate_partitions(
                trace,
                costs,
                fraction=fraction,
                rng=rng,
                gram_transition_scores=gram_scores[index],
            )
            if fraction == PRIMARY_FRACTION:
                all_primary[index] = partitions
            if index not in test_indices:
                continue
            for objective, cost in costs.items():
                # Zero utility is matched-random performance and one is the
                # objective-specific oracle, making columns comparable despite
                # their different raw cost scales.
                for method, boundaries in partitions.items():
                    regret = normalized_regret(
                        cost,
                        boundaries,
                        partitions[ORACLE_NAMES[objective]],
                        random_samples,
                    )
                    if regret is None:
                        continue
                    utility = 1.0 - regret
                    utilities[method][objective].append(float(utility))
                    utility_groups[method][objective].append(trace.sample_id)

            oracle_pairs = [
                (left, right)
                for i, left in enumerate(OBJECTIVES)
                for right in OBJECTIVES[i + 1 :]
            ]
            for left, right in oracle_pairs:
                left_boundaries = partitions[ORACLE_NAMES[left]]
                right_boundaries = partitions[ORACLE_NAMES[right]]
                agreements[(left, right)].append(
                    (
                        boundary_jaccard(left_boundaries, right_boundaries),
                        boundary_f1(left_boundaries, right_boundaries, tolerance=1),
                        partition_variation_of_information(
                            left_boundaries,
                            right_boundaries,
                            len(trace.pca),
                        ),
                    )
                )

        mean_utilities = {
            method: {
                objective: float(np.mean(values))
                for objective, values in objective_values.items()
            }
            for method, objective_values in utilities.items()
        }
        utility_intervals = {
            method: {
                objective: grouped_bootstrap_summary(
                    values,
                    utility_groups[method][objective],
                )
                for objective, values in objective_values.items()
            }
            for method, objective_values in utilities.items()
        }
        agreement_rows = [
            {
                "left": left,
                "right": right,
                "boundary_fraction": fraction,
                "jaccard": float(np.mean([value[0] for value in values])),
                "f1_tolerance_1": float(np.mean([value[1] for value in values])),
                "variation_of_information": float(
                    np.mean([value[2] for value in values])
                ),
            }
            for (left, right), values in agreements.items()
        ]
        key = f"fraction_{fraction:.1f}"
        result[key] = {
            "utilities": mean_utilities,
            "utility_question_bootstrap": utility_intervals,
            "boundary_agreement": agreement_rows,
            "pareto_front": pareto_front(mean_utilities),
            "heuristic_rank_correlations": objective_rank_correlations(mean_utilities),
            "best_worst_case": best_worst_case(mean_utilities),
            "test_trajectories": len(test_indices),
        }
    result["primary"] = result["fraction_0.2"]
    result["primary_partitions"] = all_primary
    return result


def grouped_bootstrap_summary(
    values: list[float],
    groups: list[str],
    *,
    resamples: int = 4000,
) -> dict[str, Any]:
    """Summarize a metric with a question-level nonparametric bootstrap.

    Args:
        values: Values to summarize or transform.
        groups: Group labels used to prevent cross-question leakage.
        resamples: Number of bootstrap resamples.

    Returns:
        The resulting keyed records or metrics.
    """
    by_group: defaultdict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    group_means = np.asarray(
        [np.mean(by_group[group]) for group in sorted(by_group)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(421)
    draws = rng.choice(
        group_means,
        size=(resamples, len(group_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(group_means.mean()),
        "95ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "questions": len(group_means),
    }


def objective_rank_correlations(
    utilities: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    """Compare heuristic rankings across objectives without oracle tautologies.

    Args:
        utilities: Utility scores for candidate segmentations.

    Returns:
        The resulting ordered records or values.
    """
    methods = [
        method
        for method in (
            "fixed_windows",
            "raw_curvature",
            "answer_peaks",
            "gram_transitions",
            "h4_transitions",
            "random",
        )
        if method in utilities
    ]
    rows: list[dict[str, float | str]] = []
    for index, left in enumerate(OBJECTIVES):
        for right in OBJECTIVES[index + 1 :]:
            correlation = spearmanr(
                [utilities[method][left] for method in methods],
                [utilities[method][right] for method in methods],
            ).statistic
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(correlation),
                }
            )
    return rows


def best_worst_case(
    utilities: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Identify the method with the strongest minimum cross-objective utility.

    Args:
        utilities: Utility scores for candidate segmentations.

    Returns:
        The resulting keyed records or metrics.
    """
    minima = {method: min(scores.values()) for method, scores in utilities.items()}
    method = max(minima, key=minima.get)
    return {
        "method": method,
        "minimum_utility": float(minima[method]),
        "near_optimal_on_all_objectives": minima[method] >= 0.9,
    }
