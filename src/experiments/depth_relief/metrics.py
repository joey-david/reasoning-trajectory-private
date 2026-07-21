"""Depth metrics and aggregation for controlled checkpoint experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Return a numerically stable probability vector along the final axis."""
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    """Compute base-two Jensen-Shannon divergence between distributions."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (left + right)
    left_term = np.zeros_like(left)
    right_term = np.zeros_like(right)
    left_mask = left > 0
    right_mask = right > 0
    left_term[left_mask] = left[left_mask] * (
        np.log2(left[left_mask]) - np.log2(midpoint[left_mask])
    )
    right_term[right_mask] = right[right_mask] * (
        np.log2(right[right_mask]) - np.log2(midpoint[right_mask])
    )
    return float(0.5 * (left_term.sum() + right_term.sum()))


def settling_depth(divergences: np.ndarray, *, threshold: float) -> int:
    """Apply DTR's running-minimum threshold to per-layer full-vocabulary JSD."""
    values = np.asarray(divergences, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("divergences must have shape [layers]")
    envelope = np.minimum.accumulate(values)
    settled = np.flatnonzero(envelope <= threshold)
    return int(settled[0]) if len(settled) else len(values) - 1


def normalized_recovery(value: float, corrupt: float, clean: float) -> float | None:
    """Scale an intervention between corrupt and clean endpoints when identifiable."""
    denominator = clean - corrupt
    if abs(denominator) < 1e-8:
        return None
    return float((value - corrupt) / denominator)


def bootstrap_mean_ci(
    values: Iterable[float], *, seed: int = 0, draws: int = 2000
) -> dict[str, float | int | None]:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
    }


def cluster_bootstrap_mean_ci(
    values: Iterable[float],
    clusters: Iterable[str],
    *,
    seed: int = 0,
    draws: int = 2000,
) -> dict[str, float | int | None]:
    """Bootstrap equal-weight cluster means for grouped experimental items."""
    array = np.asarray(list(values), dtype=np.float64)
    labels = np.asarray(list(clusters), dtype=str)
    if len(array) != len(labels):
        raise ValueError("Cluster labels must align with values")
    if len(array) == 0:
        return {
            "n": 0,
            "cluster_n": 0,
            "mean": None,
            "ci95": [None, None],
        }
    unique = np.unique(labels)
    means = np.asarray([array[labels == label].mean() for label in unique])
    rng = np.random.default_rng(seed)
    sampled = rng.choice(means, size=(draws, len(means)), replace=True).mean(axis=1)
    return {
        "n": int(len(array)),
        "cluster_n": int(len(unique)),
        "mean": float(means.mean()),
        "ci95": [
            float(np.quantile(sampled, 0.025)),
            float(np.quantile(sampled, 0.975)),
        ],
    }


def _condition_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row["conditions"][name] for row in rows if name in row.get("conditions", {})]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate depth relief, dose response, branch control, and causal checks."""
    if not rows:
        raise ValueError("Cannot summarize an empty depth-relief result set")
    relief = []
    matched_relief = []
    curve_relief = []
    self_gap = []
    counterfactual_branch = []
    dose: defaultdict[int, list[float]] = defaultdict(list)
    dose_curve: defaultdict[int, list[float]] = defaultdict(list)
    family_relief: defaultdict[str, list[float]] = defaultdict(list)
    threshold_relief: defaultdict[str, list[float]] = defaultdict(list)
    threshold_matched_relief: defaultdict[str, list[float]] = defaultdict(list)
    steering: defaultdict[str, list[float]] = defaultdict(list)
    matched_none_gold = 0
    causal_eligible = 0
    for row in rows:
        conditions = row["conditions"]
        if "none" in conditions and "gold" in conditions:
            none = conditions["none"]
            gold = conditions["gold"]
            value = float(none["settling_depth"] - gold["settling_depth"])
            relief.append(value)
            if "dtr_jsd_auc" in none and "dtr_jsd_auc" in gold:
                curve_relief.append(
                    float(none["dtr_jsd_auc"] - gold["dtr_jsd_auc"])
                )
            family_relief[str(row["family"])].append(value)
            matched = bool(
                none.get("is_expected_unconstrained", none["is_expected"])
                and gold.get("is_expected_unconstrained", gold["is_expected"])
            )
            if matched:
                matched_relief.append(value)
                matched_none_gold += 1
            for threshold, none_depth in none.get(
                "settling_depth_by_threshold", {}
            ).items():
                gold_depth = gold.get("settling_depth_by_threshold", {}).get(threshold)
                if gold_depth is not None:
                    threshold_value = float(none_depth - gold_depth)
                    threshold_relief[str(threshold)].append(threshold_value)
                    if matched:
                        threshold_matched_relief[str(threshold)].append(
                            threshold_value
                        )
            if all(
                "final_candidate_probabilities" in condition
                for condition in (none, gold)
            ):
                target = int(row["next_state"])
                steering["gold_correct_probability_gain"].append(
                    float(
                        gold["final_candidate_probabilities"][target]
                        - none["final_candidate_probabilities"][target]
                    )
                )
        if "self" in conditions and "gold" in conditions:
            self_gap.append(float(conditions["self"]["settling_depth"] - conditions["gold"]["settling_depth"]))
        if "counterfactual" in conditions:
            counterfactual = conditions["counterfactual"]
            counterfactual_branch.append(
                float(
                    bool(
                        counterfactual.get(
                            "is_expected_unconstrained",
                            counterfactual["is_expected"],
                        )
                    )
                )
            )
            gold = conditions.get("gold")
            if gold and gold.get("is_expected_unconstrained", gold["is_expected"]):
                if counterfactual.get(
                    "is_expected_unconstrained", counterfactual["is_expected"]
                ):
                    causal_eligible += 1
            none = conditions.get("none")
            if none and all(
                "final_candidate_probabilities" in condition
                for condition in (none, counterfactual)
            ):
                target = int(counterfactual["expected_next_state"])
                steering["counterfactual_branch_probability_gain"].append(
                    float(
                        counterfactual["final_candidate_probabilities"][target]
                        - none["final_candidate_probabilities"][target]
                    )
                )
        if "random" in conditions and "none" in conditions:
            random = conditions["random"]
            none = conditions["none"]
            if all(
                "final_candidate_probabilities" in condition
                for condition in (none, random)
            ):
                target = int(random["expected_next_state"])
                steering["random_branch_probability_gain"].append(
                    float(
                        random["final_candidate_probabilities"][target]
                        - none["final_candidate_probabilities"][target]
                    )
                )
        for condition in conditions.values():
            revealed = condition.get("revealed_bits")
            if revealed is not None and condition.get("expected_next_state") == row["next_state"]:
                dose[int(revealed)].append(float(condition["settling_depth"]))
                if "dtr_jsd_auc" in condition:
                    dose_curve[int(revealed)].append(float(condition["dtr_jsd_auc"]))

    causal_attempted = [row for row in rows if row.get("causal")]
    causal = [
        row["causal"]
        for row in causal_attempted
        if row["conditions"]["gold"].get("is_expected_unconstrained", False)
        and row["conditions"]["counterfactual"].get(
            "is_expected_unconstrained", False
        )
    ]
    target_redirect = []
    hidden_override = []
    read_depths = []
    state_depths = []
    for result in causal:
        target_redirect.extend(
            float(value["counterfactual_branch_probability"])
            for value in result.get("target_interchange", [])
        )
        if "gold_hidden_into_counterfactual" in result.get("channel_decomposition", {}):
            hidden_override.append(
                float(result["channel_decomposition"]["gold_hidden_into_counterfactual"]["correct_probability"])
            )
        if result.get("read_depth") is not None:
            read_depths.append(float(result["read_depth"]))
        if result.get("state_control_depth") is not None:
            state_depths.append(float(result["state_control_depth"]))

    dose_means = {str(level): float(np.mean(values)) for level, values in sorted(dose.items())}
    dose_levels = sorted(dose)
    dose_slope = None
    if len(dose_levels) >= 2:
        dose_slope = float(np.polyfit(dose_levels, [dose_means[str(level)] for level in dose_levels], 1)[0])
    dose_curve_means = {
        str(level): float(np.mean(values))
        for level, values in sorted(dose_curve.items())
    }
    dose_curve_slope = None
    dose_curve_levels = sorted(dose_curve)
    if len(dose_curve_levels) >= 2:
        dose_curve_slope = float(
            np.polyfit(
                dose_curve_levels,
                [dose_curve_means[str(level)] for level in dose_curve_levels],
                1,
            )[0]
        )
    failures: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        writer = row["writer"]
        self_condition = row.get("conditions", {}).get("self")
        if not writer.get("correct_top1_at_any_layer"):
            label = "compute_failure"
        elif not writer.get("is_correct_unconstrained", writer.get("is_correct")):
            label = "write_failure"
        elif not self_condition or not self_condition.get(
            "is_expected_unconstrained", self_condition.get("is_expected")
        ):
            label = "read_or_transition_failure"
        else:
            label = "success"
        failures[label] += 1
    condition_names = sorted(
        {name for row in rows for name in row.get("conditions", {})}
    )
    validity = {
        name: {
            "n": len(values),
            "forced_choice_expected_rate": float(
                np.mean([bool(value["is_expected"]) for value in values])
            ),
            "unconstrained_expected_rate": float(
                np.mean(
                    [
                        bool(value.get("is_expected_unconstrained", False))
                        for value in values
                    ]
                )
            ),
            "mean_candidate_probability_mass": float(
                np.mean(
                    [value.get("candidate_probability_mass", 1.0) for value in values]
                )
            ),
        }
        for name in condition_names
        if (values := _condition_rows(rows, name))
    }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "families": sorted({str(row["family"]) for row in rows}),
        "depth_relief": {
            "all": bootstrap_mean_ci(relief, seed=11),
            "matched_accuracy": bootstrap_mean_ci(matched_relief, seed=12),
            "by_family": {
                family: bootstrap_mean_ci(values, seed=20 + index)
                for index, (family, values) in enumerate(sorted(family_relief.items()))
            },
            "jsd_curve_area": bootstrap_mean_ci(curve_relief, seed=13),
            "by_threshold": {
                threshold: {
                    "all": bootstrap_mean_ci(values, seed=50 + index),
                    "matched_accuracy": bootstrap_mean_ci(
                        threshold_matched_relief[threshold], seed=60 + index
                    ),
                }
                for index, (threshold, values) in enumerate(
                    sorted(threshold_relief.items(), key=lambda item: float(item[0]))
                )
            },
        },
        "self_vs_gold_depth_gap": bootstrap_mean_ci(self_gap, seed=31),
        "counterfactual_rule_consistent_rate": float(np.mean(counterfactual_branch)) if counterfactual_branch else None,
        "dose_response": {
            "mean_depth_by_revealed_bits": dose_means,
            "linear_slope": dose_slope,
            "mean_jsd_curve_area_by_revealed_bits": dose_curve_means,
            "jsd_curve_area_linear_slope": dose_curve_slope,
        },
        "register_steering": {
            name: bootstrap_mean_ci(values, seed=70 + index)
            for index, (name, values) in enumerate(sorted(steering.items()))
        },
        "causal": {
            "attempted_case_count": len(causal_attempted),
            "valid_case_count": len(causal),
            "mean_target_interchange_counterfactual_probability": float(np.mean(target_redirect)) if target_redirect else None,
            "mean_gold_hidden_override_correct_probability": float(np.mean(hidden_override)) if hidden_override else None,
            "read_depth": bootstrap_mean_ci(read_depths, seed=41),
            "state_control_depth": bootstrap_mean_ci(state_depths, seed=42),
        },
        "failure_taxonomy": dict(sorted(failures.items())),
        "validity": {
            "conditions": validity,
            "matched_none_gold_unconstrained": {
                "n": matched_none_gold,
                "total": len(rows),
                "rate": matched_none_gold / len(rows),
            },
            "causal_eligible_gold_counterfactual_unconstrained": {
                "n": causal_eligible,
                "total": len(rows),
                "rate": causal_eligible / len(rows),
            },
            "writer_unconstrained_correct_rate": float(
                np.mean(
                    [
                        bool(
                            row["writer"].get(
                                "is_correct_unconstrained",
                                row["writer"].get("is_correct", False),
                            )
                        )
                        for row in rows
                    ]
                )
            ),
        },
    }
