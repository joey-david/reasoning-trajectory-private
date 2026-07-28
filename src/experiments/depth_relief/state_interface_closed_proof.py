"""Reduce closed-transition proof-state confirmation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci
from .state_interface_challenge import (
    _write_summary,
    challenge_dir,
    configured_challenge_profiles,
)


def _completed_summary(run_path: Path, profile: str) -> dict[str, Any] | None:
    path = challenge_dir(run_path, profile) / "summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    return summary if summary.get("complete") else None


def _mean(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def _plot(
    path: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    primary: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    depths = [1, 2, 3]
    for condition, color in (
        ("compressed_3bit", "#cc6677"),
        ("canonical_4bit", "#4477aa"),
        ("redundant_5bit", "#228833"),
    ):
        summaries = [
            results.get(source, {}).get(condition, {}).get("balanced_depth")
            for source in primary
        ]
        summaries = [summary for summary in summaries if summary]
        if summaries:
            axes[0].plot(
                depths,
                [
                    100
                    * sum(
                        _mean(
                            summary,
                            "by_active_transition_count",
                            str(depth),
                            "semantic_state_accuracy",
                        )
                        for summary in summaries
                    )
                    / len(summaries)
                    for depth in depths
                ],
                marker="o",
                color=color,
                label=condition.replace("_", " "),
            )
    axes[0].set_title("Matched-endpoint proof depth")
    axes[0].set_xlabel("Active deductions")
    axes[0].set_ylabel("Semantic state accuracy (%)")
    axes[0].legend(frameon=False, fontsize=7)

    horizons = [16, 64, 128, 256]
    redundant = [
        results[source]["redundant_5bit"]["length"] for source in primary
    ]
    axes[1].plot(
        horizons,
        [
            100
            * sum(
                _mean(
                    summary,
                    "by_surface_horizon",
                    str(horizon),
                    "semantic_state_accuracy",
                )
                for summary in redundant
            )
            / len(redundant)
            for horizon in horizons
        ],
        marker="o",
        color="#228833",
    )
    axes[1].set_title("Fixed-depth surface horizon")
    axes[1].set_xlabel("Rules")

    width5 = results["closure_width5"]
    values = [
        100
        * _mean(
            width5[condition]["width5_depth"],
            "by_active_transition_count",
            "4",
            "semantic_state_accuracy",
        )
        for condition in ("rate_16", "rate_32")
    ]
    axes[2].bar(["4-bit code", "5-bit code"], values, color=["#cc6677", "#4477aa"])
    axes[2].set_title("Five-fact depth-4 rate test")

    for axis in axes:
        axis.set_ylim(0, 105)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_closed_proof_confirmation(run_path: Path) -> dict[str, Any]:
    """Apply the locked closure, depth, capacity, and proof-use gates."""
    config = load_config(run_path)
    experiment = config["state_interface_closed_proof_confirmation"]
    profiles = configured_challenge_profiles(run_path)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for profile in profiles:
        _write_summary(run_path, profile)
        summary = _completed_summary(run_path, profile)
        if summary is None:
            missing.append(profile)
            continue
        source, condition, template = profile.split("__", 2)
        results.setdefault(source, {}).setdefault(condition, {})[template] = summary

    primary = [str(value) for value in experiment["primary_sources"]]
    required = [
        (source, "redundant_5bit", template)
        for source in primary
        for template in ("balanced_depth", "length", "topology", "rule_selection")
    ]
    complete = not missing and all(
        template in results.get(source, {}).get(condition, {})
        for source, condition, template in required
    )
    checks: dict[str, bool] = {"all_profiles_complete": complete}
    metrics: dict[str, Any] = {}
    if complete:
        depth3 = [
            _mean(
                results[source]["redundant_5bit"]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        rule_selection = [
            _mean(
                results[source]["redundant_5bit"]["rule_selection"],
                "interface",
                "accuracy",
            )
            for source in primary
        ]
        h256 = [
            _mean(
                results[source]["redundant_5bit"]["length"],
                "by_surface_horizon",
                "256",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        h256_advantage = [
            _mean(
                results[source]["redundant_5bit"]["length"],
                "by_surface_horizon",
                "256",
                "interface_minus_outcome",
            )
            for source in primary
        ]
        length_drop = [
            _mean(
                results[source]["redundant_5bit"]["length"],
                "by_surface_horizon",
                "16",
                "semantic_state_accuracy",
            )
            - h256[index]
            for index, source in enumerate(primary)
        ]
        false_positive = [
            float(
                results[source]["redundant_5bit"]["length"][
                    "by_surface_horizon"
                ]["256"]["false_positive_fact_rate"]
            )
            for source in primary
        ]
        all_facts_excess = [
            float(
                results[source]["redundant_5bit"]["length"][
                    "by_surface_horizon"
                ]["256"]["all_facts_prediction_rate"]
            )
            - float(
                results[source]["redundant_5bit"]["length"][
                    "by_surface_horizon"
                ]["256"]["all_facts_target_rate"]
            )
            for source in primary
        ]
        width5 = results[str(experiment["width5_source"])]
        width5_exact = _mean(
            width5["rate_32"]["width5_depth"],
            "by_active_transition_count",
            "4",
            "semantic_state_accuracy",
        )
        width5_compressed = _mean(
            width5["rate_16"]["width5_depth"],
            "by_active_transition_count",
            "4",
            "semantic_state_accuracy",
        )
        metrics = {
            "redundant_depth3_semantic": {
                "per_seed": depth3,
                **bootstrap_mean_ci(depth3, seed=84_101),
            },
            "redundant_rule_selection": {
                "per_seed": rule_selection,
                **bootstrap_mean_ci(rule_selection, seed=84_102),
            },
            "redundant_h256_semantic": {
                "per_seed": h256,
                **bootstrap_mean_ci(h256, seed=84_103),
            },
            "redundant_h256_minus_outcome": {
                "per_seed": h256_advantage,
                **bootstrap_mean_ci(h256_advantage, seed=84_104),
            },
            "h16_minus_h256_semantic": {
                "per_seed": length_drop,
                **bootstrap_mean_ci(length_drop, seed=84_105),
            },
            "h256_false_positive_fact_rate": {
                "per_seed": false_positive,
                **bootstrap_mean_ci(false_positive, seed=84_106),
            },
            "h256_all_facts_rate_excess": {
                "per_seed": all_facts_excess,
                **bootstrap_mean_ci(all_facts_excess, seed=84_107),
            },
            "width5_depth4_semantic": width5_exact,
            "width5_rate32_minus_rate16_depth4": (
                width5_exact - width5_compressed
            ),
        }
        gates = experiment["gates"]
        checks.update(
            redundant_depth3=all(
                value
                >= float(
                    gates["min_redundant_depth3_semantic_accuracy_each_seed"]
                )
                for value in depth3
            ),
            reusable_rule_selection=all(
                value
                >= float(
                    gates["min_redundant_rule_selection_accuracy_each_seed"]
                )
                for value in rule_selection
            ),
            h256_semantic=all(
                value
                >= float(
                    gates["min_redundant_h256_semantic_accuracy_each_seed"]
                )
                for value in h256
            ),
            h256_beats_outcome=(
                metrics["redundant_h256_minus_outcome"]["ci95"][0] > 0
                and sum(h256_advantage) / len(h256_advantage)
                >= float(gates["min_mean_redundant_h256_minus_outcome"])
            ),
            length_stable=(
                sum(length_drop) / len(length_drop)
                <= float(gates["max_mean_h16_minus_h256_drop"])
            ),
            false_positive_control=(
                max(false_positive)
                <= float(gates["max_false_positive_fact_rate"])
            ),
            no_all_facts_collapse=(
                max(all_facts_excess)
                <= float(gates["max_all_facts_rate_excess"])
            ),
            width5_depth4=(
                width5_exact
                >= float(gates["min_width5_depth4_semantic_accuracy"])
            ),
            width5_rate_separation=(
                width5_exact - width5_compressed
                >= float(gates["min_width5_rate32_minus_rate16_depth4"])
            ),
        )

    status = (
        "pending"
        if not complete
        else ("passed" if all(checks.values()) else "failed")
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "missing_profiles": missing,
        "profile_count": len(profiles),
        "complete_profile_count": len(profiles) - len(missing),
    }
    output = run_path / "evaluation"
    write_json(output / "closed_proof_confirmation_summary.json", summary)
    if complete:
        _plot(
            output / "closed_proof_confirmation.png",
            results=results,
            primary=primary,
        )
    return summary
