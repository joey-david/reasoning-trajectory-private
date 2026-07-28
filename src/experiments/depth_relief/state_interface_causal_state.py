"""Reduce the rate, gauge, and proof-depth causal-state tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .state_interface_challenge import (
    _read,
    _write_summary,
    challenge_dir,
    configured_challenge_profiles,
)


def _mean(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def _load_information(run: Path, condition: str) -> dict[str, float]:
    path = run / "evaluation/interfaces" / condition / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text())
    return {
        f"{kind}_{key}": float(value)
        for kind in ("true", "predicted")
        for key, value in summary[f"{kind}_code_information"].items()
    }


def _pooled_horizon_advantage(
    run_path: Path,
    profiles: dict[str, dict[str, Any]],
    sources: list[str],
    condition: str,
    horizon: int,
) -> dict[str, Any]:
    differences: list[int] = []
    clusters: list[str] = []
    for source in sources:
        profile = f"{source}__{condition}__length"
        programs = _read(challenge_dir(run_path, profile) / "programs.jsonl")
        interface = {
            str(row["id"]): row
            for row in _read(
                challenge_dir(run_path, profile) / "interface_cases.jsonl"
            )
        }
        owner = str(profiles[profile]["outcome_owner_profile"])
        outcome = {
            str(row["id"]): row
            for row in _read(
                challenge_dir(run_path, owner) / "outcome_cases.jsonl"
            )
        }
        for program in programs:
            if int(program["history_steps"]) != horizon:
                continue
            case_id = str(program["id"])
            interface_correct = bool(
                interface[case_id]["predicted_final"]
                and interface[case_id]["predicted_final"][
                    "is_expected_unconstrained"
                ]
            )
            outcome_correct = bool(
                outcome[case_id]["conditions"]["one_pass_compose"][
                    "is_expected_unconstrained"
                ]
            )
            differences.append(int(interface_correct) - int(outcome_correct))
            clusters.append(f"{source}:{program['program_context']}")
    return cluster_bootstrap_mean_ci(differences, clusters, seed=85_401)


def _plot_rate(
    path: Path, results: dict[str, dict[str, dict[str, Any]]]
) -> None:
    import matplotlib.pyplot as plt

    conditions = (
        "compressed_3bit",
        "canonical_4bit",
        "padded_5bit",
        "redundant_5bit",
    )
    labels = ("3-bit\nlossy", "4-bit\ncanonical", "5-bit\npadded", "5-bit\naliased")
    full = [results["closure_seed1"][condition]["full_support"] for condition in conditions]
    exact = [100 * _mean(summary, "overall", "exact_state_accuracy") for summary in full]
    final = [100 * _mean(summary, "overall", "interface_accuracy") for summary in full]
    x = list(range(len(conditions)))
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    axis.bar([value - 0.19 for value in x], exact, 0.38, label="Exact state")
    axis.bar([value + 0.19 for value in x], final, 0.38, label="Final answer")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Full-support rate and code control")
    axis.legend(frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_gauge(
    path: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    information: dict[str, dict[str, dict[str, float]]],
    sources: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for source in sources:
        points = []
        for condition, marker, color in (
            ("canonical_4bit", "o", "#4477aa"),
            ("redundant_5bit", "s", "#cc6677"),
        ):
            x = information[source][condition]["true_code_given_state_bits"]
            y = 100 * _mean(
                results[source][condition]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            points.append((x, y))
            axis.scatter(x, y, marker=marker, color=color, s=55)
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#999999",
            linewidth=0.8,
        )
    padded_x = information["closure_seed1"]["padded_5bit"][
        "true_code_given_state_bits"
    ]
    padded_y = 100 * _mean(
        results["closure_seed1"]["padded_5bit"]["balanced_depth"],
        "by_active_transition_count",
        "3",
        "exact_state_accuracy",
    )
    axis.scatter(
        padded_x,
        padded_y,
        marker="^",
        color="#228833",
        s=65,
        label="5-bit padded",
    )
    axis.scatter([], [], marker="o", color="#4477aa", label="4-bit canonical")
    axis.scatter([], [], marker="s", color="#cc6677", label="5-bit aliased")
    axis.set_xlabel("Imposed code alias entropy, H(C|S) (bits)")
    axis.set_ylabel("Depth-3 exact state accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Path aliases add a gauge-fixing burden")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_depth_length(
    path: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    sources: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    horizons = (16, 64, 128, 256)
    for metric, label, color in (
        ("interface_accuracy", "State interface", "#4477aa"),
        ("outcome_accuracy", "One pass", "#cc6677"),
    ):
        axes[0].plot(
            horizons,
            [
                100
                * sum(
                    _mean(
                        results[source]["canonical_4bit"]["length"],
                        "by_surface_horizon",
                        str(horizon),
                        metric,
                    )
                    for source in sources
                )
                / len(sources)
                for horizon in horizons
            ],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_title("Surface length at fixed active depth")
    axes[0].set_xlabel("Rules in stream")
    axes[0].set_ylabel("Final accuracy (%)")
    axes[0].legend(frameon=False)

    depths = (1, 2, 3)
    for condition, label, color in (
        ("canonical_4bit", "Canonical", "#4477aa"),
        ("redundant_5bit", "Aliased", "#cc6677"),
    ):
        axes[1].plot(
            depths,
            [
                100
                * sum(
                    _mean(
                        results[source][condition]["balanced_depth"],
                        "by_active_transition_count",
                        str(depth),
                        "exact_state_accuracy",
                    )
                    for source in sources
                )
                / len(sources)
                for depth in depths
            ],
            marker="o",
            label=label,
            color=color,
        )
    axes[1].set_title("Active deductions at fixed length")
    axes[1].set_xlabel("State-changing deductions")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_ylim(0, 105)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_causal_state_phase(run_path: Path) -> dict[str, Any]:
    """Apply the locked causal-state tests and render their three figures."""
    config = load_config(run_path)
    experiment = config["state_interface_causal_state"]
    profiles = configured_challenge_profiles(run_path)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for profile in profiles:
        summary = _write_summary(run_path, profile)
        if not summary.get("complete"):
            missing.append(profile)
            continue
        source, condition, template = profile.split("__", 2)
        results.setdefault(source, {}).setdefault(condition, {})[template] = summary

    sources = [str(value) for value in experiment["primary_sources"]]
    condition_names = {
        str(key): str(value) for key, value in experiment["conditions"].items()
    }
    complete = not missing
    checks: dict[str, bool] = {"all_profiles_complete": complete}
    metrics: dict[str, Any] = {}
    if complete:
        information: dict[str, dict[str, dict[str, float]]] = {}
        for source_spec in config["state_interface_challenge_matrix"]["sources"]:
            source = str(source_spec["name"])
            interface_run = Path(str(source_spec["interface_run"]))
            for condition_spec in source_spec["conditions"]:
                condition = (
                    str(condition_spec)
                    if isinstance(condition_spec, str)
                    else str(condition_spec["name"])
                )
                information.setdefault(source, {})[condition] = _load_information(
                    interface_run, condition
                )
        canonical = condition_names["canonical"]
        aliased = condition_names["aliased"]
        lossy = condition_names["lossy"]
        padded = condition_names["padded"]
        canonical_full = [
            _mean(results[source][canonical]["full_support"], "overall", "exact_state_accuracy")
            for source in sources
        ]
        canonical_depth3 = [
            _mean(
                results[source][canonical]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        canonical_h256 = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "256",
                "interface_accuracy",
            )
            for source in sources
        ]
        length_drop = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "16",
                "interface_accuracy",
            )
            - canonical_h256[index]
            for index, source in enumerate(sources)
        ]
        alias_depth3_gap = [
            canonical_depth3[index]
            - _mean(
                results[source][aliased]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            for index, source in enumerate(sources)
        ]
        seed1 = sources[0]
        padded_gaps = [
            abs(
                _mean(results[seed1][canonical][template], *keys)
                - _mean(results[seed1][padded][template], *keys)
            )
            for template, keys in (
                ("full_support", ("overall", "exact_state_accuracy")),
                (
                    "balanced_depth",
                    ("by_active_transition_count", "3", "exact_state_accuracy"),
                ),
                (
                    "length",
                    ("by_surface_horizon", "256", "interface_accuracy"),
                ),
            )
        ]
        lossy_gap = canonical_full[0] - _mean(
            results[seed1][lossy]["full_support"],
            "overall",
            "exact_state_accuracy",
        )
        h256_advantage = _pooled_horizon_advantage(
            run_path, profiles, sources, canonical, 256
        )
        metrics = {
            "canonical_full_support_exact": {
                "per_seed": canonical_full,
                **bootstrap_mean_ci(canonical_full, seed=85_101),
            },
            "canonical_depth3_exact": {
                "per_seed": canonical_depth3,
                **bootstrap_mean_ci(canonical_depth3, seed=85_102),
            },
            "canonical_h256_final": {
                "per_seed": canonical_h256,
                **bootstrap_mean_ci(canonical_h256, seed=85_103),
            },
            "canonical_h256_minus_outcome": h256_advantage,
            "canonical_h16_minus_h256": {
                "per_seed": length_drop,
                **bootstrap_mean_ci(length_drop, seed=85_104),
            },
            "canonical_minus_aliased_depth3_exact": {
                "per_seed": alias_depth3_gap,
                **bootstrap_mean_ci(alias_depth3_gap, seed=85_105),
            },
            "canonical_minus_lossy_full_support_exact_seed1": lossy_gap,
            "max_canonical_padded_gap_seed1": max(padded_gaps),
            "predicted_code_information": information,
        }
        gates = experiment["gates"]
        checks.update(
            canonical_full_support=all(
                value >= float(gates["min_canonical_full_support_exact_each_seed"])
                for value in canonical_full
            ),
            canonical_active_depth=all(
                value >= float(gates["min_canonical_depth3_exact_each_seed"])
                for value in canonical_depth3
            ),
            canonical_long_horizon=all(
                value >= float(gates["min_canonical_h256_final_each_seed"])
                for value in canonical_h256
            ),
            canonical_beats_one_pass=(
                h256_advantage["mean"]
                >= float(gates["min_mean_canonical_h256_minus_outcome"])
                and h256_advantage["ci95"][0]
                > float(gates["min_canonical_h256_advantage_ci_low"])
            ),
            surface_length_stable=(
                sum(length_drop) / len(length_drop)
                <= float(gates["max_mean_canonical_h16_to_h256_drop"])
            ),
            below_rate_is_lossy=(
                lossy_gap >= float(gates["min_lossy_exact_gap_seed1"])
            ),
            unused_capacity_is_not_harmful=(
                max(padded_gaps) <= float(gates["max_padded_exact_gap_seed1"])
            ),
            aliases_are_harmful=(
                sum(alias_depth3_gap) / len(alias_depth3_gap)
                >= float(gates["min_mean_alias_exact_gap"])
            ),
        )

    status = "pending" if not complete else ("passed" if all(checks.values()) else "failed")
    summary = {
        "schema_version": 1,
        "hypothesis": (
            "Reusable proof state requires a sufficient, canonical, "
            "transition-closed interface; active state changes, not padded "
            "surface length, set the extrapolation burden."
        ),
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "missing_profiles": missing,
        "profile_count": len(profiles),
        "complete_profile_count": len(profiles) - len(missing),
    }
    output = run_path / "evaluation"
    write_json(output / "causal_state_phase_summary.json", summary)
    if complete:
        _plot_rate(output / "rate_sufficiency.png", results)
        _plot_gauge(
            output / "gauge_closure.png",
            results=results,
            information=information,
            sources=sources,
        )
        _plot_depth_length(
            output / "depth_length.png", results=results, sources=sources
        )
    return summary
