"""Reduce the balanced multi-seed, multi-scale proof-state confirmation."""

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


def _summary(run_path: Path, profile: str) -> dict[str, Any] | None:
    path = challenge_dir(run_path, profile) / "summary.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    return result if result.get("complete") else None


def _mean(summary: dict[str, Any], section: str, key: str, metric: str) -> float:
    return float(summary[section][key][metric]["mean"])


def _plot_confirmation(
    output: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    primary_sources: list[str],
    scale_sources: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    colors = {
        "compressed_3bit": "#cc6677",
        "canonical_4bit": "#4477aa",
        "redundant_5bit": "#228833",
    }
    labels = {
        "compressed_3bit": "3-bit compressed",
        "canonical_4bit": "4-bit canonical",
        "redundant_5bit": "5-bit redundant",
    }
    for condition in labels:
        values = []
        depths = [1, 2, 3, 4]
        for depth in depths:
            per_seed = [
                _mean(
                    results[source][condition]["depth"],
                    "by_active_transition_count",
                    str(depth),
                    "semantic_state_accuracy",
                )
                for source in primary_sources
                if condition in results.get(source, {})
                and "depth" in results[source][condition]
            ]
            values.append(
                100 * sum(per_seed) / len(per_seed) if per_seed else float("nan")
            )
        axes[0].plot(
            depths,
            values,
            marker="o",
            label=labels[condition],
            color=colors[condition],
        )
    axes[0].set_title("State recovery vs deduction depth")
    axes[0].set_xlabel("Active deductions")
    axes[0].set_ylabel("Semantic state accuracy (%)")
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].legend(frameon=False, fontsize=8)

    for source in scale_sources:
        summary = results.get(source, {}).get("redundant_5bit", {}).get("length")
        if summary:
            horizons = [16, 64, 128]
            axes[1].plot(
                horizons,
                [
                    100
                    * _mean(
                        summary,
                        "by_surface_horizon",
                        str(horizon),
                        "semantic_state_accuracy",
                    )
                    for horizon in horizons
                ],
                marker="o",
                label=source,
            )
    axes[1].set_title("Surface length at fixed depth mix")
    axes[1].set_xlabel("Rules in prompt")
    axes[1].set_ylabel("Semantic state accuracy (%)")
    axes[1].set_xticks([16, 64, 128])
    if axes[1].lines:
        axes[1].legend(frameon=False, fontsize=8)

    topologies = ["independent", "chain", "conjunction"]
    primary_topology = [
        (
            source,
            results.get(source, {}).get("redundant_5bit", {}).get("topology"),
        )
        for source in primary_sources
    ]
    values = []
    for topology in topologies:
        per_seed = [
            _mean(
                summary,
                "by_proof_topology",
                topology,
                "semantic_state_accuracy",
            )
            for _, summary in primary_topology
            if summary
        ]
        values.append(100 * sum(per_seed) / len(per_seed) if per_seed else float("nan"))
    axes[2].bar(topologies, values, color=["#66ccee", "#4477aa", "#228833"])
    axes[2].set_title("Held-out proof topology, depth 3-4")
    axes[2].set_ylabel("Semantic state accuracy (%)")
    axes[2].tick_params(axis="x", rotation=15)

    for axis in axes:
        axis.set_ylim(0, 105)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compare_proof_confirmation(run_path: Path) -> dict[str, Any]:
    """Aggregate completed profiles and apply the prespecified proof gates."""
    config = load_config(run_path)
    confirmation = config["state_interface_proof_confirmation"]
    profiles = configured_challenge_profiles(run_path)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for profile in profiles:
        source, condition, template = profile.split("__", 2)
        _write_summary(run_path, profile)
        summary = _summary(run_path, profile)
        if summary is None:
            missing.append(profile)
            continue
        results.setdefault(source, {}).setdefault(condition, {})[template] = summary

    primary = [str(value) for value in confirmation["primary_sources"]]
    gates = confirmation["gates"]
    required = [
        (source, condition, template)
        for source in primary
        for condition, templates in {
            "compressed_3bit": ("depth",),
            "canonical_4bit": ("depth", "length", "topology"),
            "redundant_5bit": ("depth", "length", "topology"),
        }.items()
        for template in templates
    ]
    primary_complete = all(
        template in results.get(source, {}).get(condition, {})
        for source, condition, template in required
    )
    checks: dict[str, bool] = {"primary_profiles_complete": primary_complete}
    metrics: dict[str, Any] = {}
    if primary_complete:
        redundant_state = [
            _mean(
                results[source]["redundant_5bit"]["depth"],
                "by_active_transition_count",
                "4",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        redundant_answer = [
            _mean(
                results[source]["redundant_5bit"]["depth"],
                "by_active_transition_count",
                "4",
                "interface_accuracy",
            )
            for source in primary
        ]
        canonical_state = [
            _mean(
                results[source]["canonical_4bit"]["depth"],
                "by_active_transition_count",
                "4",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        differences = [
            left - right for left, right in zip(redundant_state, canonical_state)
        ]
        length_drops = [
            _mean(
                results[source]["redundant_5bit"]["length"],
                "by_surface_horizon",
                "16",
                "semantic_state_accuracy",
            )
            - _mean(
                results[source]["redundant_5bit"]["length"],
                "by_surface_horizon",
                "128",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        conjunction = [
            _mean(
                results[source]["redundant_5bit"]["topology"],
                "by_proof_topology",
                "conjunction",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        compressed = [
            _mean(
                results[source]["compressed_3bit"]["depth"],
                "by_active_transition_count",
                "4",
                "semantic_state_accuracy",
            )
            for source in primary
        ]
        metrics = {
            "redundant_depth4_state": {
                "per_seed": redundant_state,
                **bootstrap_mean_ci(redundant_state, seed=83_801),
            },
            "redundant_depth4_answer": {
                "per_seed": redundant_answer,
                **bootstrap_mean_ci(redundant_answer, seed=83_802),
            },
            "redundant_minus_canonical_depth4_state": {
                "per_seed": differences,
                **bootstrap_mean_ci(differences, seed=83_803),
            },
            "h16_minus_h128_state": {
                "per_seed": length_drops,
                **bootstrap_mean_ci(length_drops, seed=83_804),
            },
            "redundant_conjunction_state": {
                "per_seed": conjunction,
                **bootstrap_mean_ci(conjunction, seed=83_805),
            },
            "compressed_depth4_state": {
                "per_seed": compressed,
                **bootstrap_mean_ci(compressed, seed=83_806),
            },
        }
        checks.update(
            redundant_state_each_seed=(
                min(redundant_state)
                >= float(gates["min_redundant_depth4_semantic_accuracy_each_seed"])
            ),
            redundant_answer_each_seed=(
                min(redundant_answer)
                >= float(gates["min_redundant_depth4_answer_accuracy_each_seed"])
            ),
            redundant_beats_canonical=(
                sum(differences) / len(differences)
                >= float(gates["min_mean_redundant_minus_canonical_depth4"])
            ),
            length_stable=(
                sum(length_drops) / len(length_drops)
                <= float(gates["max_mean_h128_minus_h16_drop"])
            ),
            conjunction_transfer=(
                min(conjunction) >= float(gates["min_redundant_conjunction_accuracy"])
            ),
            compressed_rate_control=(
                max(compressed)
                <= float(gates["max_compressed_depth4_semantic_accuracy"])
            ),
        )

    output_dir = run_path / "evaluation"
    result = {
        "schema_version": 1,
        "status": (
            "passed"
            if checks and all(checks.values())
            else ("failed" if primary_complete else "partial")
        ),
        "profile_count": len(profiles),
        "complete_profile_count": len(profiles) - len(missing),
        "missing_profiles": missing,
        "primary_sources": primary,
        "scale_sources": list(confirmation["scale_sources"]),
        "family_sources": list(confirmation["family_sources"]),
        "metrics": metrics,
        "checks": checks,
        "results": results,
        "uncertainty_contract": (
            "Each profile uses program-context bootstrap intervals. Primary "
            "aggregate intervals resample three independent Qwen-7B seeds."
        ),
    }
    write_json(output_dir / "proof_confirmation_summary.json", result)
    _plot_confirmation(
        output_dir / "proof_confirmation.png",
        results=results,
        primary_sources=primary,
        scale_sources=[str(value) for value in confirmation["scale_sources"]],
    )
    return result
