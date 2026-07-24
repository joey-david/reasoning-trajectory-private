#!/usr/bin/env python3
"""Build the paper-facing state-handoff evidence index and figures."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "state_handoff_paper"
FIGURES = OUT / "figures"

QWEN_32B = (
    ROOT
    / "runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history"
)
QWEN_7B = ROOT / "runs/Qwen2.5-7B-Instruct/interventions"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mean(value: dict[str, Any]) -> float:
    return float(value["mean"])


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def style_axis(axis: plt.Axes, *, ylabel: str = "Accuracy (%)") -> None:
    axis.set_ylim(0, 105)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_diagnosis() -> dict[str, Any]:
    path = QWEN_32B / "depth_relief/factorization_summary.json"
    data = read_json(path)
    h2 = data["by_history"]["2"]["accuracy"]
    labels = ["Read", "Update", "Synthesize", "Compose"]
    values = [100 * mean(h2[key.lower()]) for key in labels]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.bar(labels, values, color=["#4477aa", "#4477aa", "#66ccee", "#cc6677"])
    style_axis(axis)
    axis.set_title("Local operations succeed; serial composition fails (Qwen2.5-32B, h2)")
    for index, value in enumerate(values):
        axis.text(index, value + 2, f"{value:.1f}", ha="center", fontsize=8)
    save_figure(figure, "01_native_factorization.png")
    return {"source": str(path.relative_to(ROOT)), "h2_accuracy": dict(zip(labels, values))}


def plot_handoff() -> dict[str, Any]:
    path = QWEN_32B / "depth_relief/explicit_handoff/summary.json"
    data = read_json(path)
    horizons = [2, 4]
    keys = ["one_pass_compose", "lm_self_handoff", "gold_handoff", "stepwise_explicit"]
    labels = ["One pass", "Self handoff", "Gold handoff", "Stepwise"]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    for key, label in zip(keys, labels):
        values = [
            100 * mean(data["by_horizon"][str(horizon)]["conditions"][key]["accuracy"])
            for horizon in horizons
        ]
        axis.plot(horizons, values, marker="o", label=label)
    style_axis(axis)
    axis.set_xticks(horizons)
    axis.set_xlabel("History operations")
    axis.set_title("Deleting history after an explicit state handoff")
    axis.legend(frameon=False, fontsize=8)
    save_figure(figure, "02_explicit_handoff.png")
    return {
        "source": str(path.relative_to(ROOT)),
        "h2_self_handoff": mean(data["by_horizon"]["2"]["conditions"]["lm_self_handoff"]["accuracy"]),
        "h2_one_pass": mean(data["by_horizon"]["2"]["conditions"]["one_pass_compose"]["accuracy"]),
        "gold_handoff": mean(data["overall"]["conditions"]["gold_handoff"]["accuracy"]),
        "stepwise": mean(data["overall"]["conditions"]["stepwise_explicit"]["accuracy"]),
    }


def plot_rate() -> dict[str, Any]:
    path = (
        QWEN_7B
        / "state_interface_rate_controls/evaluation/interfaces/comparison_summary.json"
    )
    data = read_json(path)
    horizons = [2, 4, 8, 16]
    conditions = [
        ("context_bound", "Context-bound", "#999999"),
        ("compressed_2bit", "2-bit", "#cc6677"),
        ("canonical_opaque", "3-bit", "#4477aa"),
        ("redundant_4bit", "4-bit redundant", "#228833"),
    ]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    for key, label, color in conditions:
        values = [100 * data["horizon_accuracy"][key][str(h)] for h in horizons]
        axis.plot(horizons, values, marker="o", label=label, color=color)
    style_axis(axis)
    axis.axhline(50, color="#cc6677", linestyle="--", linewidth=0.8)
    axis.set_xticks(horizons)
    axis.set_xlabel("Recursive operations")
    axis.set_title("Declared channel rate controls recoverable state")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(figure, "03_rate_capacity.png")
    return {"source": str(path.relative_to(ROOT)), "horizon_accuracy": data["horizon_accuracy"]}


def plot_closure() -> dict[str, Any]:
    path = (
        QWEN_7B
        / "state_interface_closure_finetune/evaluation/closure_comparison.json"
    )
    data = read_json(path)
    horizons = [2, 4, 8, 16]
    cells = data["conditions"]["redundant_4bit"]["by_horizon"]
    closure = [100 * cells[str(h)]["answer_accuracy"]["closure"] for h in horizons]
    endpoint = [
        100 * cells[str(h)]["answer_accuracy"]["endpoint_control"] for h in horizons
    ]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.plot(horizons, closure, marker="o", label="Transition closure", color="#228833")
    axis.plot(horizons, endpoint, marker="o", label="Endpoint control", color="#cc6677")
    style_axis(axis)
    axis.set_xticks(horizons)
    axis.set_xlabel("Recursive operations")
    axis.set_title("Training the reusable transition beats fitting endpoints")
    axis.legend(frameon=False, fontsize=8)
    save_figure(figure, "04_closure_training.png")
    return {
        "source": str(path.relative_to(ROOT)),
        "h8_difference": cells["8"]["answer_difference"],
        "h16_difference": cells["16"]["answer_difference"],
        "matched_compute": data["matched_compute"],
    }


def stress_average(summary: dict[str, Any]) -> dict[int, float]:
    values: dict[int, list[float]] = defaultdict(list)
    for cell, metrics in summary["by_cell"].items():
        match = re.search(r"_h(2|4|8|16)$", cell)
        if match:
            values[int(match.group(1))].append(mean(metrics["answer_accuracy"]))
    return {horizon: sum(items) / len(items) for horizon, items in values.items()}


def plot_stress() -> dict[str, Any]:
    path = (
        QWEN_7B
        / "state_interface_closure_stress/evaluation/stress/probe/comparison_summary.json"
    )
    data = read_json(path)
    horizons = [2, 4, 8, 16]
    closure = stress_average(data["summaries"]["closure_redundant"])
    endpoint = stress_average(data["summaries"]["endpoint_redundant"])
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    axis.plot(
        horizons,
        [100 * closure[h] for h in horizons],
        marker="o",
        label="Transition closure",
        color="#228833",
    )
    axis.plot(
        horizons,
        [100 * endpoint[h] for h in horizons],
        marker="o",
        label="Endpoint control",
        color="#cc6677",
    )
    style_axis(axis)
    axis.set_xticks(horizons)
    axis.set_xlabel("Recursive operations")
    axis.set_title("Average over five shifted history families")
    axis.legend(frameon=False, fontsize=8)
    save_figure(figure, "05_distribution_shift.png")
    return {"source": str(path.relative_to(ROOT)), "closure": closure, "endpoint": endpoint}


def plot_register() -> dict[str, Any]:
    path = (
        QWEN_7B
        / "state_interface_register_confirm_seed1/evaluation/replication_summary.json"
    )
    data = read_json(path)
    global_interface = 100 * data["metrics"]["interface_answer_accuracy"]["mean"]
    global_outcome = 100 * data["metrics"]["outcome_answer_accuracy"]["mean"]
    local_values = []
    for seed in (1, 2, 3):
        seed_path = (
            QWEN_7B
            / f"state_interface_register_confirm_seed{seed}/evaluation/generalization_summary.json"
        )
        cell = read_json(seed_path)["cells"][
            "canonical_4bit/register_machine/heldout/h32"
        ]
        local_values.append(mean(cell["conditional_semantic_transition_accuracy"]))
    local = 100 * sum(local_values) / len(local_values)
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    labels = ["One-step transition", "Final interface", "One-pass control"]
    values = [local, global_interface, global_outcome]
    axis.bar(labels, values, color=["#4477aa", "#cc6677", "#999999"])
    style_axis(axis)
    axis.set_title("Small local errors compound over mixed h32 programs")
    axis.tick_params(axis="x", labelrotation=12)
    for index, value in enumerate(values):
        axis.text(index, value + 2, f"{value:.1f}", ha="center", fontsize=8)
    save_figure(figure, "06_local_global_reliability.png")
    return {
        "source": str(path.relative_to(ROOT)),
        "conditional_transition_accuracy": local / 100,
        "final_interface_accuracy": global_interface / 100,
        "outcome_accuracy": global_outcome / 100,
        "gate": data["gate"],
    }


def plot_proof_depth() -> dict[str, Any]:
    base = QWEN_7B / "state_interface_proof_depth_fullrate/evaluation/challenges"
    paths = {
        "Canonical 4-bit": base / "proof_depth_h64_canonical4/summary.json",
        "Redundant 5-bit": base / "proof_depth_h64_redundant5/summary.json",
    }
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    evidence: dict[str, Any] = {}
    for label, path in paths.items():
        data = read_json(path)
        depths = sorted(int(key) for key in data["by_active_transition_count"])
        values = [
            100
            * mean(data["by_active_transition_count"][str(depth)]["interface_accuracy"])
            for depth in depths
        ]
        axis.plot(depths, values, marker="o", label=label)
        evidence[label] = {
            "source": str(path.relative_to(ROOT)),
            "overall_interface": data["interface"]["accuracy"],
            "overall_outcome": data["outcome"]["accuracy"],
            "by_active_transition_count": data["by_active_transition_count"],
        }
    style_axis(axis)
    axis.axhline(50, color="#999999", linestyle="--", linewidth=0.8, label="Binary chance")
    axis.set_xticks([0, 1, 2, 3, 4])
    axis.set_xlabel("State-changing deductions (surface length fixed)")
    axis.set_title("Redundant code survives the deepest proof stratum")
    axis.legend(frameon=False, fontsize=8)
    save_figure(figure, "07_proof_active_depth.png")
    return evidence


def plot_new_register_rate() -> dict[str, Any]:
    paths = [
        QWEN_7B
        / f"state_interface_register_redundant5_seed{seed}"
        / "evaluation/interfaces/redundant_5bit/summary.json"
        for seed in (1, 2)
    ]
    canonical_replication = read_json(
        QWEN_7B
        / "state_interface_register_confirm_seed1/evaluation/replication_summary.json"
    )
    canonical = 100 * canonical_replication["metrics"]["interface_answer_accuracy"]["mean"]
    redundant = [
        100 * mean(read_json(path)["by_horizon"]["32"]["predicted_answer_accuracy"])
        for path in paths
    ]
    figure, axis = plt.subplots(figsize=(5.4, 3.2))
    labels = ["4-bit\n3-seed mean", "5-bit\nseed 1", "5-bit\nseed 2"]
    values = [canonical, *redundant]
    axis.bar(labels, values, color=["#4477aa", "#228833", "#228833"])
    style_axis(axis)
    axis.axhline(6.25, color="#999999", linestyle="--", linewidth=0.8)
    axis.set_title("One extra bit does not repair free-form mixed h32 execution")
    for index, value in enumerate(values):
        axis.text(index, value + 1.5, f"{value:.1f}", ha="center", fontsize=8)
    save_figure(figure, "08_register_redundancy_negative.png")
    return {
        "sources": [str(path.relative_to(ROOT)) for path in paths],
        "canonical_4bit_three_seed_mean": canonical / 100,
        "redundant_5bit_completed_seeds": [value / 100 for value in redundant],
        "seed3_status": "training complete; evaluation partial at 510/640; excluded",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": 1,
        "parts": {
            "1_native_diagnosis": plot_diagnosis(),
            "2_explicit_handoff": plot_handoff(),
            "3_rate_capacity": plot_rate(),
            "4_transition_closure": plot_closure(),
            "5_distribution_shift": plot_stress(),
            "6_register_limit": plot_register(),
            "7_proof_depth": plot_proof_depth(),
            "8_redundancy_limit": plot_new_register_rate(),
        },
        "status_note": (
            "Only complete summaries enter aggregate metrics. "
            "Register redundant-5-bit seed 3 is excluded because evaluation stopped at 510/640."
        ),
    }
    with (OUT / "evidence.json").open("w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(OUT), "figure_count": 8}, indent=2))


if __name__ == "__main__":
    main()
