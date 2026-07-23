"""Saved comparison reports for rate-controlled state interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


def compare_state_interface_conditions(run_path: Path) -> dict[str, Any]:
    """Compare equal-compute code contracts and apply the continuation gate."""
    from .state_interface_evaluation import (
        INTERFACE_EVALUATION_ROOT,
        interface_evaluation_dir,
    )

    config = load_config(run_path).get("state_handoff_training", {})
    conditions = tuple(str(value) for value in config.get("conditions", ()))
    summaries = {}
    for condition in conditions:
        path = interface_evaluation_dir(run_path, condition) / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing interface evaluation summary: {condition}")
        summary = json.loads(path.read_text())
        if not summary.get("complete"):
            raise RuntimeError(f"Interface evaluation is incomplete: {condition}")
        summaries[condition] = summary
    gate_config = config.get("interface_gate", {})
    primary = str(
        gate_config.get(
            "primary_condition",
            "canonical_opaque"
            if "canonical_opaque" in summaries
            else "canonical_4bit",
        )
    )
    if primary not in summaries:
        raise ValueError(f"Interface comparison lacks primary condition {primary!r}")
    accuracy = {
        condition: {
            horizon: values["predicted_answer_accuracy"]["mean"]
            for horizon, values in summary["by_horizon"].items()
        }
        for condition, summary in summaries.items()
    }
    min_h8 = float(
        gate_config.get(
            "min_primary_h8", gate_config.get("min_canonical_h8", 0.90)
        )
    )
    min_h16 = float(
        gate_config.get(
            "min_primary_h16", gate_config.get("min_canonical_h16", 0.80)
        )
    )
    min_context_gap = float(gate_config.get("min_context_gap", 0.20))
    checks = {
        "primary_h8": accuracy[primary].get("8", 0.0) >= min_h8,
        "primary_h16": accuracy[primary].get("16", 0.0) >= min_h16,
        "primary_gold_consumer": summaries[primary]["by_horizon"]["8"][
            "gold_code_answer_accuracy"
        ]["mean"]
        >= 0.95,
    }
    if "context_bound" in accuracy:
        checks["canonical_beats_context_bound"] = (
            accuracy[primary].get("8", 0.0)
            - accuracy["context_bound"].get("8", 0.0)
            >= min_context_gap
        )
    for condition, thresholds in gate_config.get(
        "required_accuracy", {}
    ).items():
        if condition not in accuracy:
            raise ValueError(f"Interface gate lacks condition {condition!r}")
        for horizon, threshold in thresholds.items():
            checks[f"{condition}_h{horizon}"] = (
                accuracy[condition].get(str(horizon), 0.0) >= float(threshold)
            )
    result = {
        "schema_version": 1,
        "conditions": list(conditions),
        "primary_condition": primary,
        "horizon_accuracy": accuracy,
        "summaries": summaries,
        "interface_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "thresholds": {
                "min_primary_h8": min_h8,
                "min_primary_h16": min_h16,
                "min_context_gap": min_context_gap,
            },
            "checks": checks,
        },
    }
    from .state_interface_interchange import analyze_interface_interchange

    result["interchange"] = {
        condition: analyze_interface_interchange(run_path, condition)
        for condition in conditions
    }
    from .state_interface_equivalence import analyze_predicted_code_equivalence

    result["predicted_equivalence"] = {
        condition: analyze_predicted_code_equivalence(run_path, condition)
        for condition in conditions
    }
    output = run_path / INTERFACE_EVALUATION_ROOT
    write_json(output / "comparison_summary.json", result)
    _write_interface_plot(output / "interface_accuracy.png", result)
    return result


def _write_interface_plot(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    horizons = sorted(
        {
            int(horizon)
            for values in summary["horizon_accuracy"].values()
            for horizon in values
        }
    )
    figure, axis = plt.subplots(figsize=(7, 4))
    for condition, values in summary["horizon_accuracy"].items():
        axis.plot(
            horizons,
            [values.get(str(horizon), float("nan")) for horizon in horizons],
            marker="o",
            label=condition.replace("_", " "),
        )
    first = next(iter(summary["summaries"].values()))
    chance = 2 ** -float(first.get("semantic_state_entropy_bits", 3.0))
    axis.axhline(chance, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(horizons)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("History horizon")
    axis.set_ylabel("Recursive answer accuracy")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
