"""Saved-artifact reports and plots for state-handoff experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _frozen_screen_summary(run_path: Path) -> dict[str, Any]:
    factorization = json.loads(
        (run_path / "depth_relief/factorization_summary.json").read_text()
    )
    handoff = json.loads(
        (run_path / "depth_relief/explicit_handoff/summary.json").read_text()
    )
    return {
        "read_accuracy": factorization["controls"]["read"],
        "update_accuracy": factorization["controls"]["update"],
        "constituent_step_accuracy": factorization["controls"]["constituent_steps"],
        "by_horizon": {
            horizon: {
                "one_pass_compose_accuracy": factorization["by_history"][horizon][
                    "accuracy"
                ]["compose"],
                "synthesize_accuracy": factorization["by_history"][horizon][
                    "accuracy"
                ]["synthesize"],
                "gold_handoff_accuracy": values["conditions"]["gold_handoff"][
                    "accuracy"
                ],
                "self_handoff_accuracy": values["conditions"]["lm_self_handoff"][
                    "accuracy"
                ],
                "stepwise_handoff_accuracy": values["conditions"][
                    "stepwise_explicit"
                ]["accuracy"],
            }
            for horizon, values in handoff["by_horizon"].items()
        },
    }


def _write_comparison_plots(run_path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    horizons = [2, 4, 8]
    x = np.arange(len(horizons))
    width = 0.25
    figure, axis = plt.subplots(figsize=(7, 4))
    for offset, (key, label) in enumerate(
        (
            ("outcome_only", "Outcome only"),
            ("explicit_handoff_predicted", "Predicted handoff"),
            ("explicit_handoff_gold", "Gold handoff"),
        )
    ):
        axis.bar(
            x + (offset - 1) * width,
            [summary["horizon_accuracy"][str(horizon)][key] for horizon in horizons],
            width,
            label=label,
        )
    axis.set_xticks(x, [f"h{horizon}" for horizon in horizons])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_path / "evaluation/horizon_accuracy.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        [f"h{horizon}" for horizon in horizons],
        [
            summary["horizon_accuracy"][str(horizon)]["explicit_handoff_gold"]
            - summary["horizon_accuracy"][str(horizon)][
                "explicit_handoff_predicted"
            ]
            for horizon in horizons
        ],
    )
    axis.set_ylabel("Gold minus predicted handoff accuracy")
    axis.axhline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(run_path / "evaluation/handoff_gap.png", dpi=160)
    plt.close(figure)
