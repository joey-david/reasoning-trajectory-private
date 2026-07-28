"""Paper-facing plots for the causal-state confirmation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .state_interface_challenge import _read, challenge_dir


def _mean(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def plot_rate(
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
    full = [
        results["closure_seed1"][condition]["full_support"]
        for condition in conditions
    ]
    exact = [
        100 * _mean(summary, "overall", "exact_state_accuracy")
        for summary in full
    ]
    final = [
        100 * _mean(summary, "overall", "interface_accuracy")
        for summary in full
    ]
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


def plot_gauge(
    path: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    information: dict[str, dict[str, dict[str, Any]]],
    sources: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for source in sources:
        depth_three = []
        for condition in ("padded_5bit", "redundant_5bit"):
            x = information[source][condition]["true"][
                "code_given_state_bits"
            ]
            full_y = 100 * _mean(
                results[source][condition]["full_support"],
                "overall",
                "exact_state_accuracy",
            )
            depth_y = 100 * _mean(
                results[source][condition]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            depth_three.append((x, depth_y))
            axis.scatter(
                x,
                full_y,
                marker="o",
                facecolors="none",
                edgecolors="#777777",
                s=55,
            )
            axis.scatter(x, depth_y, marker="s", color="#cc6677", s=55)
        axis.plot(
            [point[0] for point in depth_three],
            [point[1] for point in depth_three],
            color="#cc6677",
            linewidth=1.0,
        )
    canonical_values = [
        100
        * _mean(
            results[source]["canonical_4bit"]["balanced_depth"],
            "by_active_transition_count",
            "3",
            "exact_state_accuracy",
        )
        for source in sources
    ]
    axis.axhline(
        sum(canonical_values) / len(canonical_values),
        color="#4477aa",
        linestyle="--",
        linewidth=1.2,
        label="4-bit canonical, depth 3",
    )
    axis.scatter(
        [],
        [],
        marker="o",
        facecolors="none",
        edgecolors="#777777",
        label="Full-support coverage",
    )
    axis.scatter([], [], marker="s", color="#cc6677", label="Depth-3 deduction")
    axis.set_xlabel("Imposed code alias entropy, H(C|S) (bits)")
    axis.set_ylabel("Exact state accuracy (%)")
    axis.set_ylim(0, 105)
    axis.set_title("Aliases preserve coverage but break recursive deduction")
    axis.legend(frameon=False, fontsize=8)
    axis.grid(color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_depth_length(
    path: Path,
    *,
    results: dict[str, dict[str, dict[str, Any]]],
    sources: list[str],
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.5))
    horizons = (16, 64, 128, 256)
    for condition, label, color in (
        ("canonical_4bit", "Canonical", "#4477aa"),
        ("redundant_5bit", "Aliased", "#cc6677"),
    ):
        axes[0].plot(
            horizons,
            [
                100
                * sum(
                    _mean(
                        results[source][condition]["length"],
                        "by_surface_horizon",
                        str(horizon),
                        "exact_state_accuracy",
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
    axes[0].set_ylabel("Exact state accuracy (%)")
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
    for metric, label, color in (
        ("interface_accuracy", "State interface", "#4477aa"),
        ("outcome_accuracy", "One pass", "#cc6677"),
    ):
        axes[2].plot(
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
    axes[2].set_title("Behavioral consequence")
    axes[2].set_xlabel("Rules in stream")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.set_ylim(0, 105)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_consumer_basis(path: Path, *, run_path: Path) -> None:
    """Plot the future-query truth table induced by intervened gold codes."""
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = (
        ("compressed_3bit", "3-bit lossy"),
        ("canonical_4bit", "4-bit canonical"),
        ("padded_5bit", "5-bit padded"),
        ("redundant_5bit", "5-bit aliased"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(10.8, 5.0), sharey=True)
    image = None
    for axis, (condition, label) in zip(axes, conditions):
        profile = f"closure_seed1__{condition}__full_support"
        programs = {
            str(row["id"]): row
            for row in _read(challenge_dir(run_path, profile) / "programs.jsonl")
        }
        interface = _read(
            challenge_dir(run_path, profile) / "interface_cases.jsonl"
        )
        cells: list[list[list[float]]] = [
            [[] for _ in range(4)] for _ in range(16)
        ]
        for row in interface:
            program = programs[str(row["id"])]
            prediction = row["gold_final"].get("unconstrained_prediction")
            if prediction is not None:
                cells[int(program["current_state"])][
                    int(program["proof_query_bit"])
                ].append(float(prediction))
        matrix = np.asarray(
            [
                [
                    sum(values) / len(values) if values else np.nan
                    for values in state
                ]
                for state in cells
            ]
        )
        image = axis.imshow(
            matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto"
        )
        for state in range(16):
            for bit in range(4):
                expected = int(bool(state & (1 << bit)))
                observed = matrix[state, bit]
                if np.isnan(observed) or int(observed >= 0.5) != expected:
                    axis.text(
                        bit,
                        state,
                        "×",
                        ha="center",
                        va="center",
                        color="#cc3311",
                        fontsize=8,
                    )
        axis.set_title(label, fontsize=9)
        axis.set_xticks(range(4), [f"b{bit}" for bit in range(4)])
        axis.set_xlabel("Future fact query")
    axes[0].set_yticks(range(16), [str(state) for state in range(16)])
    axes[0].set_ylabel("Intervened semantic state")
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            fraction=0.02,
            pad=0.02,
            label="Predicted yes rate",
        )
    figure.suptitle("Gold-code causal consumer truth table", fontsize=11)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
