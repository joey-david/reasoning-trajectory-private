"""Exact discrete information measures for state-interface artifacts."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Hashable, Iterable

from src.runtime.artifact_store import write_json

from .state_handoff_continuation import read_continuation_cases
from .state_handoff_evaluation import read_evaluation_cases


INFORMATION_PATH = Path("evaluation/information_summary.json")


def discrete_entropy(values: Iterable[Hashable]) -> float:
    """Return empirical Shannon entropy in bits."""
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def conditional_entropy(
    values: Iterable[tuple[Hashable, Hashable]],
) -> float:
    """Return H(left | right) from exact discrete counts."""
    rows = list(values)
    if not rows:
        return 0.0
    right_counts = Counter(right for _, right in rows)
    pair_counts = Counter(rows)
    total = len(rows)
    result = 0.0
    for (left, right), count in pair_counts.items():
        joint = count / total
        conditional = count / right_counts[right]
        result -= joint * math.log2(conditional)
    return result


def mutual_information(values: Iterable[tuple[Hashable, Hashable]]) -> float:
    """Return exact empirical mutual information in bits."""
    rows = list(values)
    return discrete_entropy(left for left, _ in rows) - conditional_entropy(rows)


def conditional_mutual_information(
    values: Iterable[tuple[Hashable, Hashable, Hashable]],
) -> float:
    """Return I(left; middle | right) from exact discrete counts."""
    rows = list(values)
    state_given_context = conditional_entropy(
        (left, right) for left, _, right in rows
    )
    state_given_code_context = conditional_entropy(
        (left, (middle, right)) for left, middle, right in rows
    )
    return state_given_context - state_given_code_context


def rate_capacity_table(state_count: int = 8) -> list[dict[str, Any]]:
    """Return deterministic lossless ceilings around the true state entropy."""
    if state_count < 2 or state_count & (state_count - 1):
        raise ValueError("The exact rate table requires a power-of-two state count")
    true_bits = math.log2(state_count)
    return [
        {
            "codebook_size": codebook,
            "capacity_bits": math.log2(codebook),
            "relative_to_state_entropy_bits": math.log2(codebook) - true_bits,
            "deterministic_balanced_state_accuracy_ceiling": min(
                1.0, codebook / state_count
            ),
            "lossless_possible": codebook >= state_count,
        }
        for codebook in (2, 4, 8, 16)
    ]


def _predicted_code(row: dict[str, Any]) -> int | None:
    value = row["conditions"]["state"].get("unconstrained_prediction")
    return int(value) if value is not None else None


def summarize_code_information(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure state information and excess within-state code variation."""
    valid = [row for row in rows if _predicted_code(row) is not None]
    states = [int(row["current_state"]) for row in valid]
    codes = [int(_predicted_code(row)) for row in valid]
    state_code = list(zip(states, codes))
    code_entropy = discrete_entropy(codes)
    state_entropy = discrete_entropy(states)
    state_given_code = conditional_entropy(state_code)
    code_given_state = conditional_entropy((code, state) for state, code in state_code)
    state_information = mutual_information(state_code)
    return {
        "row_count": len(rows),
        "valid_code_count": len(valid),
        "invalid_code_count": len(rows) - len(valid),
        "state_entropy_bits": state_entropy,
        "code_entropy_bits": code_entropy,
        "effective_codebook_size": 2**code_entropy,
        "state_given_code_bits": state_given_code,
        "code_given_state_bits": code_given_state,
        "state_information_bits": state_information,
        "fraction_of_state_information_retained": (
            state_information / state_entropy if state_entropy else None
        ),
        "path_invariance_exact": code_given_state == 0.0,
    }


def _continuation_information(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = {}
    for block_size, horizon in sorted(
        {(int(row["block_size"]), int(row["history_steps"])) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if int(row["block_size"]) == block_size
            and int(row["history_steps"]) == horizon
        ]
        valid = [row for row in selected if row["predicted_state"] is not None]
        pairs = [
            (int(row["current_state"]), int(row["predicted_state"]))
            for row in valid
        ]
        step_correct = [
            bool(step["locally_correct"])
            for row in selected
            for step in row["predicted_steps"]
        ]
        cells[f"block{block_size}_h{horizon}"] = {
            "state_information_bits": mutual_information(pairs),
            "state_given_code_bits": conditional_entropy(pairs),
            "code_given_state_bits": conditional_entropy(
                (code, state) for state, code in pairs
            ),
            "local_transition_error_rate": (
                1 - sum(step_correct) / len(step_correct) if step_correct else None
            ),
            "end_to_end_state_error_rate": 1
            - sum(bool(row["state_correct"]) for row in selected) / len(selected),
            "end_to_end_answer_error_rate": 1
            - sum(
                bool(row["final"]["is_expected_unconstrained"])
                for row in selected
            )
            / len(selected),
        }
    return cells


def analyze_state_handoff_information(
    run_path: Path, *, continuation_profile: str = "probe"
) -> dict[str, Any]:
    """Analyze saved pilot rows and optional recursive continuation rows."""
    explicit = read_evaluation_cases(run_path, "explicit_handoff")
    if not explicit:
        raise RuntimeError("Explicit-handoff evaluation artifacts are missing")
    by_horizon = {
        str(horizon): summarize_code_information(
            [row for row in explicit if int(row["history_steps"]) == horizon]
        )
        for horizon in sorted({int(row["history_steps"]) for row in explicit})
    }
    continuation = read_continuation_cases(run_path, continuation_profile)
    summary = {
        "schema_version": 1,
        "state_count": 8,
        "true_state_entropy_bits": 3.0,
        "rate_capacity_table": rate_capacity_table(),
        "terminal_handoff_by_horizon": by_horizon,
        "continuation_profile": continuation_profile,
        "recursive_by_cell": (
            _continuation_information(continuation) if continuation else None
        ),
        "contracts": {
            "sufficiency": "H(answer | code, future rule) = 0",
            "minimality": "capacity reaches but need not exceed H(state) = 3 bits",
            "invariance": "H(code | state) = 0",
            "closure": "the emitted code remains a valid input to the same transition map",
        },
    }
    write_json(run_path / INFORMATION_PATH, summary)
    _write_information_plots(run_path, summary)
    return summary


def _write_information_plots(run_path: Path, summary: dict[str, Any]) -> None:
    """Render the rate ceiling and observed terminal information profiles."""
    import matplotlib.pyplot as plt

    rate = summary["rate_capacity_table"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [row["capacity_bits"] for row in rate],
        [row["deterministic_balanced_state_accuracy_ceiling"] for row in rate],
        marker="o",
    )
    axis.axvline(3, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Interface capacity (bits)")
    axis.set_ylabel("Exact state-accuracy ceiling")
    axis.set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(run_path / "evaluation/rate_capacity.png", dpi=160)
    plt.close(figure)

    horizons = sorted(
        int(value) for value in summary["terminal_handoff_by_horizon"]
    )
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        horizons,
        [
            summary["terminal_handoff_by_horizon"][str(horizon)][
                "state_information_bits"
            ]
            for horizon in horizons
        ],
        marker="o",
        label="I(state; code)",
    )
    axis.plot(
        horizons,
        [
            summary["terminal_handoff_by_horizon"][str(horizon)][
                "code_given_state_bits"
            ]
            for horizon in horizons
        ],
        marker="o",
        label="H(code | state)",
    )
    axis.axhline(3, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(horizons)
    axis.set_xlabel("History horizon")
    axis.set_ylabel("Bits")
    axis.legend()
    figure.tight_layout()
    figure.savefig(run_path / "evaluation/state_information.png", dpi=160)
    plt.close(figure)
