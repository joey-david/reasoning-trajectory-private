"""Causal code substitution analysis from saved opaque-consumer calls."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json

from .metrics import bootstrap_mean_ci
from .state_handoff_data import TEST_PATH, read_programs
from .state_interface_evaluation import (
    interface_evaluation_dir,
    read_interface_evaluation_cases,
)


def _prediction(result: dict[str, Any]) -> int | None:
    value = result.get("unconstrained_prediction")
    return int(value) if value is not None else None


def analyze_interface_interchange(
    run_path: Path, condition: str
) -> dict[str, Any]:
    """Substitute saved donor codes under the recipient's FINAL context."""
    rows = read_interface_evaluation_cases(run_path, condition)
    if not rows:
        raise RuntimeError(f"No interface evaluation rows for {condition}")
    programs = {
        str(row["id"]): row for row in read_programs(run_path / TEST_PATH)
    }
    groups: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["history_steps"]), str(row["program_context"]))].append(row)
    same_preserves = []
    different_follows = []
    different_preserves = []
    matrix_values: defaultdict[tuple[int, int], list[bool]] = defaultdict(list)
    donor_records = []
    for recipient in rows:
        group = groups[(int(recipient["history_steps"]), str(recipient["program_context"]))]
        recipient_state = int(recipient["current_state"])
        recipient_path = int(recipient["path_code"])
        same = next(
            donor
            for donor in group
            if int(donor["current_state"]) == recipient_state
            and int(donor["path_code"]) != recipient_path
        )
        same_output = _prediction(same["gold_final"])
        recipient_answer = int(programs[recipient["id"]]["next_state"])
        same_preserves.append(same_output == recipient_answer)
        for donor_state in range(8):
            donor = next(
                donor
                for donor in group
                if int(donor["current_state"]) == donor_state
                and int(donor["path_code"]) % 2 == recipient_path % 2
            )
            observed = _prediction(donor["gold_final"])
            donor_answer = int(programs[donor["id"]]["next_state"])
            follows = observed == donor_answer
            matrix_values[(recipient_state, donor_state)].append(follows)
            if donor_state != recipient_state:
                different_follows.append(follows)
                different_preserves.append(observed == recipient_answer)
        donor_records.append(
            {
                "recipient_id": recipient["id"],
                "same_state_donor_id": same["id"],
                "different_state": (recipient_state + 1) % 8,
            }
        )
    matrix = [
        [
            sum(matrix_values[(recipient, donor)])
            / len(matrix_values[(recipient, donor)])
            for donor in range(8)
        ]
        for recipient in range(8)
    ]
    summary = {
        "schema_version": 1,
        "condition": condition,
        "case_count": len(rows),
        "same_state_preservation_accuracy": bootstrap_mean_ci(
            same_preserves, seed=10_100
        ),
        "different_state_donor_follow_accuracy": bootstrap_mean_ci(
            different_follows, seed=10_101
        ),
        "different_state_recipient_preservation_accuracy": bootstrap_mean_ci(
            different_preserves, seed=10_102
        ),
        "interchange_matrix": matrix,
        "donor_contract_sample": donor_records[:32],
        "artifact_reuse_contract": (
            "Within one program context, a saved gold-consumer call depends only on "
            "the substituted code and the shared FINAL rule; no history is present."
        ),
    }
    output = interface_evaluation_dir(run_path, condition)
    write_json(output / "interchange_summary.json", summary)
    _write_interchange_matrix(output / "interchange_matrix.png", matrix)
    return summary


def _write_interchange_matrix(path: Path, matrix: list[list[float]]) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axis.set_xlabel("Donor semantic state")
    axis.set_ylabel("Recipient semantic state")
    axis.set_xticks(range(8))
    axis.set_yticks(range(8))
    figure.colorbar(image, ax=axis, label="Output follows donor state")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
