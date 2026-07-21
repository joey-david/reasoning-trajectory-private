"""Cluster-aware behavioral summary for the matched-history assay."""

from __future__ import annotations

from typing import Any, Callable

from .abstraction_interchange import behavior_qualified
from .metrics import cluster_bootstrap_mean_ci


def _correct(row: dict[str, Any], condition: str) -> bool:
    return bool(row["conditions"][condition]["is_expected_unconstrained"])


def _summarize_subset(
    cases: dict[str, dict[str, Any]], rows: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    def interval(
        selected: list[dict[str, Any]],
        value: Callable[[dict[str, Any]], float],
        offset: int,
    ) -> dict[str, Any]:
        return cluster_bootstrap_mean_ci(
            [value(row) for row in selected],
            [str(cases[str(row["id"])]["abstraction_group"]) for row in selected],
            seed=seed + offset,
        )

    conditions = {
        name: interval(rows, lambda row, name=name: _correct(row, name), index)
        for index, name in enumerate(("read", "update", "synthesize", "compose"))
    }
    local = [
        row
        for row in rows
        if _correct(row, "read")
        and _correct(row, "update")
        and all(
            _correct(row, f"history_step_{step}")
            for step in range(1, int(row["history_steps"]) + 1)
        )
    ]
    qualified = [row for row in rows if behavior_qualified(row)]
    failures = [row for row in qualified if not _correct(row, "compose")]
    local_ids = {str(row["id"]) for row in local}
    qualified_ids = {str(row["id"]) for row in qualified}
    distinguishable = [
        row
        for row in failures
        if int(row["diagnostic_targets"]["final_on_start"])
        != int(row["next_state"])
    ]
    return {
        "case_count": len(rows),
        "condition_accuracy": conditions,
        "local_competence": interval(
            rows, lambda row: str(row["id"]) in local_ids, 10
        ),
        "synthesis_given_local_competence": interval(
            local, lambda row: _correct(row, "synthesize"), 11
        ),
        "causal_qualification": interval(
            rows, lambda row: str(row["id"]) in qualified_ids, 12
        ),
        "compose_given_causal_qualification": interval(
            qualified, lambda row: _correct(row, "compose"), 13
        ),
        "serial_failure_count": len(failures),
        "distinguishable_serial_failure_count": len(distinguishable),
        "final_on_start_given_distinguishable_failure": interval(
            distinguishable,
            lambda row: (
                row["conditions"]["compose"]["unconstrained_prediction"]
                == int(row["diagnostic_targets"]["final_on_start"])
            ),
            14,
        ),
    }


def summarize_abstraction_behavior(
    cases: dict[str, dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Report behavior with program contexts as the inferential unit."""
    if set(cases) != set(rows):
        raise ValueError("Behavior rows do not exactly cover abstraction cases")
    all_rows = list(rows.values())
    horizons = sorted({int(row["history_steps"]) for row in all_rows})
    return {
        "inference_unit": "program context",
        "overall": _summarize_subset(cases, all_rows, seed=3100),
        "by_horizon": {
            str(horizon): _summarize_subset(
                cases,
                [row for row in all_rows if int(row["history_steps"]) == horizon],
                seed=3200 + horizon * 100,
            )
            for horizon in horizons
        },
    }
