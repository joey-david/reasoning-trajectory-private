"""Objective-specific outcomes for matched sentence-boundary interventions."""

from __future__ import annotations

from typing import Any

from src.experiments.boundary_interventions import (
    grouped_mean_interval,
    pair_common_prefix_fraction,
)
from src.experiments.symbolic import extract_symbolic_updates


CAUSAL_OBJECTIVES = ("answer", "object", "correctness", "compression")


def objective_specificity(
    target_pairs: dict[tuple[str, float], tuple[dict[str, Any], dict[str, Any]]],
    control_pairs: dict[tuple[str, float], tuple[dict[str, Any], dict[str, Any]]],
    objective: str,
    *,
    complete_only: bool = True,
) -> dict[str, Any]:
    """Compare objective-matched boundary disruption with random controls.

    Args:
        target_pairs: Objective-family baseline/ablation pairs by question and position.
        control_pairs: Position-matched random baseline/ablation pairs.
        objective: Outcome objective to score.
        complete_only: Whether any token-limit pair must be excluded.

    Returns:
        Question-balanced target-minus-random disruption and coverage.
    """
    if objective not in CAUSAL_OBJECTIVES:
        raise ValueError(f"Unsupported causal objective: {objective!r}")
    keys = sorted(target_pairs.keys() & control_pairs.keys())
    if complete_only:
        keys = [
            key
            for key in keys
            if not any(
                row["hit_token_limit"]
                for row in (*target_pairs[key], *control_pairs[key])
            )
        ]
    groups = [sample_id for sample_id, _position in keys]
    differences = [
        pair_disruption(target_pairs[key], objective)
        - pair_disruption(control_pairs[key], objective)
        for key in keys
    ]
    return {
        "objective": objective,
        "outcome": outcome_definition(objective),
        "complete_only": complete_only,
        "matched_points": len(keys),
        "questions": len(set(groups)),
        "target_minus_random": grouped_mean_interval(differences, groups),
    }


def pair_disruption(
    pair: tuple[dict[str, Any], dict[str, Any]],
    objective: str,
) -> float:
    """Measure intervention disruption under one explicit objective.

    Args:
        pair: Baseline and ablated continuation records.
        objective: Outcome objective to score.

    Returns:
        Nonnegative disruption, except signed correctness harm.
    """
    baseline, ablated = pair
    if objective == "answer":
        return float(baseline.get("produced_answer") != ablated.get("produced_answer"))
    if objective == "object":
        left = continuation_relations(baseline)
        right = continuation_relations(ablated)
        union = left | right
        return 1.0 - len(left & right) / max(len(union), 1)
    if objective == "correctness":
        return float(bool(baseline["is_correct"]) - bool(ablated["is_correct"]))
    if objective == "compression":
        return 1.0 - pair_common_prefix_fraction(pair)
    raise ValueError(f"Unsupported causal objective: {objective!r}")


def continuation_relations(row: dict[str, Any]) -> set[str]:
    """Extract the verified final relation set from a continuation.

    Args:
        row: Boundary-intervention continuation record.

    Returns:
        Canonical relation atoms present at the final symbolic update.
    """
    updates = extract_symbolic_updates(
        str(row.get("produced_text", "")),
        token_count=len(row.get("generated_token_ids", [])),
    )
    if not updates:
        return set()
    return {relation for relation in updates[-1].graph_signature.split("|") if relation}


def outcome_definition(objective: str) -> str:
    """Describe the intervention outcome associated with an objective.

    Args:
        objective: Outcome objective name.

    Returns:
        Human-readable metric definition.
    """
    return {
        "answer": "indicator that the extracted terminal answer changes",
        "object": "Jaccard distance between verified continuation relation sets",
        "correctness": "baseline correctness minus ablated correctness",
        "compression": "one minus baseline/ablated token-prefix agreement",
    }[objective]
