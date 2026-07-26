"""Proof-specific diagnostics over saved recursive interface steps."""

from __future__ import annotations

from typing import Any

from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci


def proof_step_diagnostics(
    ids: list[str],
    *,
    program_index: dict[str, dict[str, Any]],
    interface_index: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    semantic_values = [
        bool(step["locally_semantic_correct"])
        for case_id in ids
        for step in interface_index[case_id]["steps"]
        if step.get("locally_semantic_correct") is not None
    ]
    exact_steps = [
        (case_id, step)
        for case_id in ids
        for step in interface_index[case_id]["steps"]
        if len(step.get("predicted_semantic_states", ())) == 1
    ]
    false_positive = sum(
        (
            int(step["predicted_semantic_states"][0])
            & ~int(step["true_output_state"])
        ).bit_count()
        for _, step in exact_steps
    )
    false_negative = sum(
        (
            int(step["true_output_state"])
            & ~int(step["predicted_semantic_states"][0])
        ).bit_count()
        for _, step in exact_steps
    )
    fact_decisions = sum(
        int(program_index[case_id]["bits"]) for case_id, _ in exact_steps
    )
    all_facts_predictions = sum(
        int(step["predicted_semantic_states"][0])
        == (1 << int(program_index[case_id]["bits"])) - 1
        for case_id, step in exact_steps
    )
    all_facts_targets = sum(
        int(step["true_output_state"])
        == (1 << int(program_index[case_id]["bits"])) - 1
        for case_id, step in exact_steps
    )
    result: dict[str, Any] = {
        "local_quotient_semantic_closure": bootstrap_mean_ci(
            semantic_values, seed=seed
        ),
        "false_positive_fact_rate": (
            false_positive / fact_decisions if fact_decisions else None
        ),
        "false_negative_fact_rate": (
            false_negative / fact_decisions if fact_decisions else None
        ),
        "exact_state_step_count": len(exact_steps),
        "all_facts_prediction_rate": (
            all_facts_predictions / len(exact_steps) if exact_steps else None
        ),
        "all_facts_target_rate": (
            all_facts_targets / len(exact_steps) if exact_steps else None
        ),
    }
    if all("gold_final" in interface_index[case_id] for case_id in ids):
        clusters = [
            str(program_index[case_id]["program_context"]) for case_id in ids
        ]
        result["gold_code_continuation_accuracy"] = cluster_bootstrap_mean_ci(
            [
                bool(
                    interface_index[case_id]["gold_final"][
                        "is_expected_unconstrained"
                    ]
                )
                for case_id in ids
            ],
            clusters,
            seed=seed + 1,
        )
    return result


def proof_transition_class_summary(
    interface_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_classes = sorted(
        {
            str(step["proof_transition_class"])
            for row in interface_rows
            for step in row["steps"]
            if step.get("proof_transition_class") is not None
        }
    )
    result = {}
    for index, transition_class in enumerate(transition_classes):
        selected = [
            step
            for row in interface_rows
            for step in row["steps"]
            if step.get("proof_transition_class") == transition_class
        ]
        result[transition_class] = {
            "step_count": len(selected),
            "semantic_accuracy": bootstrap_mean_ci(
                [bool(step["locally_semantic_correct"]) for step in selected],
                seed=83_600 + index,
            ),
        }
    return result
