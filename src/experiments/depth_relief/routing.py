"""Matched explicit-state confirmation for serial state routing."""

from __future__ import annotations

from typing import Any

from .benchmark import candidate_token_ids, format_model_prompt, state_symbols, state_text
from .factorization import (
    render_factorization_history,
    render_factorization_preamble,
    render_factorization_rule,
)
from .metrics import bootstrap_mean_ci
from .qualification import (
    evaluate_prompt_conditions_hf,
    evaluate_prompt_conditions_mlx,
)


CONDITIONS = ("materialized", "counterfactual")
DEFAULT_GATE = {
    "min_accuracy_lower": 0.85,
    "min_candidate_mass_lower": 0.80,
    "min_joint_correct": 50,
}


def select_routing_cases(
    cases: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select cases where Read, Update, and Synthesize already succeeded."""
    indexed = {str(case["id"]): case for case in cases}
    selected = []
    for row in rows:
        conditions = row["conditions"]
        if all(
            conditions[name]["is_expected_unconstrained"]
            for name in ("read", "update", "synthesize")
        ):
            selected.append(indexed[str(row["id"])])
    return selected


def render_routing_prompts(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Render gold and counterfactual states in an otherwise identical full context."""
    final = render_factorization_rule(case, case["final_rule"])
    prefix = (
        render_factorization_preamble(case)
        + f"Start state: {state_text(case, int(case['initial_state']))}.\n"
        + render_factorization_history(case)
        + "\nMaterialized current state: "
    )
    suffix = (
        ".\nThe materialized state is authoritative. Use it as the current state; "
        "do not repeat the numbered steps.\n"
        f"FINAL: {final}.\nApply FINAL exactly once and return the resulting state.\n"
        "Answer="
    )
    specifications = (
        ("materialized", int(case["current_state"]), int(case["next_state"])),
        (
            "counterfactual",
            int(case["counterfactual_state"]),
            int(case["counterfactual_next_state"]),
        ),
    )
    return [
        {
            "name": name,
            "text": format_model_prompt(
                tokenizer, prefix + state_text(case, state) + suffix, config
            ),
            "expected_next_state": expected,
        }
        for name, state, expected in specifications
    ]


def validate_routing_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_routing_prompts(tokenizer=tokenizer, case=case, config=config)
    token_rows = [
        tokenizer.encode(prompt["text"], add_special_tokens=False) for prompt in prompts
    ]
    differences = [
        index
        for index, (left, right) in enumerate(zip(token_rows[0], token_rows[1]))
        if left != right
    ]
    if len(token_rows[0]) != len(token_rows[1]) or len(differences) != 1:
        raise ValueError("Routing prompts must differ at exactly one state token")
    for prompt in prompts:
        candidate_token_ids(tokenizer, prompt["text"], state_symbols(case))
    return {
        "id": case["id"],
        "condition_count": len(prompts),
        "token_count": len(token_rows[0]),
        "state_token_index": differences[0],
    }


def _record(
    case: dict[str, Any], conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": case["id"],
        "history_family": case["history_family"],
        "format": case["format"],
        "bits": int(case["bits"]),
        "state_representation": str(case.get("state_representation", "decimal")),
        "state_symbols": list(state_symbols(case)),
        "history_steps": int(case["history_steps"]),
        "current_state": int(case["current_state"]),
        "counterfactual_state": int(case["counterfactual_state"]),
        "next_state": int(case["next_state"]),
        "counterfactual_next_state": int(case["counterfactual_next_state"]),
        "conditions": conditions,
    }


def evaluate_routing_case_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_routing_prompts(tokenizer=tokenizer, case=case, config=config)
    return _record(
        case,
        evaluate_prompt_conditions_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_symbols=state_symbols(case),
        ),
    )


def evaluate_routing_case_mlx(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_routing_prompts(tokenizer=tokenizer, case=case, config=config)
    return _record(
        case,
        evaluate_prompt_conditions_mlx(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_symbols=state_symbols(case),
        ),
    )


def summarize_routing_rows(
    rows: list[dict[str, Any]], gate_config: dict[str, Any]
) -> dict[str, Any]:
    """Require both factual rescue and rule-consistent counterfactual steering."""
    if not rows:
        raise ValueError("Cannot summarize empty routing-confirmation results")
    gate = {**DEFAULT_GATE, **gate_config}

    def correct(row: dict[str, Any], name: str) -> bool:
        return bool(row["conditions"][name]["is_expected_unconstrained"])

    accuracy = {
        name: bootstrap_mean_ci(
            [correct(row, name) for row in rows], seed=800 + index
        )
        for index, name in enumerate(CONDITIONS)
    }
    candidate_mass = bootstrap_mean_ci(
        [
            float(row["conditions"][name]["candidate_probability_mass"])
            for row in rows
            for name in CONDITIONS
        ],
        seed=810,
    )
    joint = sum(
        correct(row, "materialized") and correct(row, "counterfactual")
        for row in rows
    )

    def lower(stat: dict[str, Any]) -> float:
        value = stat["ci95"][0]
        return float(value) if value is not None else float("-inf")

    checks = {
        "materialized_accuracy": lower(accuracy["materialized"])
        >= float(gate["min_accuracy_lower"]),
        "counterfactual_accuracy": lower(accuracy["counterfactual"])
        >= float(gate["min_accuracy_lower"]),
        "candidate_mass": lower(candidate_mass)
        >= float(gate["min_candidate_mass_lower"]),
        "joint_correct": joint >= int(gate["min_joint_correct"]),
    }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "accuracy": accuracy,
        "candidate_probability_mass": candidate_mass,
        "joint_correct": {"n": joint, "rate": joint / len(rows)},
        "by_format": {
            value: {
                name: bootstrap_mean_ci(
                    [correct(row, name) for row in rows if row["format"] == value],
                    seed=820 + format_index * 10 + condition_index,
                )
                for condition_index, name in enumerate(CONDITIONS)
            }
            for format_index, value in enumerate(
                sorted({str(row["format"]) for row in rows})
            )
        },
        "gate": {"thresholds": gate, "checks": checks, "passed": all(checks.values())},
    }
