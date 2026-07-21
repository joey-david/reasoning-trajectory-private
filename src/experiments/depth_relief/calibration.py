"""Sentinel-based task-frontier calibration for causal depth relief."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .benchmark import (
    PromptSpec,
    apply_rule,
    build_transition_case,
    candidate_token_ids,
    decimal_state_symbols,
    format_model_prompt,
    format_prompt_spec,
    rule_text,
)
from .metrics import bootstrap_mean_ci
from .qualification import (
    evaluate_prompt_conditions_hf,
    evaluate_prompt_conditions_mlx,
)


CONDITIONS = ("direct", "none", "none_alt", "gold", "counterfactual")
DEFAULT_DISCOVERY_GATE = {
    "min_direct_accuracy": 0.90,
    "min_register_accuracy": 0.90,
    "min_absent_accuracy": 0.60,
    "min_absent_invariance": 0.80,
    "min_candidate_mass": 0.80,
    "min_joint_none_gold": 8,
    "target_none_accuracy": 0.75,
}


def build_calibration_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the discovery grid over simple histories and a fixed final lookup."""
    history_families = tuple(config.get("history_families", ("add", "xor", "affine")))
    final_family = str(config.get("final_family", "pointer"))
    widths = tuple(int(value) for value in config.get("bits", (2, 3)))
    histories = tuple(int(value) for value in config.get("history_steps", (1, 2, 3, 4)))
    examples = int(config.get("examples_per_cell", 12))
    seed = int(config.get("seed", 0))
    cases = []
    for history_family in history_families:
        for history_steps in histories:
            for width in widths:
                for example in range(examples):
                    case = build_transition_case(
                        history_family=str(history_family),
                        final_family=final_family,
                        width=width,
                        example_index=example,
                        seed=seed,
                        history_steps=history_steps,
                    )
                    case["id"] = (
                        f"{history_family}_to_{final_family}_h{history_steps}"
                        f"_b{width}_{example:04d}"
                    )
                    cases.append(case)
    return cases


def calibration_condition_specs(
    case: dict[str, Any], sentinels: tuple[str, str]
) -> list[dict[str, Any]]:
    """Define two absent sentinels and the matched numeric checkpoint controls."""
    return [
        {"name": "direct", "checkpoint": None, "expected": case["next_state"]},
        {"name": "none", "checkpoint": sentinels[0], "expected": case["next_state"]},
        {
            "name": "none_alt",
            "checkpoint": sentinels[1],
            "expected": case["next_state"],
        },
        {
            "name": "gold",
            "checkpoint": str(case["current_state"]),
            "expected": case["next_state"],
        },
        {
            "name": "counterfactual",
            "checkpoint": str(case["counterfactual_state"]),
            "expected": case["counterfactual_next_state"],
        },
    ]


def _history_text(case: dict[str, Any]) -> str:
    return "\n".join(
        f"Step {index}: {rule_text(rule, int(case['bits']))}."
        for index, rule in enumerate(case["history"], 1)
    )


def render_calibration_prompt(
    case: dict[str, Any], condition: dict[str, Any]
) -> PromptSpec:
    """Render one bracketed checkpoint whose payload occupies exactly one token."""
    if condition["name"] == "direct":
        raise ValueError("The direct control has no checkpoint payload")
    width = int(case["bits"])
    prefix = (
        "Follow the state-transition instructions exactly.\n"
        f"Start state: {int(case['initial_state'])}.\n"
        f"{_history_text(case)}\n"
        "Checkpoint state=["
    )
    checkpoint = str(condition["checkpoint"])
    suffix = (
        "].\nA decimal checkpoint is authoritative. A word means the checkpoint is "
        "absent, so derive the state from the numbered steps.\n"
        f"FINAL: {rule_text(case['final_rule'], width)}. Apply FINAL exactly once.\n"
        "Return only the resulting decimal state.\nAnswer="
    )
    return PromptSpec(
        text=prefix + checkpoint + suffix,
        checkpoint_start=len(prefix),
        checkpoint_end=len(prefix) + len(checkpoint),
    )


def render_calibration_direct_prompt(case: dict[str, Any]) -> str:
    """Render the shared one-step final-transition positive control."""
    return (
        f"Apply this operation to state {int(case['current_state'])}: "
        f"{rule_text(case['final_rule'], int(case['bits']))}. "
        "Return only the resulting decimal state.\nAnswer="
    )


def _sentinels(config: dict[str, Any]) -> tuple[str, str]:
    values = tuple(str(value) for value in config.get("sentinels", ("unknown", "missing")))
    if len(values) != 2 or len(set(values)) != 2:
        raise ValueError("Calibration needs exactly two distinct absent sentinels")
    if any(not value.isalpha() for value in values):
        raise ValueError("Absent sentinels must be alphabetic words")
    return values


def render_calibration_case_prompts(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Render prompts and prove that matched conditions differ at one token only."""
    rendered = []
    matched_ids: list[list[int]] = []
    checkpoint_token_index: int | None = None
    for condition in calibration_condition_specs(case, _sentinels(config)):
        name = str(condition["name"])
        if name == "direct":
            text = format_model_prompt(
                tokenizer, render_calibration_direct_prompt(case), config
            )
            checkpoint_span = None
        else:
            prompt = format_prompt_spec(
                tokenizer, render_calibration_prompt(case, condition), config
            )
            text = prompt.text
            checkpoint_span = (prompt.checkpoint_start, prompt.checkpoint_end)
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if matched_ids:
                differences = [
                    index
                    for index, (left, right) in enumerate(zip(matched_ids[0], token_ids))
                    if left != right
                ]
                if len(token_ids) != len(matched_ids[0]) or len(differences) != 1:
                    raise ValueError(
                        f"Condition {name!r} is not a one-token checkpoint substitution"
                    )
                if checkpoint_token_index is None:
                    checkpoint_token_index = differences[0]
                elif checkpoint_token_index != differences[0]:
                    raise ValueError("Checkpoint token moved across matched conditions")
            matched_ids.append(token_ids)
        rendered.append(
            {
                "name": name,
                "text": text,
                "checkpoint_char_span": checkpoint_span,
                "expected_next_state": int(condition["expected"]),
            }
        )
    if checkpoint_token_index is None:
        raise ValueError("Calibration conditions did not expose one checkpoint token")
    for prompt in rendered[1:]:
        prompt["checkpoint_token_index"] = checkpoint_token_index
    return rendered


def validate_calibration_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate alignment and one-token answer candidates without model inference."""
    prompts = render_calibration_case_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    for prompt in prompts:
        candidate_token_ids(
            tokenizer,
            prompt["text"],
            decimal_state_symbols(2 ** int(case["bits"])),
        )
    matched = prompts[1]
    return {
        "id": case["id"],
        "condition_count": len(prompts),
        "matched_token_count": len(
            tokenizer.encode(matched["text"], add_special_tokens=False)
        ),
        "checkpoint_token_index": int(matched["checkpoint_token_index"]),
    }


def _calibration_record(
    case: dict[str, Any], conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": case["id"],
        "history_family": case["history_family"],
        "final_family": case["final_family"],
        "bits": int(case["bits"]),
        "history_steps": int(case["history_steps"]),
        "next_state": int(case["next_state"]),
        "counterfactual_next_state": int(case["counterfactual_next_state"]),
        "conditions": conditions,
    }


def evaluate_calibration_case_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one frontier case through the shared HF scorer."""
    prompts = render_calibration_case_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    return _calibration_record(
        case,
        evaluate_prompt_conditions_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_symbols=decimal_state_symbols(2 ** int(case["bits"])),
        ),
    )


def evaluate_calibration_case_mlx(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one frontier case through the shared MLX scorer."""
    prompts = render_calibration_case_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    return _calibration_record(
        case,
        evaluate_prompt_conditions_mlx(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            candidate_symbols=decimal_state_symbols(2 ** int(case["bits"])),
        ),
    )


def _correct(row: dict[str, Any], name: str) -> bool:
    return bool(row["conditions"][name]["is_expected_unconstrained"])


def _cell_summary(
    rows: list[dict[str, Any]], gate: dict[str, Any], seed: int
) -> dict[str, Any]:
    accuracy = {
        name: bootstrap_mean_ci([_correct(row, name) for row in rows], seed=seed + i)
        for i, name in enumerate(CONDITIONS)
    }
    absence_agreement = bootstrap_mean_ci(
        [
            row["conditions"]["none"]["unconstrained_prediction"]
            == row["conditions"]["none_alt"]["unconstrained_prediction"]
            for row in rows
        ],
        seed=seed + 10,
    )
    candidate_mass = bootstrap_mean_ci(
        [
            float(row["conditions"][name]["candidate_probability_mass"])
            for row in rows
            for name in CONDITIONS
        ],
        seed=seed + 11,
    )
    joint = sum(_correct(row, "none") and _correct(row, "gold") for row in rows)
    checks = {
        "direct": accuracy["direct"]["mean"] >= gate["min_direct_accuracy"],
        "register": min(
            accuracy["gold"]["mean"], accuracy["counterfactual"]["mean"]
        )
        >= gate["min_register_accuracy"],
        "absent": min(accuracy["none"]["mean"], accuracy["none_alt"]["mean"])
        >= gate["min_absent_accuracy"],
        "absent_invariance": absence_agreement["mean"]
        >= gate["min_absent_invariance"],
        "candidate_mass": candidate_mass["mean"] >= gate["min_candidate_mass"],
        "joint_none_gold": joint >= gate["min_joint_none_gold"],
    }
    return {
        "case_count": len(rows),
        "accuracy": accuracy,
        "absence_prediction_invariance": absence_agreement,
        "candidate_probability_mass": candidate_mass,
        "joint_none_gold": joint,
        "gold_minus_none_accuracy": bootstrap_mean_ci(
            [int(_correct(row, "gold")) - int(_correct(row, "none")) for row in rows],
            seed=seed + 12,
        ),
        "checks": checks,
        "eligible": all(checks.values()),
    }


def summarize_calibration_rows(
    rows: list[dict[str, Any]], gate_config: dict[str, Any]
) -> dict[str, Any]:
    """Find a behaviorally valid task frontier without authorizing depth capture."""
    if not rows:
        raise ValueError("Cannot summarize an empty frontier calibration")
    gate = {**DEFAULT_DISCOVERY_GATE, **gate_config}
    grouped: defaultdict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["history_family"]),
                str(row["final_family"]),
                int(row["bits"]),
                int(row["history_steps"]),
            )
        ].append(row)
    cells = []
    for index, (key, values) in enumerate(sorted(grouped.items())):
        summary = _cell_summary(values, gate, seed=1000 + 20 * index)
        summary.update(
            {
                "history_family": key[0],
                "final_family": key[1],
                "bits": key[2],
                "history_steps": key[3],
            }
        )
        cells.append(summary)
    eligible = [cell for cell in cells if cell["eligible"]]
    target = float(gate["target_none_accuracy"])
    eligible.sort(
        key=lambda cell: (
            abs(float(cell["accuracy"]["none"]["mean"]) - target),
            -int(cell["joint_none_gold"]),
            -int(cell["history_steps"]),
        )
    )
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "cell_count": len(cells),
        "gate": gate,
        "eligible_cell_count": len(eligible),
        "selected_cell": eligible[0] if eligible else None,
        "cells": cells,
        "depth_capture_authorized": False,
        "next_stage": "held_out_confirmation" if eligible else "stop_or_redesign",
    }


def summarize_history_execution(
    rows: list[dict[str, Any]], cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Diagnose whether absent-checkpoint errors simply skip the history."""
    indexed = {str(case["id"]): case for case in cases}
    by_history: defaultdict[int, list[str]] = defaultdict(list)
    for row in rows:
        case = indexed[str(row["id"])]
        modulus = 2 ** int(row["bits"])
        correct = int(case["next_state"])
        skipped = apply_rule(
            case["final_rule"], int(case["initial_state"]), modulus
        )
        if skipped == correct:
            outcome = "ambiguous"
        else:
            prediction = row["conditions"]["none"]["unconstrained_prediction"]
            if prediction == correct:
                outcome = "correct"
            elif prediction == skipped:
                outcome = "skips_all_history"
            else:
                outcome = "other"
        by_history[int(row["history_steps"])].append(outcome)

    def summarize(outcomes: list[str]) -> dict[str, Any]:
        counts = {name: outcomes.count(name) for name in (
            "correct",
            "skips_all_history",
            "other",
            "ambiguous",
        )}
        distinct = len(outcomes) - counts["ambiguous"]
        return {
            "case_count": len(outcomes),
            "distinct_target_count": distinct,
            "counts": counts,
            "rates_among_distinct_targets": {
                name: counts[name] / distinct if distinct else None
                for name in ("correct", "skips_all_history", "other")
            },
        }

    return {
        "all": summarize([outcome for values in by_history.values() for outcome in values]),
        "by_history": {
            str(history): summarize(values)
            for history, values in sorted(by_history.items())
        },
    }
