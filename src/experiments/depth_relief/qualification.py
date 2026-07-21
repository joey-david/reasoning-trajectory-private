"""Behavioral qualification for the causal depth-relief assay."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from .benchmark import (
    candidate_token_ids,
    decimal_state_symbols,
    format_model_prompt,
    format_prompt_spec,
    qualification_condition_specs,
    render_qualification_direct_prompt,
    render_qualification_prompt,
)
from .metrics import bootstrap_mean_ci, softmax


PromptScorer = Callable[[str, list[int]], dict[str, Any]]
DEFAULT_GATE = {
    "min_direct_accuracy_lower": 0.90,
    "min_register_accuracy_lower": 0.85,
    "min_history_accuracy_lower": 0.60,
    "min_invalid_invariance_lower": 0.60,
    "min_candidate_mass_lower": 0.80,
    "min_joint_none_gold": 50,
}


def score_logits(logits: np.ndarray, candidate_ids: list[int]) -> dict[str, Any]:
    """Reduce one final-token logit vector to the qualification record."""
    logits = np.asarray(logits, dtype=np.float64)
    full = softmax(logits)
    candidate = softmax(logits[candidate_ids])
    maximum = float(logits.max())
    log_normalizer = maximum + float(np.log(np.exp(logits - maximum).sum()))
    top_token = int(np.argmax(logits))
    return {
        "prediction": int(np.argmax(candidate)),
        "unconstrained_prediction": (
            candidate_ids.index(top_token) if top_token in candidate_ids else None
        ),
        "unconstrained_token_id": top_token,
        "candidate_probability_mass": float(full[candidate_ids].sum()),
        "final_candidate_probabilities": candidate.tolist(),
        "final_candidate_logprobabilities": (
            logits[candidate_ids] - log_normalizer
        ).tolist(),
    }


def _evaluate_prompt_conditions(
    *,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    candidate_symbols: list[str] | tuple[str, ...],
    score: PromptScorer,
) -> dict[str, dict[str, Any]]:
    """Score a rendered behavior protocol through the shared final-logit contract."""
    conditions: dict[str, dict[str, Any]] = {}
    for prompt in prompts:
        name = str(prompt["name"])
        text = str(prompt["text"])
        checkpoint_span = prompt.get("checkpoint_char_span")
        candidate_ids = candidate_token_ids(tokenizer, text, candidate_symbols)
        record = score(text, candidate_ids)
        expected = int(prompt["expected_next_state"])
        record.update(
            {
                "expected_next_state": expected,
                "is_expected": record["prediction"] == expected,
                "is_expected_unconstrained": record["unconstrained_prediction"]
                == expected,
                "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
                "checkpoint_char_span": (
                    list(checkpoint_span) if checkpoint_span is not None else None
                ),
            }
        )
        conditions[name] = record
    return conditions


def evaluate_prompt_conditions_hf(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    candidate_symbols: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Score any rendered behavior protocol with one HF final-logit path."""
    import torch

    device = model.get_input_embeddings().weight.device

    def score(text: str, candidate_ids: list[int]) -> dict[str, Any]:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
        logits = output.logits[0, -1].float().cpu().numpy()
        return score_logits(logits, candidate_ids)

    return _evaluate_prompt_conditions(
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_symbols=candidate_symbols,
        score=score,
    )


def evaluate_prompt_conditions_mlx(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    candidate_symbols: list[str] | tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Score any rendered behavior protocol with one MLX final-logit path."""
    import mlx.core as mx

    def score(text: str, candidate_ids: list[int]) -> dict[str, Any]:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        logits = model(mx.array([token_ids]))[0, -1].astype(mx.float32)
        mx.eval(logits)
        return score_logits(np.asarray(logits), candidate_ids)

    return _evaluate_prompt_conditions(
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_symbols=candidate_symbols,
        score=score,
    )


def render_qualification_case_prompts(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render and validate direct plus token-aligned register prompts."""
    rendered = []
    token_count: int | None = None
    for condition in qualification_condition_specs(case):
        name = str(condition["name"])
        if name == "direct":
            text = format_model_prompt(
                tokenizer,
                render_qualification_direct_prompt(case),
                config,
            )
            checkpoint_span: tuple[int, int] | None = None
        else:
            prompt = format_prompt_spec(
                tokenizer,
                render_qualification_prompt(case, condition),
                config,
            )
            text = prompt.text
            checkpoint_span = (prompt.checkpoint_start, prompt.checkpoint_end)
            current_count = len(tokenizer.encode(text, add_special_tokens=False))
            if token_count is None:
                token_count = current_count
            elif token_count != current_count:
                raise ValueError(
                    f"Condition {name!r} breaks qualification alignment: "
                    f"{current_count} != {token_count} tokens"
                )
        rendered.append(
            {
                "name": name,
                "text": text,
                "checkpoint_char_span": checkpoint_span,
                "expected_next_state": int(condition["expected_next_state"]),
            }
        )
    return rendered


def validate_qualification_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate prompt alignment and one-token candidates without model inference."""
    prompts = render_qualification_case_prompts(
        tokenizer=tokenizer,
        case=case,
        config=config,
    )
    candidate_symbols = decimal_state_symbols(2 ** int(case["bits"]))
    for prompt in prompts:
        candidate_token_ids(tokenizer, prompt["text"], candidate_symbols)
    return {
        "id": case["id"],
        "condition_count": len(prompts),
        "matched_token_count": len(
            tokenizer.encode(prompts[1]["text"], add_special_tokens=False)
        ),
        "checkpoint_char_span": list(prompts[1]["checkpoint_char_span"]),
    }


def _qualification_record(
    case: dict[str, Any], conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Attach stable qualification metadata to shared condition scores."""
    return {
        "schema_version": 1,
        "id": case["id"],
        "family": case["family"],
        "bits": int(case["bits"]),
        "history_steps": int(case["history_steps"]),
        "next_state": int(case["next_state"]),
        "counterfactual_next_state": int(case["counterfactual_next_state"]),
        "conditions": conditions,
    }


def evaluate_qualification_case_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one qualification case with a lightweight final-logit HF forward pass."""
    prompts = render_qualification_case_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    conditions = evaluate_prompt_conditions_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_symbols=decimal_state_symbols(2 ** int(case["bits"])),
    )
    return _qualification_record(case, conditions)


def evaluate_qualification_case_mlx(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run one qualification case with the cached local MLX model."""
    prompts = render_qualification_case_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    conditions = evaluate_prompt_conditions_mlx(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_symbols=decimal_state_symbols(2 ** int(case["bits"])),
    )
    return _qualification_record(case, conditions)


def _expected(row: dict[str, Any], condition: str) -> bool:
    return bool(row["conditions"][condition]["is_expected_unconstrained"])


def summarize_qualification_rows(
    rows: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    """Summarize behavior and decide whether depth measurement is admissible."""
    if not rows:
        raise ValueError("Cannot summarize an empty qualification result set")
    gate = {**DEFAULT_GATE, **gate}
    names = ("direct", "none", "gold", "counterfactual", "invalid")
    condition_stats = {}
    for index, name in enumerate(names):
        condition_stats[name] = {
            "accuracy": bootstrap_mean_ci(
                [_expected(row, name) for row in rows], seed=100 + index
            ),
            "candidate_probability_mass": bootstrap_mean_ci(
                [
                    float(row["conditions"][name]["candidate_probability_mass"])
                    for row in rows
                ],
                seed=110 + index,
            ),
        }

    none_gold_pairs = [
        (_expected(row, "none"), _expected(row, "gold")) for row in rows
    ]
    joint_none_gold = sum(none and gold for none, gold in none_gold_pairs)
    none_gold_outcomes = {
        "both_correct": joint_none_gold,
        "none_only": sum(none and not gold for none, gold in none_gold_pairs),
        "gold_only": sum(not none and gold for none, gold in none_gold_pairs),
        "both_wrong": sum(not none and not gold for none, gold in none_gold_pairs),
    }
    gold_accuracy_uplift = bootstrap_mean_ci(
        [int(gold) - int(none) for none, gold in none_gold_pairs],
        seed=301,
    )
    invalid_agreement = [
        row["conditions"]["none"]["unconstrained_prediction"]
        == row["conditions"]["invalid"]["unconstrained_prediction"]
        for row in rows
    ]
    invalid_joint_correct_agreement = [
        agreement and _expected(row, "none") and _expected(row, "invalid")
        for row, agreement in zip(rows, invalid_agreement)
    ]
    flag_pair_both_correct = [
        _expected(row, "none") and _expected(row, "counterfactual") for row in rows
    ]
    flag_ignored_on_none = [
        _expected(row, "counterfactual")
        and row["conditions"]["none"]["unconstrained_prediction"]
        == int(row["counterfactual_next_state"])
        for row in rows
    ]
    invalid_probability_shift = []
    for row in rows:
        target = int(row["next_state"])
        invalid_probability_shift.append(
            abs(
                float(
                    row["conditions"]["invalid"]["final_candidate_probabilities"][
                        target
                    ]
                )
                - float(
                    row["conditions"]["none"]["final_candidate_probabilities"][
                        target
                    ]
                )
            )
        )

    histories: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        histories[int(row["history_steps"])].append(row)
    by_history = {
        str(history): {
            name: bootstrap_mean_ci(
                [_expected(row, name) for row in values],
                seed=200 + history * 10 + index,
            )
            for index, name in enumerate(names)
        }
        for history, values in sorted(histories.items())
    }

    def lower(stat: dict[str, Any]) -> float:
        value = stat["ci95"][0]
        return float(value) if value is not None else float("-inf")

    all_mass = bootstrap_mean_ci(
        [
            float(row["conditions"][name]["candidate_probability_mass"])
            for row in rows
            for name in names
        ],
        seed=300,
    )
    invalid_invariance = bootstrap_mean_ci(invalid_agreement, seed=302)
    checks = {
        "direct_accuracy": lower(condition_stats["direct"]["accuracy"])
        >= float(gate["min_direct_accuracy_lower"]),
        "gold_accuracy": lower(condition_stats["gold"]["accuracy"])
        >= float(gate["min_register_accuracy_lower"]),
        "counterfactual_accuracy": lower(
            condition_stats["counterfactual"]["accuracy"]
        )
        >= float(gate["min_register_accuracy_lower"]),
        "none_accuracy": lower(condition_stats["none"]["accuracy"])
        >= float(gate["min_history_accuracy_lower"]),
        "invalid_accuracy": lower(condition_stats["invalid"]["accuracy"])
        >= float(gate["min_history_accuracy_lower"]),
        "invalid_invariance": lower(invalid_invariance)
        >= float(gate["min_invalid_invariance_lower"]),
        "candidate_probability_mass": lower(all_mass)
        >= float(gate["min_candidate_mass_lower"]),
        "joint_none_gold": joint_none_gold >= int(gate["min_joint_none_gold"]),
    }
    return {
        "schema_version": 2,
        "case_count": len(rows),
        "bits": sorted({int(row["bits"]) for row in rows}),
        "history_steps": sorted({int(row["history_steps"]) for row in rows}),
        "conditions": condition_stats,
        "by_history": by_history,
        "joint_none_gold_unconstrained": {
            "n": joint_none_gold,
            "rate": joint_none_gold / len(rows),
            "paired_outcomes": none_gold_outcomes,
            "gold_minus_none_accuracy": gold_accuracy_uplift,
        },
        "invalid_register": {
            "prediction_invariance": invalid_invariance,
            "joint_correct_prediction_invariance": bootstrap_mean_ci(
                invalid_joint_correct_agreement,
                seed=303,
            ),
            "mean_absolute_correct_probability_shift": float(
                np.mean(invalid_probability_shift)
            ),
        },
        "valid_flag_control": {
            "none_and_counterfactual_both_correct": bootstrap_mean_ci(
                flag_pair_both_correct,
                seed=304,
            ),
            "flag_ignored_on_none": bootstrap_mean_ci(
                flag_ignored_on_none,
                seed=305,
            ),
        },
        "all_condition_candidate_probability_mass": all_mass,
        "gate": {
            "thresholds": gate,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
