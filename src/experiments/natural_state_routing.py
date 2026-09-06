"""Transfer frozen state-routing heads to natural arithmetic traces."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from reasoning_trajectory.token_alignment import (
    token_range_for_chars,
    token_spans_for_row,
)
from src.analysis.answers import answers_match, extract_answer
from src.experiments.state_routing_repair import forward_logits_ids, steer_logits_ids
from src.experiments.symbolic import extract_symbolic_updates, parse_number


_NUMBER_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _equations(text: str, token_count: int) -> list[dict[str, Any]]:
    equations = []
    for update in extract_symbolic_updates(text, token_count=token_count):
        if update.operator != "OPERATE":
            continue
        raw = text[update.char_start : update.char_end]
        equals = raw.rfind("=")
        if equals < 0:
            continue
        results = list(_NUMBER_RE.finditer(raw[equals + 1 :]))
        if len(results) != 1:
            continue
        result = results[0]
        result_start = update.char_start + equals + 1 + result.start()
        operands = [
            {
                "span": [update.char_start + match.start(), update.char_start + match.end()],
                "text": match.group(),
                "value": parse_number(match.group()),
            }
            for match in _NUMBER_RE.finditer(raw[:equals])
        ]
        equations.append(
            {
                "span": [update.char_start, update.char_end],
                "text": raw,
                "operands": operands,
                "result_span": [result_start, result_start + len(result.group())],
                "result_text": result.group(),
                "value": update.value,
            }
        )
    return equations


def build_natural_cases(
    generations: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    *,
    produced_answer_regex: str,
    gold_answer_regex: str,
) -> list[dict[str, Any]]:
    """Mine one unambiguous computed-result reuse site per generated trace."""
    samples = {str(row["id"]): row for row in dataset}
    cases = []
    for generation in generations:
        sample_id = str(generation["sample_id"])
        sample = samples[sample_id]
        text = str(generation["produced_text"])
        question_values = {
            parse_number(match.group())
            for match in _NUMBER_RE.finditer(str(sample["question"]))
        }
        equations = _equations(text, len(generation["generated_token_ids"]))
        candidates = []
        for target_index, target in enumerate(equations):
            if target_index < 2:
                continue
            target_values = {operand["value"] for operand in target["operands"]}
            for operand in target["operands"]:
                if operand["value"] in question_values:
                    continue
                sources = [
                    equation
                    for equation in equations[:target_index]
                    if equation["value"] == operand["value"]
                ]
                controls = [
                    equation
                    for equation in equations[:target_index]
                    if equation["value"] not in target_values
                    and len(_plain_number(equation["result_text"]))
                    == len(_plain_number(operand["text"]))
                ]
                if len(sources) != 1 or not controls:
                    continue
                source = sources[0]
                if _plain_number(source["result_text"]) != _plain_number(operand["text"]):
                    continue
                control = controls[-1]
                candidates.append(
                    {
                        "source": source,
                        "source_index": equations.index(source),
                        "control": control,
                        "target": target,
                        "operand": operand,
                        "target_index": target_index,
                    }
                )
        if not candidates:
            continue
        chosen = max(
            candidates,
            key=lambda row: (
                row["target_index"] - row["source_index"],
                row["operand"]["span"][0] - row["source"]["result_span"][1],
                row["target_index"],
            ),
        )
        prediction = extract_answer(text, produced_answer_regex)
        gold = extract_answer(str(sample["gold_answer"]), gold_answer_regex)
        cases.append(
            {
                "schema_version": 1,
                "id": f"{sample_id}:{chosen['operand']['span'][0]}",
                "sample_id": sample_id,
                "final_correct": bool(answers_match(prediction, gold)),
                "equation_count": len(equations),
                "source_equation_index": chosen["source_index"],
                "target_equation_index": chosen["target_index"],
                "source_span": chosen["source"]["result_span"],
                "source_value": chosen["source"]["value"],
                "source_text": chosen["source"]["result_text"],
                "wrong_source_span": chosen["control"]["result_span"],
                "wrong_source_value": chosen["control"]["value"],
                "wrong_source_text": chosen["control"]["result_text"],
                "target_span": chosen["operand"]["span"],
                "target_value": chosen["operand"]["value"],
                "target_text": chosen["operand"]["text"],
                "target_equation": chosen["target"]["text"],
            }
        )
    return cases


def _plain_number(value: str) -> str:
    return value.replace(",", "")


def natural_token_contract(
    tokenizer: Any,
    sample: dict[str, Any],
    generation: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    """Map numeric spans onto exact saved model tokens."""
    generated_ids = [int(value) for value in generation["generated_token_ids"]]
    input_ids = [int(value) for value in sample["input_ids"]]
    if len(input_ids) != int(sample["dp1_idx"]):
        raise ValueError("saved input length does not match dp1_idx")
    spans = token_spans_for_row(tokenizer, generation)

    def positions(key: str) -> list[int]:
        span = list(map(int, case[key]))
        token_range = token_range_for_chars(spans, span[0], span[1])
        if token_range is None:
            raise ValueError(f"{key} does not align to saved generated tokens")
        return list(range(token_range[0], token_range[1] + 1))

    source = positions("source_span")
    wrong = positions("wrong_source_span")
    target = positions("target_span")
    if not (len(source) == len(wrong) == len(target)):
        raise ValueError(
            f"unequal token spans: source={len(source)} wrong={len(wrong)} "
            f"target={len(target)}"
        )
    source_ids = [generated_ids[index] for index in source]
    wrong_ids = [generated_ids[index] for index in wrong]
    target_ids = [generated_ids[index] for index in target]
    if source_ids != target_ids:
        raise ValueError("source and repeated target use different token IDs")
    contrastive = [
        index
        for index, (target_id, wrong_id) in enumerate(zip(target_ids, wrong_ids))
        if target_id != wrong_id
    ]
    if not contrastive:
        raise ValueError("target and wrong source have no contrasting token")
    return {
        "input_ids": input_ids,
        "generated_ids": generated_ids,
        "source_positions": source,
        "wrong_source_positions": wrong,
        "target_positions": target,
        "target_ids": target_ids,
        "wrong_ids": wrong_ids,
        "contrastive_indices": contrastive,
    }


def _condition_score(logits: torch.Tensor, target_id: int, wrong_id: int) -> dict[str, Any]:
    return {
        "prediction_id": int(logits.argmax().item()),
        "target_id": target_id,
        "wrong_id": wrong_id,
        "target_minus_wrong": float((logits[target_id] - logits[wrong_id]).item()),
    }


def measure_natural_case(
    *,
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    generation: dict[str, Any],
    case: dict[str, Any],
    heads: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Steer frozen heads between prior computed multi-token values."""
    contract = natural_token_contract(tokenizer, sample, generation, case)
    selected = heads["selected"]
    input_ids = list(contract["input_ids"])
    generated_ids = list(contract["generated_ids"])
    target_positions = list(contract["target_positions"])
    token_rows = []
    for index in contract["contrastive_indices"]:
        target_position = int(target_positions[index])
        prefix = input_ids + generated_ids[:target_position]
        source_position = len(input_ids) + int(contract["source_positions"][index])
        wrong_position = len(input_ids) + int(
            contract["wrong_source_positions"][index]
        )
        logits = {
            "baseline": forward_logits_ids(model, prefix),
            "valid_source": steer_logits_ids(
                model, prefix, source_position, selected, alpha
            ),
            "wrong_source": steer_logits_ids(
                model, prefix, wrong_position, selected, alpha
            ),
            "matched_heads": steer_logits_ids(
                model,
                prefix,
                wrong_position,
                heads["layer_matched_controls"],
                alpha,
            ),
        }
        target_id = int(contract["target_ids"][index])
        wrong_id = int(contract["wrong_ids"][index])
        token_rows.append(
            {
                "span_index": index,
                "conditions": {
                    name: _condition_score(value, target_id, wrong_id)
                    for name, value in logits.items()
                },
            }
        )
    return {
        **case,
        "token_contract": {
            key: value
            for key, value in contract.items()
            if key not in {"input_ids", "generated_ids"}
        },
        "alpha": alpha,
        "tokens": token_rows,
    }


def summarize_natural_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source-specific causal effects after baseline alignment."""
    def stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
        aligned = [
            row
            for row in subset
            if all(
                token["conditions"]["baseline"]["prediction_id"]
                == token["conditions"]["baseline"]["target_id"]
                for token in row["tokens"]
            )
        ]

        def exact(condition: str, key: str) -> float | None:
            if not aligned:
                return None
            return float(
                np.mean(
                    [
                        all(
                            token["conditions"][condition]["prediction_id"]
                            == token["conditions"][condition][key]
                            for token in row["tokens"]
                        )
                        for row in aligned
                    ]
                )
            )

        def shift(condition: str) -> float | None:
            if not aligned:
                return None
            return float(
                np.mean(
                    [
                        np.mean(
                            [
                                token["conditions"]["baseline"][
                                    "target_minus_wrong"
                                ]
                                - token["conditions"][condition][
                                    "target_minus_wrong"
                                ]
                                for token in row["tokens"]
                            ]
                        )
                        for row in aligned
                    ]
                )
            )

        def token_exact(condition: str, key: str) -> float | None:
            tokens = [token for row in aligned for token in row["tokens"]]
            if not tokens:
                return None
            return float(
                np.mean(
                    [
                        token["conditions"][condition]["prediction_id"]
                        == token["conditions"][condition][key]
                        for token in tokens
                    ]
                )
            )

        return {
            "case_count": len(subset),
            "baseline_aligned": len(aligned),
            "final_correct_aligned": sum(row["final_correct"] for row in aligned),
            "valid_source_preservation": exact("valid_source", "target_id"),
            "wrong_source_exact": exact("wrong_source", "wrong_id"),
            "matched_head_wrong_source_exact": exact("matched_heads", "wrong_id"),
            "wrong_source_token_exact": token_exact("wrong_source", "wrong_id"),
            "matched_head_wrong_source_token_exact": token_exact(
                "matched_heads", "wrong_id"
            ),
            "mean_margin_shift_to_wrong_source": shift("wrong_source"),
            "matched_head_mean_margin_shift": shift("matched_heads"),
        }

    overall = stats(rows)
    selected_exact = overall["wrong_source_exact"]
    control_exact = overall["matched_head_wrong_source_exact"]
    selected_shift = overall["mean_margin_shift_to_wrong_source"]
    control_shift = overall["matched_head_mean_margin_shift"]
    return {
        "schema_version": 1,
        **overall,
        "by_distance": {
            "adjacent": stats(
                [
                    row
                    for row in rows
                    if row["target_equation_index"] - row["source_equation_index"] == 1
                ]
            ),
            "delayed": stats(
                [
                    row
                    for row in rows
                    if row["target_equation_index"] - row["source_equation_index"] >= 2
                ]
            ),
        },
        "gate": {
            "exact_advantage_at_least_010": selected_exact is not None
            and control_exact is not None
            and selected_exact - control_exact >= 0.1,
            "positive_margin_advantage": selected_shift is not None
            and control_shift is not None
            and selected_shift > control_shift,
        },
    }
