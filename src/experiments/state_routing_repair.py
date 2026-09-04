"""Value-preserving steering of final state reads."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import numpy as np
import torch

from src.experiments.state_routing_heads import _ANSWER_TOKEN_RE, token_contract
from src.experiments.state_routing_horizon import _token_positions
from src.models.introspection import get_decoder_layers, get_input_device


STEERING_ALPHAS = (0.1, 0.25, 0.5, 0.75)
_EVENT_RE = re.compile(r"(?m)^\s*\d+\.\s+\w+\s+must\s+(add|subtract)\s+(\d+)\.")
_LINE_RE = re.compile(r"(?m)^\s*(?P<step>\d+)[.)]\s*(?P<body>.*)$")


def steering_eligible(
    row: dict[str, Any],
    measurement: dict[str, Any],
    free_row: dict[str, Any],
) -> bool:
    """Return whether the trace contains two usable pre-answer target writes."""
    answer = _ANSWER_TOKEN_RE.search(str(measurement["generation"]["text"]))
    if answer is None or not free_row.get("eligible_routing"):
        return False
    return (
        sum(
            write["name"] == row["target"]
            and write["literal_digit"]
            and int(write["char_span"][1]) <= answer.start("answer")
            for write in free_row["parsed_writes"]
        )
        >= 2
    )


def trace_context(
    tokenizer: Any,
    row: dict[str, Any],
    measurement: dict[str, Any],
    free_row: dict[str, Any],
) -> tuple[str, int, int]:
    """Rebuild a pre-answer context with its valid and stale target writes."""
    generated = str(measurement["generation"]["text"])
    answer = _ANSWER_TOKEN_RE.search(generated)
    if answer is None or not steering_eligible(row, measurement, free_row):
        raise ValueError("steering needs two literal pre-answer target writes")
    writes = [
        write
        for write in free_row["parsed_writes"]
        if write["name"] == row["target"]
        and write["literal_digit"]
        and int(write["char_span"][1]) <= answer.start("answer")
    ]
    question = str(row["question"])
    context = question + generated[: answer.start("answer")]

    def position(write: dict[str, Any]) -> int:
        start, end = map(int, write["char_span"])
        return _token_positions(
            tokenizer, context, [len(question) + start, len(question) + end]
        )[0]

    return context, position(writes[-1]), position(writes[-2])


def _forward(model: Any, tokenizer: Any, context: str) -> torch.Tensor:
    ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(get_input_device(model))
    with torch.inference_mode():
        return model(input_ids=ids, use_cache=False).logits[0, -1].float()


def _steer(
    model: Any,
    tokenizer: Any,
    context: str,
    source_position: int,
    heads: list[dict[str, int]],
    alpha: float,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for spec in heads:
        by_layer[int(spec["layer"])].append(int(spec["head"]))
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    groups = num_heads // num_kv_heads
    values: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index, selected in by_layer.items():
        attention = get_decoder_layers(model)[layer_index].self_attn
        width = attention.o_proj.in_features // num_heads

        def capture_value(
            _module: Any,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            values[index] = output[0]

        def patch(
            _module: Any,
            inputs: tuple[torch.Tensor, ...],
            *,
            index: int = layer_index,
            layer_heads: tuple[int, ...] = tuple(selected),
            head_width: int = width,
        ) -> tuple[torch.Tensor, ...]:
            result = inputs[0].clone()
            value_width = values[index].shape[-1] // num_kv_heads
            for head in layer_heads:
                head_slice = slice(head * head_width, (head + 1) * head_width)
                kv_head = head // groups
                value_slice = slice(kv_head * value_width, (kv_head + 1) * value_width)
                current = result[0, -1, head_slice]
                source = values[index][source_position, value_slice]
                result[0, -1, head_slice] = current.lerp(source, alpha)
            return (result, *inputs[1:])

        handles.extend(
            (
                attention.v_proj.register_forward_hook(capture_value),
                attention.o_proj.register_forward_pre_hook(patch),
            )
        )
    ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(get_input_device(model))
    try:
        with torch.inference_mode():
            return model(input_ids=ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()


def _score(
    logits: torch.Tensor, candidate_ids: list[int], answer: int
) -> dict[str, Any]:
    values = logits[candidate_ids].detach().cpu().numpy()
    return {
        "prediction": int(np.argmax(values)),
        "correct_margin": float(values[answer] - np.max(np.delete(values, answer))),
    }


def steering_case(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    measurement: dict[str, Any],
    free_row: dict[str, Any],
    head_spec: dict[str, Any],
    alphas: list[float],
) -> dict[str, Any]:
    """Steer fixed heads to the target's own value vector, never a donor value."""
    context, valid, stale = trace_context(tokenizer, row, measurement, free_row)
    selected = head_spec["selected"]
    controls = head_spec["layer_matched_controls"]
    logits = {"baseline": _forward(model, tokenizer, context)}
    for alpha in alphas:
        suffix = f"{alpha:g}"
        for name, position, heads in (
            ("valid", valid, selected),
            ("stale", stale, selected),
            ("control", valid, controls),
        ):
            logits[f"{name}_{suffix}"] = _steer(
                model, tokenizer, context, position, heads, alpha
            )
    contract = token_contract(tokenizer, row)
    candidate_ids = [int(contract["candidate_ids"][digit]) for digit in range(10)]
    answer = int(row["clean_answer"])
    return {
        "schema_version": 1,
        "id": row["id"],
        "split": row["split"],
        "event_count": row["event_count"],
        "target_was_correct": bool(measurement["generation_correct"]),
        "saved_prediction": measurement["generation"]["answer"],
        "answer": answer,
        "conditions": {
            name: _score(value, candidate_ids, answer) for name, value in logits.items()
        },
    }


def intermediate_read_sites(
    row: dict[str, Any], measurement: dict[str, Any], free_row: dict[str, Any]
) -> list[dict[str, Any]]:
    """Find parsed state reads backed by a correct earlier model write."""
    generated = str(measurement["generation"]["text"])
    lines = {
        int(match.group("step")) - 1: match for match in _LINE_RE.finditer(generated)
    }
    events = list(_EVENT_RE.finditer(str(row["question"])))
    writes = {int(write["event"]): write for write in free_row["parsed_writes"]}
    gold = {
        index: int(
            row["clean"]["text"][slice(*row["clean"]["spans"][write["span_key"]])]
        )
        for index, write in enumerate(row["clean"]["writes"])
    }
    prior: dict[str, int] = {}
    sites = []
    for event, expected in enumerate(row["clean"]["writes"]):
        name = str(expected["name"])
        source_event = prior.get(name)
        prior[name] = event
        if source_event is None or event not in lines or event >= len(events):
            continue
        source = writes.get(source_event)
        if (
            source is None
            or not source["literal_digit"]
            or int(source["value"]) != gold[source_event]
        ):
            continue
        operation, amount_text = events[event].groups()
        amount = int(amount_text)
        symbol = "+" if operation == "add" else "-"
        operand = re.search(
            rf"(?<![\d-])(?P<value>\d)(?!\d)\s*{re.escape(symbol)}\s*{amount}"
            r"\s*(?:mod(?:ulo)?\s*10\s*)?=",
            lines[event].group("body"),
            re.IGNORECASE,
        )
        if operand is None:
            continue
        wrong_sources = [
            write
            for earlier, write in writes.items()
            if earlier < event
            and write["name"] != name
            and write["literal_digit"]
            and int(write["value"]) == gold[earlier]
            and int(write["value"]) != gold[source_event]
        ]
        if not wrong_sources:
            continue
        wrong_source = max(wrong_sources, key=lambda write: int(write["event"]))
        start = lines[event].start("body") + operand.start("value")
        sites.append(
            {
                "id": f"{row['id']}:{event}",
                "event": event,
                "split": row["split"],
                "operand_start": start,
                "saved_operand": int(operand.group("value")),
                "answer": gold[source_event],
                "result_answer": gold[event],
                "operation": operation,
                "amount": amount,
                "source_span": source["char_span"],
                "wrong_source_span": wrong_source["char_span"],
                "wrong_source_value": int(wrong_source["value"]),
            }
        )
    return sites


def intermediate_steering_case(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    measurement: dict[str, Any],
    site: dict[str, Any],
    head_spec: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    """Repair one state read, then test the next arithmetic write."""
    question = str(row["question"])
    generated = str(measurement["generation"]["text"])
    context = question + generated[: int(site["operand_start"])]
    offset = len(question)

    def position(span: list[int]) -> int:
        return _token_positions(
            tokenizer, context, [offset + int(span[0]), offset + int(span[1])]
        )[0]

    selected = head_spec["selected"]
    operands = {
        "baseline": _forward(model, tokenizer, context),
        "valid": _steer(
            model, tokenizer, context, position(site["source_span"]), selected, alpha
        ),
        "wrong_source": _steer(
            model,
            tokenizer,
            context,
            position(site["wrong_source_span"]),
            selected,
            alpha,
        ),
        "control_heads": _steer(
            model,
            tokenizer,
            context,
            position(site["source_span"]),
            head_spec["layer_matched_controls"],
            alpha,
        ),
    }
    contract = token_contract(tokenizer, row)
    candidate_ids = [int(contract["candidate_ids"][digit]) for digit in range(10)]
    operand_scores = {
        name: _score(logits, candidate_ids, int(site["answer"]))
        for name, logits in operands.items()
    }
    symbol = "+" if site["operation"] == "add" else "-"
    result_scores = {}
    for name, score in operand_scores.items():
        result_context = (
            context + str(score["prediction"]) + f" {symbol} {site['amount']} mod 10 ="
        )
        result_scores[name] = _score(
            _forward(model, tokenizer, result_context),
            candidate_ids,
            int(site["result_answer"]),
        )
    return {
        "schema_version": 1,
        **site,
        "conditions": {
            name: {"operand": operand_scores[name], "result": result_scores[name]}
            for name in operands
        },
    }


def summarize_intermediate_steering(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score state-read repair and its next computed write on held-out traces."""
    aligned = [
        row
        for row in rows
        if row["conditions"]["baseline"]["operand"]["prediction"]
        == row["saved_operand"]
    ]
    test = [row for row in aligned if row["split"] == "test"]
    failed = [row for row in test if row["saved_operand"] != row["answer"]]
    correct = [row for row in test if row["saved_operand"] == row["answer"]]

    def rates(name: str, subset: list[dict[str, Any]]) -> dict[str, float] | None:
        if not subset:
            return None
        operand = [
            row["conditions"][name]["operand"]["prediction"] == row["answer"]
            for row in subset
        ]
        result = [
            row["conditions"][name]["result"]["prediction"] == row["result_answer"]
            for row in subset
        ]
        return {
            "read": sum(operand) / len(subset),
            "next_write": sum(a and b for a, b in zip(operand, result)) / len(subset),
        }

    def wrong_source_follow(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(
            row["conditions"]["wrong_source"]["operand"]["prediction"]
            == row["wrong_source_value"]
            for row in subset
        ) / len(subset)

    return {
        "schema_version": 1,
        "case_count": len(rows),
        "baseline_aligned": len(aligned),
        "test_failed_reads": len(failed),
        "test_correct_reads": len(correct),
        "failed_read_repair": {
            name: rates(name, failed)
            for name in ("valid", "wrong_source", "control_heads")
        },
        "correct_read_outcomes": {
            name: rates(name, correct) for name in ("baseline", "valid")
        },
        "wrong_source_follow": {
            "failed_reads": wrong_source_follow(failed),
            "correct_reads": wrong_source_follow(correct),
        },
    }


def summarize_steering(
    rows: list[dict[str, Any]], alphas: list[float], selected_alpha: float | None = None
) -> dict[str, Any]:
    """Choose alpha on short traces, then score the fixed long-trace repair."""
    aligned = [
        row
        for row in rows
        if row["conditions"]["baseline"]["prediction"] == row["saved_prediction"]
    ]

    def split_rows(split: str, correct: bool) -> list[dict[str, Any]]:
        return [
            row
            for row in aligned
            if row["split"] == split and row["target_was_correct"] == correct
        ]

    def accuracy(condition: str, subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(
            row["conditions"][condition]["prediction"] == row["answer"]
            for row in subset
        ) / len(subset)

    development_failed = split_rows("development", False)
    development_correct = split_rows("development", True)
    development = []
    for alpha in alphas:
        key = f"valid_{alpha:g}"
        development.append(
            {
                "alpha": alpha,
                "repair": accuracy(key, development_failed),
                "preservation": accuracy(key, development_correct),
            }
        )
    if selected_alpha is None:
        safe = [row for row in development if (row["preservation"] or 0.0) >= 0.9]
        selected_alpha = max(
            safe or development,
            key=lambda row: (row["repair"] or 0.0, row["preservation"] or 0.0),
        )["alpha"]
    if selected_alpha not in alphas:
        raise ValueError(f"selected alpha {selected_alpha} was not run")
    suffix = f"{selected_alpha:g}"
    test_failed = split_rows("test", False)
    test_correct = split_rows("test", True)
    repair = accuracy(f"valid_{suffix}", test_failed)
    control = accuracy(f"control_{suffix}", test_failed)
    stale = accuracy(f"stale_{suffix}", test_failed)
    preservation = accuracy(f"valid_{suffix}", test_correct)
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "baseline_aligned": len(aligned),
        "development": development,
        "selected_alpha": selected_alpha,
        "test": {
            "failed_reads": len(test_failed),
            "correct_reads": len(test_correct),
            "valid_repair": repair,
            "matched_head_repair": control,
            "stale_source_repair": stale,
            "preservation": preservation,
        },
        "gate": {
            "repair_advantage_at_least_010": repair is not None
            and control is not None
            and stale is not None
            and repair - max(control, stale) >= 0.1,
            "preservation_at_least_090": preservation is not None
            and preservation >= 0.9,
        },
    }
