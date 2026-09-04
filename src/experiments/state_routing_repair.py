"""Value-preserving steering of final state reads."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from src.experiments.state_routing_heads import _ANSWER_TOKEN_RE, token_contract
from src.experiments.state_routing_horizon import _token_positions
from src.models.introspection import get_decoder_layers, get_input_device


STEERING_ALPHAS = (0.1, 0.25, 0.5, 0.75)


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


def _capture(
    model: Any,
    tokenizer: Any,
    context: str,
    layers: set[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    values: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layers:
        attention = get_decoder_layers(model)[layer_index].self_attn

        def value_hook(
            _module: Any,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
        ) -> None:
            values[index] = output[0].detach().clone()

        handles.append(attention.v_proj.register_forward_hook(value_hook))
    ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(get_input_device(model))
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    return values, logits


def _steer(
    model: Any,
    tokenizer: Any,
    context: str,
    source_position: int,
    values: dict[int, torch.Tensor],
    heads: list[dict[str, int]],
    alpha: float,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for spec in heads:
        by_layer[int(spec["layer"])].append(int(spec["head"]))
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    groups = num_heads // num_kv_heads
    handles = []
    for layer_index, selected in by_layer.items():
        projection = get_decoder_layers(model)[layer_index].self_attn.o_proj
        width = projection.in_features // num_heads

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

        handles.append(projection.register_forward_pre_hook(patch))
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
    layers = {int(spec["layer"]) for spec in selected}
    values, baseline = _capture(model, tokenizer, context, layers)
    logits = {"baseline": baseline}
    for alpha in alphas:
        suffix = f"{alpha:g}"
        for name, position, heads in (
            ("valid", valid, selected),
            ("stale", stale, selected),
            ("control", valid, controls),
        ):
            logits[f"{name}_{suffix}"] = _steer(
                model, tokenizer, context, position, values, heads, alpha
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
