"""Q/K-only repair of final reads in model-written reasoning traces."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from src.experiments.depth_relief.metrics import bootstrap_mean_ci
from src.experiments.state_routing_heads import _ANSWER_TOKEN_RE, token_contract
from src.experiments.state_routing_horizon import _token_positions
from src.models.introspection import get_decoder_layers, get_input_device


def trace_context(
    tokenizer: Any,
    row: dict[str, Any],
    measurement: dict[str, Any],
    free_row: dict[str, Any],
) -> tuple[str, int]:
    """Rebuild the exact pre-answer context and its last target-write token."""
    generated = str(measurement["generation"]["text"])
    answer = _ANSWER_TOKEN_RE.search(generated)
    if answer is None:
        raise ValueError("repair row has no answer marker")
    target_writes = [
        write
        for write in free_row["parsed_writes"]
        if write["name"] == row["target"]
        and int(write["char_span"][1]) <= answer.start("answer")
    ]
    if not target_writes:
        raise ValueError("repair row has no target write before its answer")
    source = target_writes[-1]
    prefix = generated[: answer.start("answer")]
    question = str(row["question"])
    context = question + prefix
    start, end = map(int, source["char_span"])
    position = _token_positions(
        tokenizer, context, [len(question) + start, len(question) + end]
    )[0]
    return context, position


def select_donors(
    target: dict[str, Any],
    free_rows: list[dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Choose fixed short donors with the same variable and a different answer."""
    pools: dict[str, list[str]] = defaultdict(list)
    target_row = dataset[str(target["id"])]
    for row in sorted(free_rows, key=lambda value: str(value["id"])):
        source = dataset[str(row["id"])]
        if (
            row.get("eligible_routing")
            and source["split"] == "development"
            and source["target"] == target_row["target"]
            and source["clean_answer"] != target_row["clean_answer"]
        ):
            pools["successful" if row["generation_correct"] else "failed"].append(
                str(row["id"])
            )
    if not pools["successful"] or not pools["failed"]:
        raise ValueError(f"no matched donors for {target['id']}")
    return {name: values[0] for name, values in pools.items()}


def _capture_qk(
    model: Any,
    tokenizer: Any,
    context: str,
    source_position: int,
    layers: set[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor]:
    captures: dict[int, dict[str, torch.Tensor]] = defaultdict(dict)
    handles = []
    for layer_index in layers:
        attention = get_decoder_layers(model)[layer_index].self_attn
        for name in ("q_proj", "k_proj"):
            projection = getattr(attention, name)

            def capture(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = layer_index,
                key: str = name,
            ) -> None:
                position = -1 if key == "q_proj" else source_position
                captures[index][key] = output[0, position].detach().clone()

            handles.append(projection.register_forward_hook(capture))
    ids = tokenizer(context, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(get_input_device(model))
    try:
        with torch.inference_mode():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    return dict(captures), logits


def _patched_logits(
    model: Any,
    tokenizer: Any,
    context: str,
    source_position: int,
    donor: dict[int, dict[str, torch.Tensor]],
    heads: list[dict[str, int]],
    *,
    patch_q: bool,
    patch_k: bool,
) -> torch.Tensor:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for spec in heads:
        by_layer[int(spec["layer"])].append(int(spec["head"]))
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    groups = num_heads // num_kv_heads
    handles = []
    for layer_index, layer_heads in by_layer.items():
        attention = get_decoder_layers(model)[layer_index].self_attn
        if patch_q:
            q_width = attention.q_proj.out_features // num_heads

            def patch_query(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = layer_index,
                selected: tuple[int, ...] = tuple(layer_heads),
                width: int = q_width,
            ) -> torch.Tensor:
                result = output.clone()
                for head in selected:
                    slc = slice(head * width, (head + 1) * width)
                    result[0, -1, slc] = donor[index]["q_proj"][slc]
                return result

            handles.append(attention.q_proj.register_forward_hook(patch_query))
        if patch_k:
            k_width = attention.k_proj.out_features // num_kv_heads
            kv_heads = tuple(sorted({head // groups for head in layer_heads}))

            def patch_key(
                _module: Any,
                _inputs: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                *,
                index: int = layer_index,
                selected: tuple[int, ...] = kv_heads,
                width: int = k_width,
            ) -> torch.Tensor:
                result = output.clone()
                for head in selected:
                    slc = slice(head * width, (head + 1) * width)
                    result[0, source_position, slc] = donor[index]["k_proj"][slc]
                return result

            handles.append(attention.k_proj.register_forward_hook(patch_key))
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


def repair_case(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    measurement: dict[str, Any],
    free_row: dict[str, Any],
    successful: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    failed: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    head_spec: dict[str, Any],
) -> dict[str, Any]:
    """Patch only read-query and source-key vectors; keep target values fixed."""
    context, source = trace_context(tokenizer, row, measurement, free_row)
    selected = head_spec["selected"]
    controls = head_spec["layer_matched_controls"]
    layers = {int(spec["layer"]) for spec in selected}

    donor_captures = {}
    donor_ids = {}
    for name, (donor_row, donor_measurement, donor_free) in (
        ("successful", successful),
        ("failed", failed),
    ):
        donor_context, donor_source = trace_context(
            tokenizer, donor_row, donor_measurement, donor_free
        )
        donor_captures[name], _ = _capture_qk(
            model, tokenizer, donor_context, donor_source, layers
        )
        donor_ids[name] = donor_row["id"]
    _, baseline = _capture_qk(model, tokenizer, context, source, layers)
    conditions = {"baseline": baseline}
    for name, patch_q, patch_k, heads, donor_name in (
        ("successful_q", True, False, selected, "successful"),
        ("successful_k", False, True, selected, "successful"),
        ("successful_qk", True, True, selected, "successful"),
        ("control_heads_qk", True, True, controls, "successful"),
        ("failed_donor_qk", True, True, selected, "failed"),
    ):
        conditions[name] = _patched_logits(
            model,
            tokenizer,
            context,
            source,
            donor_captures[donor_name],
            heads,
            patch_q=patch_q,
            patch_k=patch_k,
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
        "donors": donor_ids,
        "conditions": {
            name: _score(logits, candidate_ids, answer)
            for name, logits in conditions.items()
        },
    }


def summarize_repairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Q/K repair with equal-size head and failed-donor controls."""
    aligned = [
        row
        for row in rows
        if row["conditions"]["baseline"]["prediction"] == row["saved_prediction"]
    ]
    failures = [row for row in aligned if not row["target_was_correct"]]
    correct = [row for row in aligned if row["target_was_correct"]]

    def accuracy(condition: str, subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(
            row["conditions"][condition]["prediction"] == row["answer"]
            for row in subset
        ) / len(subset)

    def margin_change(condition: str) -> dict[str, Any]:
        return bootstrap_mean_ci(
            [
                row["conditions"][condition]["correct_margin"]
                - row["conditions"]["baseline"]["correct_margin"]
                for row in failures
            ],
            seed=260530235,
        )

    repair = {
        name: accuracy(name, failures)
        for name in (
            "successful_q",
            "successful_k",
            "successful_qk",
            "control_heads_qk",
            "failed_donor_qk",
        )
    }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "baseline_aligned": len(aligned),
        "failed_reads": len(failures),
        "correct_reads": len(correct),
        "repair_accuracy": repair,
        "margin_change": {
            name: margin_change(name)
            for name in (
                "successful_q",
                "successful_k",
                "successful_qk",
                "control_heads_qk",
                "failed_donor_qk",
            )
        },
        "preservation": accuracy("successful_qk", correct),
    }
