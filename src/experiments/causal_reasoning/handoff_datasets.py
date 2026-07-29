"""Second-wave datasets for handoff bandwidth and query reuse."""

from __future__ import annotations

import random
from typing import Any

from .dataset_utils import (
    delimited_span_prompt,
    experiment_row,
    marker_prompt,
)


def _ledger_trace(state: int, order: list[int]) -> list[str]:
    current = 0
    lines = []
    for step, bit in enumerate(order, start=1):
        current |= 1 << bit
        bits = " ".join(
            str(int(bool(current & (1 << position))))
            for position in range(4)
        )
        lines.append(f"Ledger {step}: {bits}")
    while len(lines) < 4:
        bits = " ".join(
            str(int(bool(current & (1 << position))))
            for position in range(4)
        )
        lines.append(f"Ledger {len(lines) + 1}: {bits}")
    return lines


def _blank_trace() -> list[str]:
    return [f"Ledger {step}: 0 0 0 0" for step in range(1, 5)]


def _boundary_text(trace: list[str], query: int) -> str:
    return "\n".join(
        [
            "Track the four-bit fact ledger shown below.",
            "[TRACE START]",
            *trace,
            "[TRACE END]",
            "<<HANDOFF>>",
            f"Is fact {chr(65 + query)} true in the final ledger?",
            "Return 1 for yes or 0 for no.",
            "Answer=",
        ]
    )


def _boundary_prompts(
    trace: list[str], query: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = _boundary_text(trace, query)
    return (
        marker_prompt(text.splitlines(), "<<HANDOFF>>"),
        delimited_span_prompt(text),
    )


def build_boundary_bandwidth(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Sweep boundary and full-trace residual bandwidth after text removal."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        state = rng.randrange(1, 16)
        query = index % 4
        order = [bit for bit in range(4) if state & (1 << bit)]
        rng.shuffle(order)
        source_boundary, source_history = _boundary_prompts(
            _ledger_trace(state, order), query
        )
        blank_boundary, blank_history = _boundary_prompts(
            _blank_trace(), query
        )
        answer = int(bool(state & (1 << query)))
        rows.append(
            experiment_row(
                experiment="boundary_bandwidth",
                index=index,
                count=count,
                prompts={
                    "source_boundary": source_boundary,
                    "source_history": source_history,
                    "blank_boundary": blank_boundary,
                    "blank_history": blank_history,
                },
                evaluations=[
                    {
                        "name": "full_context",
                        "recipient": "source_boundary",
                        "expected": answer,
                    },
                    {
                        "name": "removed_context",
                        "recipient": "blank_boundary",
                        "expected": answer,
                    },
                    {
                        "name": "boundary_one_single",
                        "recipient": "blank_boundary",
                        "source": "source_boundary",
                        "expected": answer,
                        "token_width": 1,
                        "layer_modes": ["single"],
                    },
                    {
                        "name": "boundary_one_window3",
                        "recipient": "blank_boundary",
                        "source": "source_boundary",
                        "expected": answer,
                        "token_width": 1,
                        "layer_modes": ["window3"],
                    },
                    {
                        "name": "boundary_one_all_layers",
                        "recipient": "blank_boundary",
                        "source": "source_boundary",
                        "expected": answer,
                        "token_width": 1,
                        "layer_modes": ["all"],
                    },
                    {
                        "name": "boundary_full_all_layers",
                        "recipient": "blank_boundary",
                        "source": "source_boundary",
                        "expected": answer,
                        "token_width": "all",
                        "layer_modes": ["all"],
                    },
                    {
                        "name": "history_full_single",
                        "recipient": "blank_history",
                        "source": "source_history",
                        "expected": answer,
                        "token_width": "all",
                        "layer_modes": ["single"],
                    },
                    {
                        "name": "history_full_window3",
                        "recipient": "blank_history",
                        "source": "source_history",
                        "expected": answer,
                        "token_width": "all",
                        "layer_modes": ["window3"],
                    },
                    {
                        "name": "history_full_all_layers",
                        "recipient": "blank_history",
                        "source": "source_history",
                        "expected": answer,
                        "token_width": "all",
                        "layer_modes": ["all"],
                    },
                ],
                representation_pairs=[
                    {
                        "name": "full_vs_removed_boundary",
                        "left": "source_boundary",
                        "right": "blank_boundary",
                    },
                    {
                        "name": "full_vs_removed_history",
                        "left": "source_history",
                        "right": "blank_history",
                    },
                ],
                labels={"state": state, "query": query},
                candidates=["0", "1"],
            )
        )
    return rows


def _world_trace(state: int, mode: str, query: int) -> list[str]:
    lines = []
    for bit in range(4):
        value = int(bool(state & (1 << bit)))
        visible = mode == "complete" or (
            mode == "focused" and bit == query
        )
        lines.append(f"Slot {chr(65 + bit)}: {value if visible else 0}")
    return lines


def _query_text(
    *,
    state: int,
    original_query: int,
    new_query: int,
    trace_mode: str,
) -> str:
    world = " ".join(
        f"{chr(65 + bit)}={int(bool(state & (1 << bit)))}"
        for bit in range(4)
    )
    if trace_mode == "blank":
        world = "removed"
    trace = (
        _world_trace(state, "complete", original_query)
        if trace_mode == "complete"
        else _world_trace(state, "focused", original_query)
        if trace_mode == "focused"
        else _world_trace(state, "blank", original_query)
    )
    return "\n".join(
        [
            f"World description: {world}.",
            f"Original query concerned fact {chr(65 + original_query)}.",
            "[TRACE START]",
            *trace,
            "[TRACE END]",
            "<<WORLD STATE>>",
            f"New query: is fact {chr(65 + new_query)} true?",
            "Return 1 for yes or 0 for no.",
            "Answer=",
        ]
    )


def _query_prompts(
    *,
    state: int,
    original_query: int,
    new_query: int,
    trace_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    text = _query_text(
        state=state,
        original_query=original_query,
        new_query=new_query,
        trace_mode=trace_mode,
    )
    return (
        marker_prompt(text.splitlines(), "<<WORLD STATE>>"),
        delimited_span_prompt(text),
    )


def build_query_reuse(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Compare answer-specific, native, and query-independent world states."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        state = rng.randrange(1, 15)
        query_a = index % 4
        query_b = (query_a + 1 + index % 3) % 4
        a_boundary, a_history = _query_prompts(
            state=state,
            original_query=query_a,
            new_query=query_b,
            trace_mode="focused",
        )
        b_boundary, b_history = _query_prompts(
            state=state,
            original_query=query_b,
            new_query=query_b,
            trace_mode="focused",
        )
        complete_boundary, complete_history = _query_prompts(
            state=state,
            original_query=query_a,
            new_query=query_b,
            trace_mode="complete",
        )
        blank_boundary, blank_history = _query_prompts(
            state=state,
            original_query=query_b,
            new_query=query_b,
            trace_mode="blank",
        )
        answer = int(bool(state & (1 << query_b)))
        prompts = {
            "query_a_boundary": a_boundary,
            "query_a_history": a_history,
            "query_b_boundary": b_boundary,
            "query_b_history": b_history,
            "complete_boundary": complete_boundary,
            "complete_history": complete_history,
            "blank_boundary": blank_boundary,
            "blank_history": blank_history,
        }
        evaluations = [
            {
                "name": "query_a_text",
                "recipient": "query_a_boundary",
                "expected": answer,
            },
            {
                "name": "query_b_text",
                "recipient": "query_b_boundary",
                "expected": answer,
            },
            {
                "name": "complete_text",
                "recipient": "complete_boundary",
                "expected": answer,
            },
            {
                "name": "blank_text",
                "recipient": "blank_boundary",
                "expected": answer,
            },
            {
                "name": "query_a_boundary_swap",
                "recipient": "blank_boundary",
                "source": "query_a_boundary",
                "expected": answer,
                "token_width": "all",
                "layer_modes": ["all"],
            },
            {
                "name": "query_a_history_swap",
                "recipient": "blank_history",
                "source": "query_a_history",
                "expected": answer,
                "token_width": "all",
                "layer_modes": ["all"],
            },
            {
                "name": "query_b_history_swap",
                "recipient": "blank_history",
                "source": "query_b_history",
                "expected": answer,
                "token_width": "all",
                "layer_modes": ["all"],
            },
            {
                "name": "complete_history_swap",
                "recipient": "blank_history",
                "source": "complete_history",
                "expected": answer,
                "token_width": "all",
                "layer_modes": ["all"],
            },
        ]
        rows.append(
            experiment_row(
                experiment="query_reuse",
                index=index,
                count=count,
                prompts=prompts,
                evaluations=evaluations,
                representation_pairs=[
                    {
                        "name": "query_a_vs_query_b",
                        "left": "query_a_boundary",
                        "right": "query_b_boundary",
                    },
                    {
                        "name": "query_a_vs_complete",
                        "left": "query_a_boundary",
                        "right": "complete_boundary",
                    },
                    {
                        "name": "complete_vs_blank",
                        "left": "complete_boundary",
                        "right": "blank_boundary",
                    },
                ],
                labels={
                    "state": state,
                    "query_a": query_a,
                    "query_b": query_b,
                },
                candidates=["0", "1"],
            )
        )
    return rows
