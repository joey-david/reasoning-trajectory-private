"""Deterministic paired datasets for six causal reasoning questions."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable

from .alignment_datasets import (
    build_correction_hysteresis,
    build_prospective_utility,
    build_trace_alignment,
)
from .dataset_utils import marker_prompt as _prompt
from .dataset_utils import split as _split
from .handoff_datasets import (
    build_boundary_bandwidth,
    build_query_reuse,
)
from .memory_datasets import build_information_debt


EXPERIMENTS = (
    "equivalent_state",
    "future_utility",
    "reasoning_hysteresis",
    "unresolved_dependency",
    "boundary_handoff",
    "query_switch",
    "trace_alignment",
    "prospective_utility",
    "correction_hysteresis",
    "information_debt",
    "boundary_bandwidth",
    "query_reuse",
)


def _proof_steps(
    state: int, order: list[int], *, wording: str = "derive"
) -> list[str]:
    verbs = {
        "derive": "derive",
        "establish": "establish",
    }
    verb = verbs[wording]
    return [
        f"Step {index}: {verb} fact {chr(65 + bit)}."
        for index, bit in enumerate(order, start=1)
        if state & (1 << bit)
    ]


def _proof_prompt(
    *,
    state: int,
    order: list[int],
    query: int,
    original_query: int | None = None,
    marker: str = "<<STATE>>",
    include_steps: bool = True,
    wording: str = "derive",
) -> dict[str, Any]:
    lines = [
        "Facts A, B, C, and D start false.",
        "A derived fact stays true.",
    ]
    if original_query is not None:
        lines.append(f"Original query: is fact {chr(65 + original_query)} true?")
    if include_steps:
        lines.extend(_proof_steps(state, order, wording=wording))
    else:
        lines.append("The derivation has been removed.")
    lines.extend(
        [
            marker,
            f"Query: is fact {chr(65 + query)} true?",
            "Return 1 for yes or 0 for no.",
            "Answer=",
        ]
    )
    return _prompt(lines, marker)


def _different_state(state: int, query: int) -> int:
    return state ^ (1 << query)


def build_equivalent_state(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        state = rng.randrange(1, 15)
        query = index % 4
        active = [bit for bit in range(4) if state & (1 << bit)]
        target_order = active[:]
        rng.shuffle(target_order)
        donor_order = list(reversed(target_order))
        if donor_order == target_order and len(active) > 1:
            donor_order = target_order[1:] + target_order[:1]
        opposite_bits = [
            bit
            for bit in range(4)
            if bit != query
            and bool(state & (1 << bit)) != bool(state & (1 << query))
        ]
        counter_state = _different_state(state, query)
        if opposite_bits:
            counter_state = _different_state(counter_state, opposite_bits[0])
        counter_order = [
            bit for bit in range(4) if counter_state & (1 << bit)
        ]
        same_answer_state = state ^ (1 << ((query + 1) % 4))
        same_answer_order = [
            bit for bit in range(4) if same_answer_state & (1 << bit)
        ]
        rows.append(
            {
                "schema_version": 1,
                "id": f"equivalent_state_{index:04d}",
                "experiment": "equivalent_state",
                "group": f"equivalent_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": ["0", "1"],
                "prompts": {
                    "target": _proof_prompt(
                        state=state, order=target_order, query=query
                    ),
                    "same_state": _proof_prompt(
                        state=state,
                        order=donor_order,
                        query=query,
                        wording="establish",
                    ),
                    "different_state": _proof_prompt(
                        state=counter_state, order=counter_order, query=query
                    ),
                    "same_answer_state": _proof_prompt(
                        state=same_answer_state,
                        order=same_answer_order,
                        query=query,
                    ),
                },
                "evaluations": [
                    {
                        "name": "target_baseline",
                        "recipient": "target",
                        "expected": int(bool(state & (1 << query))),
                    },
                    {
                        "name": "same_state_swap",
                        "recipient": "target",
                        "source": "same_state",
                        "expected": int(bool(state & (1 << query))),
                    },
                    {
                        "name": "different_state_swap",
                        "recipient": "target",
                        "source": "different_state",
                        "expected": int(bool(counter_state & (1 << query))),
                    },
                    {
                        "name": "same_answer_wrong_state_swap",
                        "recipient": "target",
                        "source": "same_answer_state",
                        "expected": int(bool(state & (1 << query))),
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "same_state",
                        "left": "target",
                        "right": "same_state",
                    },
                    {
                        "name": "different_state",
                        "left": "target",
                        "right": "different_state",
                    },
                    {
                        "name": "same_answer_wrong_state",
                        "left": "target",
                        "right": "same_answer_state",
                    },
                ],
                "labels": {
                    "state": state,
                    "donor_state": counter_state,
                    "same_answer_state": same_answer_state,
                    "query": query,
                },
            }
        )
    return rows


def _utility_prompt(
    *,
    a: int,
    b: int,
    useful: str,
    dependencies: int,
    distance: int,
) -> dict[str, Any]:
    selected = a if useful == "A" else b
    lines = [
        f"Goal: after the work below, reuse register {useful} exactly "
        f"{dependencies} time(s) to obtain {selected}.",
        f"Compute register A: {a - 2} + 2 = {a}.",
        f"Compute register B: {b + 1} - 1 = {b}.",
        "<<STATE>>",
    ]
    lines.extend(
        f"Deferred note {step}: no register changes." for step in range(distance)
    )
    lines.extend(
        [
            "Which register is required by the stated goal?",
            "Return 0 for register A or 1 for register B.",
            "Answer=",
        ]
    )
    return _prompt(lines)


def build_future_utility(*, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    distances = (2, 8, 24)
    for index in range(count):
        a, b = rng.sample(range(2, 10), 2)
        useful = "A" if index % 2 == 0 else "B"
        opposite = "B" if useful == "A" else "A"
        dependencies = 1 + (index % 3)
        distance = distances[(index // 3) % len(distances)]
        rows.append(
            {
                "schema_version": 1,
                "id": f"future_utility_{index:04d}",
                "experiment": "future_utility",
                "group": f"utility_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": ["0", "1"],
                "prompts": {
                    "target": _utility_prompt(
                        a=a,
                        b=b,
                        useful=useful,
                        dependencies=dependencies,
                        distance=distance,
                    ),
                    "opposite": _utility_prompt(
                        a=a,
                        b=b,
                        useful=opposite,
                        dependencies=dependencies,
                        distance=distance,
                    ),
                },
                "evaluations": [
                    {
                        "name": "target_baseline",
                        "recipient": "target",
                        "expected": 0 if useful == "A" else 1,
                    },
                    {
                        "name": "opposite_utility_swap",
                        "recipient": "target",
                        "source": "opposite",
                        "expected": 0 if opposite == "A" else 1,
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "opposite_future_utility",
                        "left": "target",
                        "right": "opposite",
                    }
                ],
                "feature_prompt": "target",
                "labels": {
                    "will_reuse": 0 if useful == "A" else 1,
                    "dependency_count": dependencies,
                    "reuse_distance": distance,
                },
            }
        )
    return rows


def _hysteresis_prompt(
    *,
    x: int,
    add: int,
    multiplier: int,
    wrong: int,
    correction: bool,
    restart: bool,
) -> dict[str, Any]:
    correct = (x + add) % 10
    lines = [f"Compute (({x} + {add}) mod 10) * {multiplier} mod 10."]
    if restart:
        lines.append(f"Use the verified intermediate value {correct}.")
    else:
        lines.extend(
            [
                f"Initial calculation: ({x} + {add}) mod 10 = {wrong}.",
                f"Plan: multiply the intermediate value by {multiplier} mod 10.",
            ]
        )
        if correction:
            lines.append(f"Correction: the intermediate value is {correct}.")
        else:
            lines.append("No correction is supplied.")
    lines.extend(["<<STATE>>", "Return one digit.", "Answer="])
    return _prompt(lines)


def build_reasoning_hysteresis(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        x = rng.randrange(10)
        add = rng.randrange(1, 10)
        multiplier = rng.choice((2, 3, 7, 9))
        correct = (x + add) % 10
        wrong = (correct + rng.randrange(1, 10)) % 10
        correct_answer = (correct * multiplier) % 10
        wrong_answer = (wrong * multiplier) % 10
        if wrong_answer == correct_answer:
            multiplier = 1
            correct_answer = correct
            wrong_answer = wrong
        rows.append(
            {
                "schema_version": 1,
                "id": f"reasoning_hysteresis_{index:04d}",
                "experiment": "reasoning_hysteresis",
                "group": f"hysteresis_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": [str(value) for value in range(10)],
                "prompts": {
                    "text_corrected": _hysteresis_prompt(
                        x=x,
                        add=add,
                        multiplier=multiplier,
                        wrong=wrong,
                        correction=True,
                        restart=False,
                    ),
                    "uncorrected": _hysteresis_prompt(
                        x=x,
                        add=add,
                        multiplier=multiplier,
                        wrong=wrong,
                        correction=False,
                        restart=False,
                    ),
                    "restart": _hysteresis_prompt(
                        x=x,
                        add=add,
                        multiplier=multiplier,
                        wrong=wrong,
                        correction=True,
                        restart=True,
                    ),
                },
                "evaluations": [
                    {
                        "name": "text_correction",
                        "recipient": "text_corrected",
                        "expected": correct_answer,
                    },
                    {
                        "name": "hidden_correction",
                        "recipient": "uncorrected",
                        "source": "restart",
                        "expected": correct_answer,
                    },
                    {
                        "name": "text_and_hidden_correction",
                        "recipient": "text_corrected",
                        "source": "restart",
                        "expected": correct_answer,
                    },
                    {
                        "name": "old_plan_in_corrected_text",
                        "recipient": "text_corrected",
                        "source": "uncorrected",
                        "expected": wrong_answer,
                    },
                    {
                        "name": "corrected_restart",
                        "recipient": "restart",
                        "expected": correct_answer,
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "corrected_vs_uncorrected",
                        "left": "text_corrected",
                        "right": "uncorrected",
                    },
                    {
                        "name": "corrected_vs_restart",
                        "left": "text_corrected",
                        "right": "restart",
                    },
                ],
                "labels": {
                    "correct_answer": correct_answer,
                    "old_plan_answer": wrong_answer,
                },
            }
        )
    return rows


def _debt_prompt(
    *,
    value: int,
    add: int,
    multiplier: int,
    delay: int,
    debt_count: int,
    mode: str,
) -> dict[str, Any]:
    lines = [f"Goal: compute ((x + {add}) * {multiplier}) mod 10."]
    if mode == "upfront":
        lines.append(f"Known now: x = {value}.")
    else:
        lines.append("x is unresolved; keep its dependency open.")
        if debt_count == 2:
            lines.append("y is also unresolved but will not affect the answer.")
    lines.append("Symbolic plan: answer = ((x + addend) * multiplier) mod 10.")
    lines.extend(f"Delay line {step}: no value changes." for step in range(delay))
    if mode == "repaid":
        lines.append(f"Debt repayment: x = {value}.")
    elif mode == "irrelevant":
        lines.append(f"An irrelevant value arrives: y = {value}.")
    lines.extend(["<<STATE>>", "Return one digit.", "Answer="])
    return _prompt(lines)


def build_unresolved_dependency(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    distances = (2, 8, 24)
    rows = []
    for index in range(count):
        value = rng.randrange(10)
        add = rng.randrange(1, 10)
        multiplier = rng.choice((1, 3, 7, 9))
        delay = distances[index % len(distances)]
        debt_count = 1 + (index % 2)
        answer = ((value + add) * multiplier) % 10
        rows.append(
            {
                "schema_version": 1,
                "id": f"unresolved_dependency_{index:04d}",
                "experiment": "unresolved_dependency",
                "group": f"debt_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": [str(value) for value in range(10)],
                "prompts": {
                    "repaid": _debt_prompt(
                        value=value,
                        add=add,
                        multiplier=multiplier,
                        delay=delay,
                        debt_count=debt_count,
                        mode="repaid",
                    ),
                    "upfront": _debt_prompt(
                        value=value,
                        add=add,
                        multiplier=multiplier,
                        delay=delay,
                        debt_count=debt_count,
                        mode="upfront",
                    ),
                    "unpaid": _debt_prompt(
                        value=value,
                        add=add,
                        multiplier=multiplier,
                        delay=delay,
                        debt_count=debt_count,
                        mode="irrelevant",
                    ),
                },
                "evaluations": [
                    {
                        "name": "debt_repaid",
                        "recipient": "repaid",
                        "expected": answer,
                    },
                    {
                        "name": "known_upfront",
                        "recipient": "upfront",
                        "expected": answer,
                    },
                    {
                        "name": "known_state_swap",
                        "recipient": "repaid",
                        "source": "upfront",
                        "expected": answer,
                    },
                    {
                        "name": "unpaid_debt",
                        "recipient": "unpaid",
                        "expected": answer,
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "repaid_vs_upfront",
                        "left": "repaid",
                        "right": "upfront",
                    },
                    {
                        "name": "repaid_vs_unpaid",
                        "left": "repaid",
                        "right": "unpaid",
                    },
                ],
                "feature_prompt": "repaid",
                "labels": {
                    "debt_count": debt_count,
                    "reuse_distance": delay,
                    "recoverable": 1,
                },
            }
        )
    return rows


def build_boundary_handoff(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    marker = "<S1> <S2> <S3>"
    rows = []
    for index in range(count):
        state = rng.randrange(1, 15)
        query = index % 4
        order = [bit for bit in range(4) if state & (1 << bit)]
        rng.shuffle(order)
        rows.append(
            {
                "schema_version": 1,
                "id": f"boundary_handoff_{index:04d}",
                "experiment": "boundary_handoff",
                "group": f"boundary_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": ["0", "1"],
                "prompts": {
                    "full": _proof_prompt(
                        state=state,
                        order=order,
                        query=query,
                        marker=marker,
                    ),
                    "truncated": _proof_prompt(
                        state=state,
                        order=order,
                        query=query,
                        marker=marker,
                        include_steps=False,
                    ),
                },
                "evaluations": [
                    {
                        "name": "full_context",
                        "recipient": "full",
                        "expected": int(bool(state & (1 << query))),
                    },
                    {
                        "name": "truncated_no_patch",
                        "recipient": "truncated",
                        "expected": int(bool(state & (1 << query))),
                    },
                    {
                        "name": "one_token_one_layer",
                        "recipient": "truncated",
                        "source": "full",
                        "expected": int(bool(state & (1 << query))),
                        "token_width": 1,
                        "layer_modes": ["single"],
                    },
                    {
                        "name": "three_tokens_one_layer",
                        "recipient": "truncated",
                        "source": "full",
                        "expected": int(bool(state & (1 << query))),
                        "token_width": 3,
                        "layer_modes": ["single"],
                    },
                    {
                        "name": "one_token_all_layers",
                        "recipient": "truncated",
                        "source": "full",
                        "expected": int(bool(state & (1 << query))),
                        "token_width": 1,
                        "layer_modes": ["all"],
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "full_vs_truncated",
                        "left": "full",
                        "right": "truncated",
                    }
                ],
                "labels": {"state": state, "query": query},
            }
        )
    return rows


def build_query_switch(*, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        state = rng.randrange(1, 15)
        query_a = index % 4
        query_b = (query_a + 1 + (index % 3)) % 4
        order = [bit for bit in range(4) if state & (1 << bit)]
        rng.shuffle(order)
        expected = int(bool(state & (1 << query_b)))
        rows.append(
            {
                "schema_version": 1,
                "id": f"query_switch_{index:04d}",
                "experiment": "query_switch",
                "group": f"query_case_{index:04d}",
                "split": _split(index, count),
                "candidate_symbols": ["0", "1"],
                "prompts": {
                    "query_a_state": _proof_prompt(
                        state=state,
                        order=order,
                        query=query_b,
                        original_query=query_a,
                    ),
                    "query_b_state": _proof_prompt(
                        state=state,
                        order=list(reversed(order)),
                        query=query_b,
                        original_query=query_b,
                    ),
                    "query_b_minimal": _proof_prompt(
                        state=state,
                        order=order,
                        query=query_b,
                        original_query=query_b,
                        include_steps=False,
                    ),
                },
                "evaluations": [
                    {
                        "name": "textual_reuse_from_query_a",
                        "recipient": "query_a_state",
                        "expected": expected,
                    },
                    {
                        "name": "query_b_from_scratch",
                        "recipient": "query_b_minimal",
                        "expected": expected,
                    },
                    {
                        "name": "hidden_reuse_from_query_a",
                        "recipient": "query_b_minimal",
                        "source": "query_a_state",
                        "expected": expected,
                    },
                    {
                        "name": "native_query_b_state",
                        "recipient": "query_b_minimal",
                        "source": "query_b_state",
                        "expected": expected,
                    },
                ],
                "representation_pairs": [
                    {
                        "name": "query_a_vs_native_query_b",
                        "left": "query_a_state",
                        "right": "query_b_state",
                    },
                    {
                        "name": "query_a_vs_no_derivation",
                        "left": "query_a_state",
                        "right": "query_b_minimal",
                    },
                ],
                "labels": {
                    "state": state,
                    "query_a": query_a,
                    "query_b": query_b,
                },
            }
        )
    return rows


BUILDERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "equivalent_state": build_equivalent_state,
    "future_utility": build_future_utility,
    "reasoning_hysteresis": build_reasoning_hysteresis,
    "unresolved_dependency": build_unresolved_dependency,
    "boundary_handoff": build_boundary_handoff,
    "query_switch": build_query_switch,
}

BUILDERS.update(
    {
        "trace_alignment": build_trace_alignment,
        "prospective_utility": build_prospective_utility,
        "correction_hysteresis": build_correction_hysteresis,
        "information_debt": build_information_debt,
        "boundary_bandwidth": build_boundary_bandwidth,
        "query_reuse": build_query_reuse,
    }
)


def build_experiment_cases(
    experiment: str, *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Build one balanced, deterministic causal-question dataset."""
    if experiment not in BUILDERS:
        raise ValueError(f"Unknown causal reasoning experiment: {experiment!r}")
    rows = BUILDERS[experiment](count=count, seed=seed)
    validate_experiment_cases(rows, experiment=experiment, expected_count=count)
    return rows


def validate_experiment_cases(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    expected_count: int,
) -> dict[str, Any]:
    """Validate identities, prompt spans, splits, and intervention references."""
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Causal reasoning case IDs must be unique")
    for row in rows:
        if row["experiment"] != experiment:
            raise ValueError("Case experiment does not match its run")
        prompts = row["prompts"]
        for prompt in prompts.values():
            text = str(prompt["text"])
            start = int(prompt["checkpoint_start"])
            end = int(prompt["checkpoint_end"])
            if not (0 <= start < end <= len(text)) or not text.endswith("Answer="):
                raise ValueError(f"Malformed prompt span in {row['id']}")
        for evaluation in row["evaluations"]:
            if evaluation["recipient"] not in prompts:
                raise ValueError(f"Unknown recipient in {row['id']}")
            if evaluation.get("source") not in {None, *prompts}:
                raise ValueError(f"Unknown source in {row['id']}")
            expected = int(evaluation["expected"])
            if not 0 <= expected < len(row["candidate_symbols"]):
                raise ValueError(
                    f"Expected candidate is outside the alphabet in {row['id']}"
                )
            width = evaluation.get("token_width", 1)
            if width != "all" and int(width) < 1:
                raise ValueError(f"Invalid token width in {row['id']}")
        pair_names = {
            str(pair["name"])
            for pair in row.get("representation_pairs", [])
        }
        for pair in row.get("representation_pairs", []):
            if pair["left"] not in prompts or pair["right"] not in prompts:
                raise ValueError(
                    f"Unknown representation pair prompt in {row['id']}"
                )
        for evaluation in row["evaluations"]:
            pair = evaluation.get("representation_pair")
            if pair is not None and str(pair) not in pair_names:
                raise ValueError(
                    f"Unknown effect representation pair in {row['id']}"
                )
        if (
            row.get("feature_prompt") is not None
            and row["feature_prompt"] not in prompts
        ):
            raise ValueError(f"Unknown feature prompt in {row['id']}")
    payload = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows
    )
    return {
        "schema_version": 1,
        "experiment": experiment,
        "case_count": len(rows),
        "split_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "validation", "test")
        },
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
