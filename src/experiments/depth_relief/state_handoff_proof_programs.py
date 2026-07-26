"""Closed Horn transitions and reusable proof-state consumers."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .benchmark import apply_rule
from .state_handoff_programs import (
    _program_contexts,
    _proof_final_rule,
    _proof_state_symbols,
    _rng,
)


CLOSED_HORN_TRANSITION_CLASSES = (
    "blocked_unary",
    "blocked_conjunction",
    "idempotent",
    "active_unconditional",
    "active_unary",
    "active_conjunction",
)


def _closed_horn_initial_states(
    transition_class: str, width: int
) -> tuple[int, ...]:
    states = []
    for state in range(2**width):
        present = state.bit_count()
        absent = width - present
        feasible = {
            "blocked_unary": absent >= 1,
            "blocked_conjunction": absent >= 1 and width >= 2,
            "idempotent": present >= 1,
            "active_unconditional": absent >= 1,
            "active_unary": present >= 1 and absent >= 1,
            "active_conjunction": present >= 2 and absent >= 1,
        }[transition_class]
        if feasible:
            states.append(state)
    return tuple(states)


def _closed_horn_rule(
    *,
    initial: int,
    transition_class: str,
    width: int,
    rng: random.Random,
) -> dict[str, Any]:
    present = [bit for bit in range(width) if initial & (1 << bit)]
    absent = [bit for bit in range(width) if not initial & (1 << bit)]
    if transition_class == "blocked_unary":
        blocked = rng.choice(absent)
        premises = [blocked]
        conclusion = rng.randrange(width)
    elif transition_class == "blocked_conjunction":
        blocked = rng.choice(absent)
        other = rng.choice([bit for bit in range(width) if bit != blocked])
        premises = sorted((blocked, other))
        conclusion = rng.randrange(width)
    elif transition_class == "idempotent":
        conclusion = rng.choice(present)
        premises = sorted(rng.sample(present, k=min(2, len(present))))
    elif transition_class == "active_unconditional":
        premises = []
        conclusion = rng.choice(absent)
    elif transition_class == "active_unary":
        premises = [rng.choice(present)]
        conclusion = rng.choice(absent)
    elif transition_class == "active_conjunction":
        premises = sorted(rng.sample(present, k=2))
        conclusion = rng.choice(absent)
    else:
        raise ValueError(f"Unknown closed Horn transition class: {transition_class}")
    return {"kind": "horn", "premises": premises, "conclusion": conclusion}


def proof_next_rule(
    *,
    state: int,
    width: int,
    seed: int,
    desired_index: int | None = None,
) -> dict[str, Any]:
    """Build four candidate rules with one state-changing choice when possible."""
    rng = _rng(seed, "proof_next_rule", state, desired_index)
    present = [bit for bit in range(width) if state & (1 << bit)]
    absent = [bit for bit in range(width) if not state & (1 << bit)]
    if not absent:
        candidates = [
            {
                "kind": "horn",
                "premises": [present[index % len(present)]],
                "conclusion": present[index % len(present)],
            }
            for index in range(4)
        ]
        return {"kind": "proof_next_rule", "candidates": candidates}
    active_index = (
        rng.randrange(4) if desired_index is None else int(desired_index) % 4
    )
    conclusion = rng.choice(absent)
    active = {
        "kind": "horn",
        "premises": ([rng.choice(present)] if present else []),
        "conclusion": conclusion,
    }
    candidates = []
    for index in range(4):
        if index == active_index:
            candidates.append(active)
        elif present and index % 2:
            fact = present[index % len(present)]
            candidates.append(
                {"kind": "horn", "premises": [fact], "conclusion": fact}
            )
        else:
            blocked = absent[(index + seed) % len(absent)]
            candidates.append(
                {
                    "kind": "horn",
                    "premises": [blocked],
                    "conclusion": blocked,
                }
            )
    applicable = [
        index
        for index, candidate in enumerate(candidates)
        if apply_rule(candidate, state, 2**width) != state
    ]
    if applicable != [active_index]:
        raise AssertionError("Proof next-rule consumer is not uniquely applicable")
    return {"kind": "proof_next_rule", "candidates": candidates}


def proof_consumer_rule(
    *,
    context: dict[str, Any],
    state: int,
    width: int,
    seed: int,
    dataset: dict[str, Any],
    path_code: int,
) -> tuple[dict[str, Any], list[str]]:
    consumers = tuple(
        str(value) for value in dataset.get("proof_consumers", ("query",))
    )
    if not consumers or any(value not in {"query", "next_rule"} for value in consumers):
        raise ValueError("Proof consumers must contain query and/or next_rule")
    consumer = consumers[(int(context["index"]) // 2 + path_code) % len(consumers)]
    if consumer == "query":
        return (
            _proof_final_rule(context=context, width=width, seed=seed),
            ["0", "1"],
        )
    rule = proof_next_rule(
        state=state,
        width=width,
        seed=seed + 101 * int(context["index"]) + path_code,
        desired_index=(int(context["index"]) + path_code) % 4,
    )
    return rule, ["0", "1", "2", "3", "4"]


def build_closed_horn_programs(
    *,
    split: str,
    semantic_count: int,
    context_count: int,
    width: int,
    seed: int,
    dataset: dict[str, Any],
) -> list[dict[str, Any]]:
    """Cover the one-rule Horn transition relation, including every identity."""
    if width not in (4, 5):
        raise ValueError("Closed Horn transitions require four or five fact bits")
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    if semantic_count < len(CLOSED_HORN_TRANSITION_CLASSES) * context_count:
        raise ValueError(
            f"{split} is too small to cover every context-transition class"
        )
    rows = []
    modulus = 2**width
    symbols = list(_proof_state_symbols(width))
    for index in range(semantic_count):
        transition_index = index % len(CLOSED_HORN_TRANSITION_CLASSES)
        transition_class = CLOSED_HORN_TRANSITION_CLASSES[transition_index]
        context_index = (index // len(CLOSED_HORN_TRANSITION_CLASSES)) % len(
            contexts
        )
        context = contexts[context_index]
        path_code = index // (
            len(CLOSED_HORN_TRANSITION_CLASSES) * len(contexts)
        )
        feasible = _closed_horn_initial_states(transition_class, width)
        initial = feasible[(path_code + context_index) % len(feasible)]
        rng = _rng(
            seed,
            "closed_horn",
            split,
            context["id"],
            transition_class,
            initial,
            path_code,
        )
        rule = _closed_horn_rule(
            initial=initial,
            transition_class=transition_class,
            width=width,
            rng=rng,
        )
        current = apply_rule(rule, initial, modulus)
        changed = current != initial
        if changed != transition_class.startswith("active_"):
            raise AssertionError("Closed Horn transition class changed incorrectly")
        final_rule, proof_answer_symbols = proof_consumer_rule(
            context=context,
            state=current,
            width=width,
            seed=seed,
            dataset=dataset,
            path_code=path_code,
        )
        semantic = {
            "family": f"horn_proof_to_{final_rule['kind']}",
            "history_family": "horn_proof",
            "final_family": str(final_rule["kind"]),
            "format": "prose",
            "bits": width,
            "initial_state": initial,
            "history": [rule],
            "final_rule": final_rule,
            "current_state": current,
            "next_state": apply_rule(final_rule, current, modulus),
            "history_steps": 1,
            "state_path": [initial, current],
            "path_code": path_code,
            "program_context": str(context["id"]),
            "program_context_split": split,
            "abstraction_group": str(context["id"]),
            "abstraction_split": split,
            "domain": "horn_proof",
            "composition_split": "seen",
            "proof_template": "closed_one_rule_transition",
            "proof_transition_class": transition_class,
            "proof_composition_active": transition_class == "active_conjunction",
            "active_transition_count": int(changed),
            "answer_symbols": proof_answer_symbols,
            "proof_consumer": (
                "next_rule"
                if final_rule["kind"] == "proof_next_rule"
                else "query"
            ),
            "state_representation": "opaque_fact_set",
            "state_symbols": symbols,
        }
        digest = hashlib.sha256(
            json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        semantic["id"] = (
            f"handoff_{split}_{context['id']}_closed_{transition_class}_"
            f"s{initial}_p{path_code}_{digest}"
        )
        rows.append(semantic)
    return rows
