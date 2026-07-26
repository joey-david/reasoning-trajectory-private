"""Small semantic program banks that isolate effective reasoning depth."""

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
)
from .state_handoff_proof_programs import proof_next_rule


def _proof_rules(
    *,
    bits: tuple[int, ...],
    initial_bits: tuple[int, ...] = (),
    target_bits: tuple[int, ...] | None = None,
    width: int = 4,
    horizon: int,
    path: int,
    seed: int,
    topology: str = "mixed",
) -> list[dict[str, Any]]:
    if topology not in {"mixed", "independent", "chain", "conjunction"}:
        raise ValueError(f"Unknown proof topology: {topology}")
    rng = random.Random(seed + 10_007 * path + 101 * sum(bits))
    active_positions = (
        tuple(
            min(horizon - 1, (index + 1) * horizon // (len(bits) + 1))
            for index in range(len(bits))
        )
        if bits
        else ()
    )
    active_by_position = dict(zip(active_positions, bits))
    rules = []
    established = list(initial_bits)
    target_bits = bits if target_bits is None else target_bits
    absent = [bit for bit in range(width) if bit not in target_bits]
    for position in range(horizon):
        if position in active_by_position:
            conclusion = active_by_position[position]
            if topology == "independent" or not established:
                premises = []
            elif topology == "chain":
                premises = established[-1:]
            elif topology == "conjunction":
                premises = established[-min(2, len(established)) :]
            else:
                premises = established[-min(2, len(established)) :]
            established.append(conclusion)
        elif absent:
            blocked = absent[(position + path) % len(absent)]
            premises = [blocked]
            conclusion = blocked
        elif established:
            conclusion = established[(position + path) % len(established)]
            premises = [conclusion]
        else:
            blocked = rng.randrange(4)
            premises = [blocked]
            conclusion = blocked
        rules.append(
            {
                "kind": "horn",
                "premises": sorted(premises),
                "conclusion": conclusion,
            }
        )
    return rules


def _balanced_proof_query(
    *,
    target: int,
    desired_answer: int,
    width: int,
    seed: int,
) -> dict[str, Any]:
    """Choose a familiar query whose answer is controlled for nonempty states."""
    if target <= 0:
        raise ValueError("Balanced proof queries require a nonempty target state")
    present = [bit for bit in range(width) if target & (1 << bit)]
    absent = [bit for bit in range(width) if not target & (1 << bit)]
    rng = random.Random(seed)
    if desired_answer:
        required = 1 << rng.choice(present)
        return {"kind": "proof_query", "required_mask": required, "mode": "all"}
    if absent:
        required = 1 << rng.choice(absent)
        return {"kind": "proof_query", "required_mask": required, "mode": "all"}
    first, second = rng.sample(present, k=2)
    return {
        "kind": "proof_query",
        "required_mask": (1 << first) | (1 << second),
        "mode": "parity",
    }


def build_proof_depth_programs(
    *,
    active_depths: tuple[int, ...],
    horizon: int,
    context_count: int,
    paths_per_depth: int,
    width: int,
    seed: int,
    split: str,
    proof_final: str = "action",
    proof_topologies: tuple[str, ...] = ("mixed",),
    balanced_queries: bool = False,
    endpoint_cardinality: int | None = None,
) -> list[dict[str, Any]]:
    """Build h-fixed Horn streams with an exact number of causal deductions."""
    if width not in (4, 5):
        raise ValueError("Proof-depth challenges require four or five fact bits")
    if not active_depths or any(not 0 <= depth <= width for depth in active_depths):
        raise ValueError(f"Active proof depth must lie in [0, {width}]")
    if horizon < max(active_depths):
        raise ValueError("Surface horizon cannot be shorter than active proof depth")
    if endpoint_cardinality is not None and (
        not max(active_depths) <= endpoint_cardinality <= width
    ):
        raise ValueError(
            "Endpoint cardinality must cover every depth and fit the state width"
        )
    if proof_final not in {"action", "query", "next_rule"}:
        raise ValueError("Proof-depth FINAL must be action, query, or next_rule")
    if balanced_queries and (proof_final != "query" or 0 in active_depths):
        raise ValueError(
            "Balanced proof queries require query FINAL and positive active depths"
        )
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    rows = []
    for context in contexts:
        for depth in active_depths:
            cardinality = depth if endpoint_cardinality is None else endpoint_cardinality
            masks = [
                mask for mask in range(2**width) if mask.bit_count() == cardinality
            ]
            for topology in proof_topologies:
                for path in range(paths_per_depth):
                    target = masks[(int(context["index"]) + path) % len(masks)]
                    target_bits = tuple(
                        bit for bit in range(width) if target & (1 << bit)
                    )
                    active_offset = (
                        (
                            seed
                            + int(context["index"])
                            + path
                            + 17 * depth
                        )
                        % len(target_bits)
                        if target_bits
                        else 0
                    )
                    ordered = (
                        target_bits[active_offset:] + target_bits[:active_offset]
                    )
                    bits = tuple(ordered[:depth])
                    initial_bits = tuple(
                        bit for bit in target_bits if bit not in bits
                    )
                    initial = sum(1 << bit for bit in initial_bits)
                    history = _proof_rules(
                        bits=bits,
                        initial_bits=initial_bits,
                        target_bits=target_bits,
                        width=width,
                        horizon=horizon,
                        path=path,
                        seed=seed + int(context["index"]),
                        topology=topology,
                    )
                    states = [initial]
                    for rule in history:
                        states.append(apply_rule(rule, states[-1], 2**width))
                    active_count = sum(
                        left != right for left, right in zip(states, states[1:])
                    )
                    if states[-1] != target or active_count != depth:
                        raise AssertionError(
                            "Proof-depth construction missed its target"
                        )
                    desired_answer = (int(context["index"]) + path) % 2
                    if proof_final == "action":
                        final_rule = {
                            "kind": "proof_action",
                            "mapping": list(context["final_rule"]["mapping"]),
                        }
                        final_answer_symbols = list(_proof_state_symbols(width))
                    elif proof_final == "query":
                        final_rule = (
                            _balanced_proof_query(
                                target=target,
                                desired_answer=desired_answer,
                                width=width,
                                seed=seed
                                + 101 * int(context["index"])
                                + 17 * path
                                + depth,
                            )
                            if balanced_queries
                            else _proof_final_rule(
                                context=context,
                                width=width,
                                seed=seed,
                            )
                        )
                        final_answer_symbols = ["0", "1"]
                    else:
                        final_rule = proof_next_rule(
                            state=target,
                            width=width,
                            seed=seed
                            + 101 * int(context["index"])
                            + 17 * path
                            + depth,
                            desired_index=desired_answer
                            + 2 * (path % 2),
                        )
                        final_answer_symbols = ["0", "1", "2", "3", "4"]
                    semantic = {
                        "family": f"horn_proof_to_{proof_final}",
                        "history_family": "horn_proof",
                        "final_family": f"proof_{proof_final}",
                        "format": "prose",
                        "bits": width,
                        "initial_state": initial,
                        "history": history,
                        "final_rule": final_rule,
                        "current_state": target,
                        "next_state": apply_rule(final_rule, target, 2**width),
                        "history_steps": horizon,
                        "state_path": states,
                        "path_code": path,
                        "program_context": str(context["id"]),
                        "program_context_split": split,
                        "abstraction_group": str(context["id"]),
                        "abstraction_split": split,
                        "domain": "horn_proof",
                        "composition_split": "heldout",
                        "proof_template": (
                            "endpoint_balanced_active_depth"
                            if endpoint_cardinality is not None
                            else "controlled_active_depth"
                        ),
                        "proof_topology": topology,
                        "endpoint_cardinality": cardinality,
                        "proof_composition_active": depth >= 3,
                        "active_transition_count": active_count,
                        "state_representation": "opaque_fact_set",
                        "state_symbols": list(_proof_state_symbols(width)),
                        "answer_symbols": final_answer_symbols,
                        "proof_consumer": proof_final,
                    }
                    if proof_final == "query":
                        semantic["balanced_query_target"] = desired_answer
                    digest = hashlib.sha256(
                        json.dumps(
                            semantic, sort_keys=True, separators=(",", ":")
                        ).encode()
                    ).hexdigest()[:12]
                    semantic["id"] = (
                        f"handoff_{split}_{context['id']}_h{horizon}_"
                        f"d{depth}_{topology}_p{path}_{digest}"
                    )
                    rows.append(semantic)
    return rows
