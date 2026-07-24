"""Small semantic program banks that isolate effective reasoning depth."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .benchmark import apply_rule
from .state_handoff_programs import (
    PROOF_STATE_SYMBOLS,
    _program_contexts,
    _proof_final_rule,
)


def _proof_rules(
    *,
    bits: tuple[int, ...],
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
    established: list[int] = []
    absent = [bit for bit in range(4) if bit not in bits]
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
) -> list[dict[str, Any]]:
    """Build h-fixed Horn streams with an exact number of causal deductions."""
    if width != 4:
        raise ValueError("Proof-depth challenges currently require four fact bits")
    if not active_depths or any(not 0 <= depth <= width for depth in active_depths):
        raise ValueError("Active proof depth must lie in [0, 4]")
    if horizon < max(active_depths):
        raise ValueError("Surface horizon cannot be shorter than active proof depth")
    if proof_final not in {"action", "query"}:
        raise ValueError("Proof-depth FINAL must be action or query")
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
            masks = [mask for mask in range(2**width) if mask.bit_count() == depth]
            for topology in proof_topologies:
                for path in range(paths_per_depth):
                    target = masks[(int(context["index"]) + path) % len(masks)]
                    bits = tuple(bit for bit in range(width) if target & (1 << bit))
                    history = _proof_rules(
                        bits=bits,
                        horizon=horizon,
                        path=path,
                        seed=seed + int(context["index"]),
                        topology=topology,
                    )
                    states = [0]
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
                    final_rule = (
                        {
                            "kind": "proof_action",
                            "mapping": list(context["final_rule"]["mapping"]),
                        }
                        if proof_final == "action"
                        else (
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
                    )
                    semantic = {
                        "family": f"horn_proof_to_{proof_final}",
                        "history_family": "horn_proof",
                        "final_family": f"proof_{proof_final}",
                        "format": "prose",
                        "bits": width,
                        "initial_state": 0,
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
                        "proof_template": "controlled_active_depth",
                        "proof_topology": topology,
                        "proof_composition_active": depth >= 3,
                        "active_transition_count": active_count,
                        "state_representation": "opaque_fact_set",
                        "state_symbols": list(PROOF_STATE_SYMBOLS),
                    }
                    if proof_final == "query":
                        semantic["answer_symbols"] = ["0", "1"]
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
