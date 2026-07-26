"""Deterministic semantic programs for state-handoff training and tests."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .abstraction import matched_addition_history
from .benchmark import apply_rule, hexadecimal_state_symbols
from .state_handoff_instruction_programs import (
    mixed_algebra_history,
    primitive_algebra_history,
    register_history,
)


PROGRAM_DOMAINS = (
    "addition",
    "mixed_algebra",
    "algebra_primitives",
    "horn_proof",
    "register_machine",
    "reasoning_mixture",
)
PROOF_STATE_SYMBOLS = tuple("абвгдежзийклмноп")
PROOF_STATE_SYMBOLS_5BIT = tuple("абвгдежзийклмнопрстуфхцчшщъыьэюя")
CLOSED_HORN_TRANSITION_CLASSES = (
    "blocked_unary",
    "blocked_conjunction",
    "idempotent",
    "active_unconditional",
    "active_unary",
    "active_conjunction",
)


def _rng(seed: int, *parts: Any) -> random.Random:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _state_path(
    initial: int, history: list[dict[str, Any]], modulus: int
) -> list[int]:
    values = [initial]
    for rule in history:
        values.append(apply_rule(rule, values[-1], modulus))
    return values


def _program_contexts(
    *, split: str, count: int, width: int, seed: int
) -> list[dict[str, Any]]:
    """Preserve the original context generator and its saved hashes."""
    modulus = 2**width
    contexts = []
    signatures: set[tuple[int, tuple[int, ...]]] = set()
    candidate = 0
    while len(contexts) < count:
        rng = random.Random(
            seed + 1_000_003 * (sum(map(ord, split)) + 1) + 10_007 * candidate
        )
        candidate += 1
        initial = rng.randrange(modulus)
        mapping = list(range(modulus))
        rng.shuffle(mapping)
        signature = (initial, tuple(mapping))
        if signature in signatures:
            continue
        signatures.add(signature)
        contexts.append(
            {
                "id": f"{split}_c{len(contexts):03d}",
                "index": len(contexts),
                "initial_state": initial,
                "final_rule": {"kind": "pointer", "mapping": mapping},
            }
        )
    return contexts


def _horn_history(
    *,
    target: int,
    path_code: int,
    history_steps: int,
    context_index: int,
    width: int,
    seed: int,
    composition_split: str,
) -> tuple[int, list[dict[str, Any]]]:
    rng = _rng(
        seed,
        "horn_proof",
        context_index,
        history_steps,
        target,
        path_code,
        composition_split,
    )
    target_bits = [bit for bit in range(width) if target & (1 << bit)]
    reserve_conjunction = composition_split == "heldout" and len(target_bits) >= 3
    reserved_conclusion = (
        rng.choice(target_bits) if reserve_conjunction else None
    )
    prefix_bits = [
        bit for bit in target_bits if bit != reserved_conclusion
    ]
    available_essential_steps = history_steps - int(reserve_conjunction)
    minimum_initial = max(0, len(prefix_bits) - available_essential_steps)
    shuffled = list(prefix_bits)
    rng.shuffle(shuffled)
    initial_bits = set(shuffled[:minimum_initial])
    initial = sum(1 << bit for bit in initial_bits)
    missing = [bit for bit in shuffled if bit not in initial_bits]
    essential_positions = set(
        rng.sample(range(available_essential_steps), k=len(missing))
    )
    history = []
    state = initial
    for position in range(history_steps):
        established = [bit for bit in range(width) if state & (1 << bit)]
        if reserve_conjunction and position == history_steps - 1:
            premises = sorted(
                rng.sample(
                    [bit for bit in target_bits if bit != reserved_conclusion],
                    k=2,
                )
            )
            conclusion = int(reserved_conclusion)
        elif position in essential_positions:
            conclusion = missing.pop(0)
            if composition_split == "heldout" and len(established) >= 2:
                premise_count = 2
            elif established:
                premise_count = 1
            else:
                premise_count = 0
            premises = sorted(rng.sample(established, k=premise_count))
        else:
            absent = [bit for bit in range(width) if not target & (1 << bit)]
            if absent:
                blocked = rng.choice(absent)
                premises = [blocked]
                conclusion = blocked
            elif not established:
                blocked = rng.randrange(width)
                premises = [blocked]
                conclusion = blocked
            else:
                conclusion = rng.choice(established)
                premise_count = min(
                    len(established),
                    2 if composition_split == "heldout" else 1,
                )
                premises = sorted(rng.sample(established, k=premise_count))
        rule = {
            "kind": "horn",
            "premises": premises,
            "conclusion": conclusion,
        }
        history.append(rule)
        state = apply_rule(rule, state, 2**width)
    if state != target:
        raise AssertionError("Constructed Horn proof missed its requested fact set")
    return initial, history


def _proof_state_symbols(width: int) -> tuple[str, ...]:
    if width == 4:
        return PROOF_STATE_SYMBOLS
    if width == 5:
        return PROOF_STATE_SYMBOLS_5BIT
    raise ValueError("Opaque proof-state labels support four or five fact bits")


def _closed_horn_initial_states(
    transition_class: str, width: int
) -> tuple[int, ...]:
    modulus = 2**width
    states = []
    for state in range(modulus):
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
    cells = [
        (context, transition_class, initial)
        for context in contexts
        for transition_class in CLOSED_HORN_TRANSITION_CLASSES
        for initial in _closed_horn_initial_states(transition_class, width)
    ]
    if semantic_count < len(CLOSED_HORN_TRANSITION_CLASSES) * context_count:
        raise ValueError(
            f"{split} is too small to cover every context-transition class"
        )
    rows = []
    modulus = 2**width
    symbols = list(_proof_state_symbols(width))
    for index in range(semantic_count):
        context, transition_class, initial = cells[index % len(cells)]
        path_code = index // len(cells)
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
        final_rule, proof_answer_symbols = _proof_consumer_rule(
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


def _resolved_domain(dataset: dict[str, Any], context_index: int) -> str:
    domain = str(dataset.get("domain", "addition"))
    if domain not in PROGRAM_DOMAINS:
        raise ValueError(f"Unknown state-handoff program domain: {domain!r}")
    if domain == "reasoning_mixture":
        return ("mixed_algebra", "horn_proof")[(context_index // 2) % 2]
    return domain


def _proof_final_rule(
    *, context: dict[str, Any], width: int, seed: int
) -> dict[str, Any]:
    rng = _rng(seed, "proof_final", context["id"])
    return {
        "kind": "proof_query",
        "required_mask": rng.randrange(1, 2**width),
        "mode": ("all", "any", "parity")[int(context["index"]) % 3],
    }


def _proof_next_rule(
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


def _proof_consumer_rule(
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
    rule = _proof_next_rule(
        state=state,
        width=width,
        seed=seed + 101 * int(context["index"]) + path_code,
        desired_index=(int(context["index"]) + path_code) % 4,
    )
    return rule, ["0", "1", "2", "3", "4"]


def _program_case(
    *,
    split: str,
    context: dict[str, Any],
    horizon: int,
    target: int,
    path_code: int,
    width: int,
    seed: int,
    dataset: dict[str, Any],
    composition_split: str = "seen",
) -> dict[str, Any]:
    modulus = 2**width
    domain = _resolved_domain(dataset, int(context["index"]))
    initial = int(context["initial_state"])
    extra: dict[str, Any] = {}
    if domain == "addition":
        history = matched_addition_history(
            initial=initial,
            target=target,
            path_code=path_code,
            history_steps=horizon,
            group_index=int(context["index"]),
            modulus=modulus,
        )
        final_rule = context["final_rule"]
        family = "add_to_pointer"
        history_family = "add"
        final_family = "pointer"
    elif domain in {"mixed_algebra", "algebra_primitives"}:
        if domain == "mixed_algebra":
            history, operation_pairs = mixed_algebra_history(
                initial=initial,
                target=target,
                path_code=path_code,
                history_steps=horizon,
                context_index=int(context["index"]),
                modulus=modulus,
                seed=seed,
                dataset=dataset,
                composition_split=composition_split,
            )
        else:
            history, operation_pairs = primitive_algebra_history(
                initial=initial,
                target=target,
                path_code=path_code,
                history_steps=horizon,
                context_index=int(context["index"]),
                modulus=modulus,
                seed=seed,
                composition_split=composition_split,
            )
        final_rule = {
            "kind": "register_dispatch",
            "mapping": list(context["final_rule"]["mapping"]),
        }
        family = f"{domain}_to_dispatch"
        history_family = domain
        final_family = "register_dispatch"
        extra = {
            "domain": domain,
            "composition_split": composition_split,
            "operation_pairs": operation_pairs,
        }
    elif domain == "horn_proof":
        initial, history = _horn_history(
            target=target,
            path_code=path_code,
            history_steps=horizon,
            context_index=int(context["index"]),
            width=width,
            seed=seed,
            composition_split=composition_split,
        )
        if str(dataset.get("proof_final", "query")) == "action":
            final_rule = {
                "kind": "proof_action",
                "mapping": list(context["final_rule"]["mapping"]),
            }
            proof_answer_symbols = list(_proof_state_symbols(width))
            proof_consumer = "action"
        else:
            final_rule, proof_answer_symbols = _proof_consumer_rule(
                context=context,
                state=target,
                width=width,
                seed=seed,
                dataset=dataset,
                path_code=path_code,
            )
            proof_consumer = (
                "next_rule"
                if final_rule["kind"] == "proof_next_rule"
                else "query"
            )
        family = f"horn_proof_to_{final_rule['kind']}"
        history_family = "horn_proof"
        final_family = str(final_rule["kind"])
        extra = {
            "domain": domain,
            "composition_split": composition_split,
            "answer_symbols": proof_answer_symbols,
            "proof_consumer": proof_consumer,
        }
    else:
        if width != 4:
            raise ValueError("Register-machine programs require four-bit state")
        initial, history, instruction_families = register_history(
            target=target,
            path_code=path_code,
            history_steps=horizon,
            context_index=int(context["index"]),
            seed=seed,
            composition_split=composition_split,
        )
        final_rule = {
            "kind": "register_dispatch",
            "mapping": list(context["final_rule"]["mapping"]),
        }
        family = "register_machine_to_dispatch"
        history_family = "register_machine"
        final_family = "register_dispatch"
        extra = {
            "domain": domain,
            "composition_split": composition_split,
            "instruction_families": instruction_families,
        }
    states = _state_path(initial, history, modulus)
    if states[-1] != target:
        raise AssertionError("Training history missed its requested state")
    if domain == "horn_proof":
        active_conjunction = any(
            len(rule.get("premises", ())) >= 2
            and all(state & (1 << int(bit)) for bit in rule["premises"])
            and apply_rule(rule, state, modulus) != state
            for state, rule in zip(states, history)
        )
        extra.update(
            proof_template=(
                "active_conjunction"
                if active_conjunction
                else "single_premise"
            ),
            proof_composition_active=active_conjunction,
        )
        if final_rule["kind"] == "proof_query":
            extra["answer_symbols"] = ["0", "1"]
    semantic = {
        "family": family,
        "history_family": history_family,
        "final_family": final_family,
        "format": "prose",
        "bits": width,
        "initial_state": initial,
        "history": history,
        "final_rule": final_rule,
        "current_state": target,
        "next_state": apply_rule(final_rule, target, modulus),
        "history_steps": horizon,
        "state_path": states,
        "path_code": path_code,
        "program_context": str(context["id"]),
        "program_context_split": split,
        "abstraction_group": str(context["id"]),
        "abstraction_split": split,
        **extra,
    }
    if domain == "horn_proof" and width in (4, 5):
        semantic.update(
            state_representation="opaque_fact_set",
            state_symbols=list(_proof_state_symbols(width)),
        )
    elif width == 4:
        configured_symbols = dataset.get("state_symbols")
        symbols = (
            [str(value) for value in configured_symbols]
            if configured_symbols is not None
            else list(hexadecimal_state_symbols(modulus))
        )
        if len(symbols) != modulus or len(set(symbols)) != modulus:
            raise ValueError("A four-bit state alphabet needs 16 unique symbols")
        semantic.update(
            state_representation=str(
                dataset.get("state_representation", "hexadecimal")
            ),
            state_symbols=symbols,
        )
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    semantic["id"] = (
        f"handoff_{split}_{context['id']}_h{horizon}_s{target}_"
        f"p{path_code}_{digest}"
    )
    return semantic


def _balanced_programs(
    *,
    split: str,
    semantic_count: int,
    horizons: tuple[int, ...],
    context_count: int,
    width: int,
    seed: int,
    dataset: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dataset = dataset or {}
    if semantic_count < context_count * len(horizons) * 2**width:
        raise ValueError(f"{split} is too small to cover every context-state cell")
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    cells = [
        (context, horizon, target)
        for context in contexts
        for horizon in horizons
        for target in range(2**width)
    ]
    rows = []
    for index in range(semantic_count):
        context, horizon, target = cells[index % len(cells)]
        path_code = index // len(cells)
        rows.append(
            _program_case(
                split=split,
                context=context,
                horizon=horizon,
                target=target,
                path_code=path_code,
                width=width,
                seed=seed,
                dataset=dataset,
            )
        )
    return rows


def build_test_programs(
    *,
    horizons: tuple[int, ...],
    context_count: int,
    paths_per_state: int,
    width: int,
    seed: int,
    split: str = "test",
    dataset: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a balanced fixed test bank for any domain and artifact split."""
    dataset = dataset or {}
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    domain = str(dataset.get("domain", "addition"))
    composition_splits = (
        tuple(str(value) for value in dataset.get(
            "test_composition_splits", ("seen", "heldout")
        ))
        if domain != "addition"
        else ("seen",)
    )
    if not composition_splits or any(
        value not in {"seen", "heldout"} for value in composition_splits
    ):
        raise ValueError("Test composition splits must contain seen and/or heldout")
    return [
        _program_case(
            split=split,
            context=context,
            horizon=horizon,
            target=target,
            path_code=path,
            width=width,
            seed=seed,
            dataset=dataset,
            composition_split=composition_split,
        )
        for context in contexts
        for horizon in horizons
        for target in range(2**width)
        for path in range(paths_per_state)
        for composition_split in composition_splits
    ]
