"""Deterministic semantic programs for state-handoff training and tests."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .abstraction import matched_addition_history
from .benchmark import apply_rule, hexadecimal_state_symbols


PROGRAM_DOMAINS = ("addition", "mixed_algebra", "horn_proof", "reasoning_mixture")
ALGEBRA_FAMILIES = ("add", "xor", "affine")
DEFAULT_SEEN_PAIRS = (
    ("add", "add"),
    ("xor", "xor"),
    ("affine", "affine"),
    ("add", "xor"),
    ("xor", "affine"),
    ("affine", "add"),
)
DEFAULT_HELDOUT_PAIRS = (
    ("xor", "add"),
    ("affine", "xor"),
    ("add", "affine"),
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


def _parse_pair(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        normalized = value.replace("->", "+")
        parts = tuple(part.strip() for part in normalized.split("+"))
    else:
        parts = tuple(str(part) for part in value)
    if len(parts) != 2 or any(part not in ALGEBRA_FAMILIES for part in parts):
        raise ValueError(f"Invalid algebra operation pair: {value!r}")
    return parts[0], parts[1]


def _operation_pairs(
    dataset: dict[str, Any], composition_split: str
) -> tuple[tuple[str, str], ...]:
    configured = dataset.get("operation_pairs", {}).get(composition_split)
    if configured is None:
        configured = (
            DEFAULT_SEEN_PAIRS
            if composition_split == "seen"
            else DEFAULT_HELDOUT_PAIRS
        )
    pairs = tuple(_parse_pair(value) for value in configured)
    if not pairs:
        raise ValueError(f"Operation-pair split {composition_split!r} is empty")
    return pairs


def _random_algebra_rule(
    family: str, *, modulus: int, rng: random.Random
) -> dict[str, Any]:
    if family == "add":
        return {"kind": "add", "value": rng.randrange(modulus)}
    if family == "xor":
        return {"kind": "xor", "mask": rng.randrange(modulus)}
    if family == "affine":
        return {
            "kind": "affine",
            "a": rng.choice(tuple(range(1, modulus, 2))),
            "c": rng.randrange(modulus),
        }
    raise ValueError(f"Unknown algebra family: {family!r}")


def _inverse_algebra_rule(
    rule: dict[str, Any], *, state: int, modulus: int
) -> int:
    kind = str(rule["kind"])
    if kind == "add":
        return (state - int(rule["value"])) % modulus
    if kind == "xor":
        return state ^ int(rule["mask"])
    if kind == "affine":
        inverse = pow(int(rule["a"]), -1, modulus)
        return (inverse * (state - int(rule["c"]))) % modulus
    raise ValueError(f"Algebra rule has no configured inverse: {kind!r}")


def _solve_algebra_rule(
    family: str,
    *,
    source: int,
    target: int,
    modulus: int,
    rng: random.Random,
) -> dict[str, Any]:
    if family == "add":
        return {"kind": "add", "value": (target - source) % modulus}
    if family == "xor":
        return {"kind": "xor", "mask": source ^ target}
    if family == "affine":
        multiplier = rng.choice(tuple(range(1, modulus, 2)))
        return {
            "kind": "affine",
            "a": multiplier,
            "c": (target - multiplier * source) % modulus,
        }
    raise ValueError(f"Unknown algebra family: {family!r}")


def _mixed_algebra_history(
    *,
    initial: int,
    target: int,
    path_code: int,
    history_steps: int,
    context_index: int,
    modulus: int,
    seed: int,
    dataset: dict[str, Any],
    composition_split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    if history_steps < 2 or history_steps % 2:
        raise ValueError("Mixed-algebra histories require complete two-operation blocks")
    rng = _rng(
        seed,
        "mixed_algebra",
        context_index,
        history_steps,
        target,
        path_code,
        composition_split,
    )
    pool = _operation_pairs(dataset, composition_split)
    pairs = [
        pool[(path_code + context_index + block) % len(pool)]
        for block in range(history_steps // 2)
    ]
    families = [family for pair in pairs for family in pair]
    history = [
        _random_algebra_rule(family, modulus=modulus, rng=rng)
        for family in families
    ]
    correction = rng.randrange(history_steps)
    source = initial
    for rule in history[:correction]:
        source = apply_rule(rule, source, modulus)
    required = target
    for rule in reversed(history[correction + 1 :]):
        required = _inverse_algebra_rule(rule, state=required, modulus=modulus)
    history[correction] = _solve_algebra_rule(
        families[correction],
        source=source,
        target=required,
        modulus=modulus,
        rng=rng,
    )
    return history, [f"{left}->{right}" for left, right in pairs]


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
    reserve_conjunction = (
        composition_split == "heldout" and len(target_bits) >= 2
    )
    available_essential_steps = history_steps - int(reserve_conjunction)
    minimum_initial = max(0, len(target_bits) - available_essential_steps)
    shuffled = list(target_bits)
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
            premises = sorted(rng.sample(target_bits, k=2))
            conclusion = rng.choice(target_bits)
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
    elif domain == "mixed_algebra":
        history, operation_pairs = _mixed_algebra_history(
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
        final_rule = {
            "kind": "register_dispatch",
            "mapping": list(context["final_rule"]["mapping"]),
        }
        family = "mixed_algebra_to_dispatch"
        history_family = "mixed_algebra"
        final_family = "register_dispatch"
        extra = {
            "domain": domain,
            "composition_split": composition_split,
            "operation_pairs": operation_pairs,
        }
    else:
        initial, history = _horn_history(
            target=target,
            path_code=path_code,
            history_steps=horizon,
            context_index=int(context["index"]),
            width=width,
            seed=seed,
            composition_split=composition_split,
        )
        final_rule = _proof_final_rule(context=context, width=width, seed=seed)
        family = "horn_proof_to_query"
        history_family = "horn_proof"
        final_family = "proof_query"
        extra = {
            "domain": domain,
            "composition_split": composition_split,
        }
    states = _state_path(initial, history, modulus)
    if states[-1] != target:
        raise AssertionError("Training history missed its requested state")
    if domain == "horn_proof":
        active_conjunction = any(
            len(rule.get("premises", ())) >= 2
            and all(state & (1 << int(bit)) for bit in rule["premises"])
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
    if width == 4:
        semantic.update(
            state_representation="hexadecimal",
            state_symbols=list(hexadecimal_state_symbols(modulus)),
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
