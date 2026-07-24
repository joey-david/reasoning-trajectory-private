"""Deterministic algebra and two-register instruction histories."""

from __future__ import annotations

import hashlib
import random
from typing import Any

from .benchmark import apply_rule


ALGEBRA_FAMILIES = ("add", "xor", "affine")
REGISTER_FAMILIES = ("add", "xor", "swap", "conditional")
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


def mixed_algebra_history(
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
    """Build histories from configured two-operation composition cells."""
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
        required = _inverse_algebra_rule(
            rule, state=required, modulus=modulus
        )
    history[correction] = _solve_algebra_rule(
        families[correction],
        source=source,
        target=required,
        modulus=modulus,
        rng=rng,
    )
    return history, [f"{left}->{right}" for left, right in pairs]


def primitive_algebra_history(
    *,
    initial: int,
    target: int,
    path_code: int,
    history_steps: int,
    context_index: int,
    modulus: int,
    seed: int,
    composition_split: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Expose every primitive at h1, then test unseen operation orders."""
    if history_steps < 1:
        raise ValueError("Primitive-algebra histories must be nonempty")
    rng = _rng(
        seed,
        "algebra_primitives",
        context_index,
        history_steps,
        target,
        path_code,
        composition_split,
    )
    offset = path_code + context_index + target
    if history_steps == 1:
        families = [ALGEBRA_FAMILIES[offset % len(ALGEBRA_FAMILIES)]]
    elif composition_split == "seen":
        families = [
            ALGEBRA_FAMILIES[offset % len(ALGEBRA_FAMILIES)]
        ] * history_steps
    else:
        families = [
            ALGEBRA_FAMILIES[(offset + position) % len(ALGEBRA_FAMILIES)]
            for position in range(history_steps)
        ]
    history = [
        _random_algebra_rule(family, modulus=modulus, rng=rng)
        for family in families
    ]
    state = initial
    for rule in history[:-1]:
        state = apply_rule(rule, state, modulus)
    history[-1] = _solve_algebra_rule(
        families[-1],
        source=state,
        target=target,
        modulus=modulus,
        rng=rng,
    )
    orders = [
        f"{left}->{right}" for left, right in zip(families, families[1:])
    ]
    return history, orders


def _register_rule(family: str, rng: random.Random) -> dict[str, Any]:
    if family == "add":
        return {
            "kind": "register_add",
            "register": rng.randrange(2),
            "value": rng.randrange(1, 4),
        }
    if family == "xor":
        return {
            "kind": "register_xor",
            "register": rng.randrange(2),
            "mask": rng.randrange(1, 4),
        }
    if family == "swap":
        return {"kind": "register_swap"}
    if family == "conditional":
        return {
            "kind": "register_cond_add",
            "source": rng.randrange(2),
            "equals": rng.randrange(4),
            "value": rng.randrange(1, 4),
        }
    raise ValueError(f"Unknown register instruction family: {family!r}")


def _inverse_register_rule(rule: dict[str, Any]) -> dict[str, Any]:
    inverse = dict(rule)
    if rule["kind"] in {"register_add", "register_cond_add"}:
        inverse["value"] = (-int(rule["value"])) % 4
    return inverse


def register_history(
    *,
    target: int,
    path_code: int,
    history_steps: int,
    context_index: int,
    seed: int,
    composition_split: str,
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Build an exactly balanced two-register program from invertible steps."""
    if history_steps < 1:
        raise ValueError("Register-machine histories must be nonempty")
    rng = _rng(
        seed,
        "register_machine",
        context_index,
        history_steps,
        target,
        path_code,
        composition_split,
    )
    offset = path_code + context_index + target
    if history_steps == 1:
        families = [REGISTER_FAMILIES[offset % len(REGISTER_FAMILIES)]]
    elif composition_split == "seen":
        families = [
            REGISTER_FAMILIES[offset % len(REGISTER_FAMILIES)]
        ] * history_steps
    else:
        families = [
            REGISTER_FAMILIES[(offset + position) % len(REGISTER_FAMILIES)]
            for position in range(history_steps)
        ]
    history = [_register_rule(family, rng) for family in families]
    initial = target
    for rule in reversed(history):
        initial = apply_rule(_inverse_register_rule(rule), initial, 16)
    return initial, history, families
