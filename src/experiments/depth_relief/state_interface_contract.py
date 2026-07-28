"""Discrete codebooks and semantic quotients for state interfaces."""

from __future__ import annotations

import hashlib
import random
from typing import Any


DEFAULT_CODE_SYMBOLS = tuple("αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘ")
DEFAULT_CANONICAL_PERMUTATION = (5, 2, 7, 1, 6, 0, 3, 4)
DEFAULT_CANONICAL_4BIT_PERMUTATION = (
    11,
    2,
    14,
    5,
    8,
    0,
    13,
    7,
    3,
    15,
    6,
    9,
    1,
    12,
    4,
    10,
)
CODEBOOK_SIZES = {
    "canonical_opaque": 8,
    "context_bound": 8,
    "compressed_2bit": 4,
    "redundant_4bit": 16,
    "compressed_3bit": 8,
    "canonical_4bit": 16,
    "padded_5bit": 32,
    "redundant_5bit": 32,
}
INTERFACE_CONDITIONS = tuple(CODEBOOK_SIZES)
RATE_CONDITION_PREFIX = "rate_"


def is_interface_condition(condition: str) -> bool:
    """Return whether a condition names a fixed or configured rate code."""
    if condition in INTERFACE_CONDITIONS:
        return True
    if not condition.startswith(RATE_CONDITION_PREFIX):
        return False
    suffix = condition.removeprefix(RATE_CONDITION_PREFIX)
    return suffix.isdigit() and 2 <= int(suffix) <= len(DEFAULT_CODE_SYMBOLS)


def interface_codebook_size(
    condition: str, interface_config: dict[str, Any]
) -> int:
    """Return the code count declared by one interface condition."""
    if condition in CODEBOOK_SIZES:
        return CODEBOOK_SIZES[condition]
    if not is_interface_condition(condition):
        raise ValueError(f"Unknown state-interface condition: {condition!r}")
    configured = interface_config.get(condition, {}).get("codebook_size")
    parsed = int(condition.removeprefix(RATE_CONDITION_PREFIX))
    size = parsed if configured is None else int(configured)
    if size != parsed:
        raise ValueError(
            f"{condition} must declare codebook_size={parsed}, got {size}"
        )
    return size


def state_count(case: dict[str, Any]) -> int:
    """Return the number of semantic states declared by a case."""
    return 2 ** int(case["bits"])


def _validate_state_space(condition: str, case: dict[str, Any]) -> int:
    count = state_count(case)
    required = {
        "canonical_opaque": 8,
        "context_bound": 8,
        "redundant_4bit": 8,
        "compressed_3bit": 16,
        "canonical_4bit": 16,
        "padded_5bit": 16,
        "redundant_5bit": 16,
    }.get(condition)
    if required is not None and count != required:
        raise ValueError(f"{condition} requires {required} semantic states, got {count}")
    if condition == "compressed_2bit" and count < 4:
        raise ValueError("compressed_2bit requires at least four semantic states")
    if condition.startswith(RATE_CONDITION_PREFIX):
        interface_codebook_size(condition, {})
    return count


def interface_code_symbols(
    condition: str, interface_config: dict[str, Any]
) -> tuple[str, ...]:
    """Return the declared one-token alphabet for one interface condition."""
    if not is_interface_condition(condition):
        raise ValueError(f"Unknown state-interface condition: {condition!r}")
    size = interface_codebook_size(condition, interface_config)
    configured = interface_config.get(condition, {}).get("symbols")
    symbols = (
        tuple(str(value) for value in configured)
        if configured
        else DEFAULT_CODE_SYMBOLS[:size]
    )
    if len(symbols) != size or len(set(symbols)) != size:
        raise ValueError(f"{condition} requires {size} unique code symbols")
    return symbols


def _context_permutation(
    case: dict[str, Any], seed: int, count: int
) -> tuple[int, ...]:
    digest = hashlib.sha256(f"{seed}:{case['program_context']}".encode()).digest()
    values = list(range(count))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    return tuple(values)


def _configured_permutation(
    *,
    condition: str,
    interface_config: dict[str, Any],
    default: tuple[int, ...],
) -> tuple[int, ...]:
    permutation = tuple(
        int(value)
        for value in interface_config.get(condition, {}).get(
            "permutation", default
        )
    )
    if sorted(permutation) != list(range(len(default))):
        raise ValueError(f"{condition} mapping must permute 0..{len(default) - 1}")
    return permutation


def interface_code_index(
    *,
    condition: str,
    case: dict[str, Any],
    state: int,
    interface_config: dict[str, Any],
    variant: int | None = None,
) -> int:
    """Map a semantic state to a rate-controlled code index."""
    count = _validate_state_space(condition, case)
    if not 0 <= int(state) < count:
        raise ValueError(f"State {state} is outside [0, {count})")
    if condition == "canonical_opaque":
        return _configured_permutation(
            condition=condition,
            interface_config=interface_config,
            default=DEFAULT_CANONICAL_PERMUTATION,
        )[int(state)]
    if condition == "canonical_4bit":
        return _configured_permutation(
            condition=condition,
            interface_config=interface_config,
            default=DEFAULT_CANONICAL_4BIT_PERMUTATION,
        )[int(state)]
    if condition == "padded_5bit":
        return 2 * int(state)
    if condition == "context_bound":
        seed = int(interface_config.get(condition, {}).get("seed", 721_701))
        return _context_permutation(case, seed, count)[int(state)]
    if condition == "compressed_2bit":
        return int(state) % 4
    if condition == "compressed_3bit":
        return int(state) % 8
    if condition in {"redundant_4bit", "redundant_5bit"}:
        nuisance = (
            int(case.get("path_code", 0)) % 2 if variant is None else int(variant)
        )
        if nuisance not in (0, 1):
            raise ValueError("The redundant code variant must be one bit")
        return 2 * int(state) + nuisance
    if condition.startswith(RATE_CONDITION_PREFIX):
        size = interface_codebook_size(condition, interface_config)
        if size < count:
            return int(state) % size
        variants = tuple(range(int(state), size, count))
        selected = int(case.get("path_code", 0)) if variant is None else int(variant)
        return variants[selected % len(variants)]
    raise ValueError(f"Unknown state-interface condition: {condition!r}")


def semantic_states_for_code(
    *,
    condition: str,
    case: dict[str, Any],
    code_index: int,
    interface_config: dict[str, Any],
) -> tuple[int, ...]:
    """Return every semantic state compatible with one code."""
    size = interface_codebook_size(condition, interface_config)
    if not 0 <= int(code_index) < size:
        return ()
    count = _validate_state_space(condition, case)
    if condition.startswith(RATE_CONDITION_PREFIX):
        if size < count:
            return tuple(
                state for state in range(count) if state % size == int(code_index)
            )
        return (int(code_index) % count,)
    return tuple(
        state
        for state in range(count)
        if interface_code_index(
            condition=condition,
            case=case,
            state=state,
            interface_config=interface_config,
            variant=code_index % 2,
        )
        == code_index
    )
