"""Build token-index selectors for regular intervals, text boundaries, and regex regions."""

from __future__ import annotations

import re
from typing import Any, Callable


Selector = Callable[[dict[str, Any]], list[int]]


def build_token_selector(spec: dict[str, Any] | None) -> Selector:
    """Compile a selector specification into a generation-row callback.

    Args:
        spec: Selector mode and mode-specific options, or ``None`` for every token.

    Returns:
        A callable that maps a generation row to valid generated-token indices.
    """
    spec = spec or {"every_n": 1}
    mode = spec.get("mode")

    if mode == "sentence_end" or spec.get("sentence_end"):
        return lambda row: sentence_end_tokens(row)

    if mode == "percentiles" or "percentiles" in spec:
        values = spec.get("percentiles")
        if values is None:
            count = max(int(spec.get("count", spec.get("P", 10))), 1)
            values = [i * 100.0 / count for i in range(count + 1)]
        return lambda row: percentile_tokens(row, values)

    if mode == "reasoning_boundaries" or spec.get("reasoning_boundaries"):
        return lambda row: reasoning_boundary_tokens(row)

    if mode == "first_last" or spec.get("first_last"):
        return lambda row: first_last_tokens(row)

    if "before_regex" in spec or "after_regex" in spec:
        pattern = spec.get("before_regex") or spec.get("after_regex")
        after = "after_regex" in spec
        return lambda row: regex_tokens(row, pattern, after)

    n = max(int(spec.get("every_n", 1)), 1)
    return lambda row: every_n_tokens(row, n)


def every_n_tokens(row: dict[str, Any], n: int) -> list[int]:
    """Select regularly spaced generated-token indices.

    Args:
        row: Generation row containing token IDs.
        n: Positive interval; values below one are treated as one.

    Returns:
        Zero-based indices from the first token at the requested interval.
    """
    return list(range(0, token_count(row), max(int(n), 1)))


def sentence_end_tokens(row: dict[str, Any]) -> list[int]:
    """Approximate generated-token indices at sentence-ending characters.

    Args:
        row: Generation row containing decoded text and token IDs.

    Returns:
        Unique valid indices corresponding proportionally to sentence ends.
    """
    text = row.get("produced_text", "")
    if not text:
        return []
    candidates = [match.end() for match in re.finditer(r"[.!?](?:\s|$)", text)]
    return unique_existing_tokens(
        row, [char_to_token_index(row, pos) for pos in candidates]
    )


def percentile_tokens(row: dict[str, Any], percentiles: list[int | float]) -> list[int]:
    """Select generated-token indices at requested trajectory percentiles.

    Args:
        row: Generation row containing token IDs.
        percentiles: Numeric positions expressed from zero to one hundred.

    Returns:
        Unique valid indices in requested order.
    """
    total = token_count(row)
    if total <= 0:
        return []
    indices = [round((float(p) / 100.0) * (total - 1)) for p in percentiles]
    return unique_existing_tokens(row, indices)


def reasoning_boundary_tokens(row: dict[str, Any]) -> list[int]:
    """Select first, reasoning-boundary, post-boundary, and final tokens.

    Args:
        row: Generation row with token IDs and optional reasoning length.

    Returns:
        Unique valid boundary indices, falling back to first and last.
    """
    total = token_count(row)
    if total <= 0:
        return []
    reasoning_length = row.get("reasoning_length")
    if reasoning_length is None:
        return first_last_tokens(row)
    boundary = min(max(int(reasoning_length) - 1, 0), total - 1)
    return unique_existing_tokens(
        row, [0, boundary, min(boundary + 1, total - 1), total - 1]
    )


def first_last_tokens(row: dict[str, Any]) -> list[int]:
    """Select the first and last generated tokens.

    Args:
        row: Generation row containing token IDs.

    Returns:
        Both endpoint indices, one index for a one-token row, or an empty list.
    """
    total = token_count(row)
    if total <= 0:
        return []
    return unique_existing_tokens(row, [0, total - 1])


def regex_tokens(row: dict[str, Any], pattern: str, after: bool) -> list[int]:
    """Select the token region before or after the first regex match boundary.

    Args:
        row: Generation row containing decoded text and token IDs.
        pattern: Regular expression searched across the complete text.
        after: Select from the match end onward when true, otherwise up to its start.

    Returns:
        Contiguous approximate token indices, or an empty list without a match.
    """
    text = row.get("produced_text", "")
    match = re.search(pattern, text, re.S)
    if not match:
        return []
    pos = match.end() if after else match.start()
    token_idx = char_to_token_index(row, pos)
    total = token_count(row)
    return list(range(token_idx, total)) if after else list(range(token_idx))


def char_to_token_index(row: dict[str, Any], char_pos: int) -> int:
    """Map a character offset proportionally onto generated-token positions.

    Args:
        row: Generation row containing decoded text and token IDs.
        char_pos: Character offset to map and clamp.

    Returns:
        A clamped approximate token index, or zero for an empty generation.
    """
    text = row.get("produced_text", "")
    total = token_count(row)
    if total <= 0:
        return 0
    if not text:
        return min(max(char_pos, 0), total - 1)
    ratio = min(max(char_pos, 0), len(text)) / max(len(text), 1)
    return min(max(round(ratio * (total - 1)), 0), total - 1)


def token_count(row: dict[str, Any]) -> int:
    """Count generated tokens in a generation row.

    Args:
        row: Generation row with optional ``generated_token_ids``.

    Returns:
        Number of generated token IDs.
    """
    return len(row.get("generated_token_ids", []))


def unique_existing_tokens(row: dict[str, Any], indices: list[int]) -> list[int]:
    """Discard duplicate and out-of-range token indices while preserving order.

    Args:
        row: Generation row used to determine the valid token range.
        indices: Candidate zero-based indices.

    Returns:
        Unique valid indices in first-occurrence order.
    """
    total = token_count(row)
    seen: set[int] = set()
    out: list[int] = []
    for idx in indices:
        if 0 <= idx < total and idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out
