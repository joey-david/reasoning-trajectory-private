"""Exact character-to-token alignment for controlled object-state anchors."""

from __future__ import annotations

from typing import Any


def anchor_token_range(
    tokenizer: Any,
    text: str,
    anchor_text: str,
) -> tuple[int, int]:
    """Return the inclusive token interval overlapping the unique anchor."""
    first = text.find(anchor_text)
    if first < 0:
        raise ValueError("anchor text is absent from the full text")
    if text.find(anchor_text, first + 1) >= 0:
        raise ValueError("anchor text is not unique in the full text")
    char_start = first
    char_end = first + len(anchor_text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > char_start and start < char_end
    ]
    if not indices:
        raise ValueError("anchor has no aligned tokens")
    return indices[0], indices[-1]


def validate_anchor_rows(tokenizer: Any, rows: list[dict[str, Any]]) -> None:
    """Fail early when any bank row cannot be aligned exactly."""
    for row in rows:
        anchor_token_range(tokenizer, str(row["text"]), str(row["anchor_text"]))
