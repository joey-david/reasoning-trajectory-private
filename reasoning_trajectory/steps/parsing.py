"""Structured reasoning-step parsing and hidden-state pooling."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np


MARKER = re.compile(
    r"(?im)^\s*(?:"
    r"Step\s+\d+\s*:?"
    r"|[0-9]+[.)]\s*"
    r"|[-*]\s+"
    r"|(?:intro|apply|rw|rewrite|simp|exact|cases|induction|constructor|have)\b"
    r"|(?:def|class|return|if|for|while|assert)\b"
    r"|#\s*edit:"
    r")"
)


@dataclass(frozen=True, slots=True)
class StructuredSpan:
    """One structured text interval and its coarse syntax labels."""

    char_start: int
    char_end: int
    text: str
    labels: tuple[str, ...]


def parse_structured_spans(text: str) -> list[StructuredSpan]:
    """Split numbered, bulleted, proof, or code reasoning at line markers."""
    matches = list(MARKER.finditer(text))
    if not matches:
        return [
            StructuredSpan(start, end, text[start:end].strip(), ("newline",))
            for start, end in _nonempty_lines(text)
        ]
    spans: list[StructuredSpan] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            spans.append(
                StructuredSpan(start, end, chunk, _labels(match.group(0)))
            )
    return spans


def pool_token_states(
    token_states: np.ndarray,
    token_ranges: Sequence[tuple[int, int]],
    *,
    pooling: str = "mean",
    attention_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Pool `[tokens, layers, hidden]` states over inclusive token ranges."""
    states = np.asarray(token_states)
    if states.ndim != 3:
        raise ValueError(f"expected [tokens,layers,hidden], got {states.shape}")
    pooled = [
        _pool_slice(
            states[max(0, start) : min(end + 1, len(states))],
            pooling,
            None
            if attention_weights is None
            else attention_weights[max(0, start) : min(end + 1, len(states))],
        )
        for start, end in token_ranges
    ]
    return np.stack(pooled) if pooled else np.empty((0, *states.shape[1:]))


def _pool_slice(
    values: np.ndarray,
    pooling: str,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Apply one supported pooling rule to a non-empty token slice."""
    if not len(values):
        raise ValueError("cannot pool an empty token range")
    if pooling == "mean":
        return values.mean(axis=0)
    if pooling == "last":
        return values[-1]
    if pooling == "max":
        return values.max(axis=0)
    if pooling == "attention":
        if weights is None:
            raise ValueError("attention pooling requires attention_weights")
        normalized = weights / max(float(weights.sum()), 1e-12)
        return np.einsum("t,tlh->lh", normalized, values)
    raise ValueError(f"unknown pooling: {pooling}")


def _nonempty_lines(text: str) -> list[tuple[int, int]]:
    """Return character spans for non-empty lines."""
    return [
        match.span()
        for match in re.finditer(r"(?m)^.+$", text)
        if match.group(0).strip()
    ]


def _labels(marker: str) -> tuple[str, ...]:
    """Classify a structural marker without interpreting its semantics."""
    value = marker.strip().lower()
    if value.startswith("step") or re.match(r"^\d+[.)]", value):
        return ("numbered",)
    if re.match(
        r"^(intro|apply|rw|rewrite|simp|exact|cases|induction|constructor|have)\b",
        value,
    ):
        return ("proof_tactic",)
    if value.startswith("# edit:") or re.match(
        r"^(def|class|return|if|for|while|assert)\b", value
    ):
        return ("code_edit",)
    return ("bullet",)
