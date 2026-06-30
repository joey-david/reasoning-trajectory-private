"""Segment generated text into sentence, sentence-group, or paragraph reasoning steps."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.analysis.token_alignment import TokenSpan, token_range_for_chars
from src.analysis.token_selectors import (
    char_to_token_index,
    sentence_end_positions,
    token_count,
)


@dataclass(slots=True)
class StepSegment:
    """Describe one text span and its approximate generated-token boundaries."""

    segmenter: str
    step_idx: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str


DEFAULT_SEGMENTERS: dict[str, dict[str, Any]] = {
    "sentence": {"mode": "sentence", "group_size": 1},
    "sentence_group_2": {"mode": "sentence", "group_size": 2},
    "paragraph": {"mode": "paragraph"},
}


def configured_segmenters(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve named step segmenters from analysis configuration.

    Args:
        cfg: Analysis configuration with optional step-classification settings.

    Returns:
        Configured segmenter specifications or the default segmenter set.
    """
    step_cfg = cfg.get("step_classification", {})
    segmenters = step_cfg.get("segmenters")
    if isinstance(segmenters, dict) and segmenters:
        return {str(name): dict(spec or {}) for name, spec in segmenters.items()}
    return dict(DEFAULT_SEGMENTERS)


def build_segments(
    row: dict[str, Any],
    segmenter_name: str,
    spec: dict[str, Any],
    token_spans: list[TokenSpan] | None = None,
) -> list[StepSegment]:
    """Convert one generation's text into token-aligned step segments.

    Args:
        row: Generation row containing produced text and token IDs.
        segmenter_name: Name recorded on every resulting segment.
        spec: Segmentation mode and optional sentence group size.
        token_spans: Exact generated-token character spans when available.

    Returns:
        Non-empty text segments with approximate inclusive token boundaries.
    """
    text = row.get("produced_text", "")
    if not text or token_count(row) <= 0:
        return []

    mode = spec.get("mode", "sentence")
    if mode == "paragraph":
        spans = paragraph_spans(text)
    else:
        spans = sentence_spans(text)
        group_size = max(int(spec.get("group_size", 1)), 1)
        if group_size > 1:
            spans = grouped_spans(spans, group_size)

    segments: list[StepSegment] = []
    for idx, (start, end) in enumerate(spans):
        token_range = token_range_for_chars(token_spans or [], start, end)
        if token_range is None:
            token_start = char_to_token_index(row, start)
            token_end = char_to_token_index(row, max(end - 1, start))
        else:
            token_start, token_end = token_range
        if token_end < token_start:
            token_start, token_end = token_end, token_start
        step_text = text[start:end].strip()
        if step_text:
            segments.append(
                StepSegment(
                    segmenter=segmenter_name,
                    step_idx=idx,
                    char_start=start,
                    char_end=end,
                    token_start=token_start,
                    token_end=token_end,
                    text=step_text,
                )
            )
    return segments


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Find trimmed spans ending at non-decimal period runs.

    Args:
        text: Generated text to segment.

    Returns:
        Start-inclusive, end-exclusive character spans.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for end in sentence_end_positions(text):
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return trim_spans(text, spans)


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Find trimmed character spans separated by blank lines.

    Args:
        text: Generated text to segment.

    Returns:
        Start-inclusive, end-exclusive paragraph spans.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return trim_spans(text, spans)


def grouped_spans(
    spans: list[tuple[int, int]], group_size: int
) -> list[tuple[int, int]]:
    """Merge consecutive spans into fixed-size groups.

    Args:
        spans: Ordered character spans.
        group_size: Maximum number of adjacent spans per group.

    Returns:
        Spans covering each group from its first start to final end.
    """
    grouped: list[tuple[int, int]] = []
    for i in range(0, len(spans), group_size):
        chunk = spans[i : i + group_size]
        if chunk:
            grouped.append((chunk[0][0], chunk[-1][1]))
    return grouped


def trim_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove surrounding whitespace from character spans and drop empty ones.

    Args:
        text: Source text indexed by the spans.
        spans: Start-inclusive, end-exclusive candidate spans.

    Returns:
        Trimmed non-empty spans in input order.
    """
    out: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            out.append((start, end))
    return out
