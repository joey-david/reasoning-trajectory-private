from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.analysis.token_selectors import char_to_token_index, token_count


@dataclass(slots=True)
class StepSegment:
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
    step_cfg = cfg.get("step_classification", {})
    segmenters = step_cfg.get("segmenters")
    if isinstance(segmenters, dict) and segmenters:
        return {str(name): dict(spec or {}) for name, spec in segmenters.items()}
    return dict(DEFAULT_SEGMENTERS)


def build_segments(
    row: dict[str, Any],
    segmenter_name: str,
    spec: dict[str, Any],
) -> list[StepSegment]:
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
        token_start = char_to_token_index(row, start)
        token_end = char_to_token_index(row, max(end - 1, start))
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
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?](?:\s+|$)", text):
        end = match.end()
        if end > start:
            spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return trim_spans(text, spans)


def paragraph_spans(text: str) -> list[tuple[int, int]]:
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


def grouped_spans(spans: list[tuple[int, int]], group_size: int) -> list[tuple[int, int]]:
    grouped: list[tuple[int, int]] = []
    for i in range(0, len(spans), group_size):
        chunk = spans[i : i + group_size]
        if chunk:
            grouped.append((chunk[0][0], chunk[-1][1]))
    return grouped


def trim_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            out.append((start, end))
    return out

