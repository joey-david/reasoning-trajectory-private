"""Shared prompt spans and deterministic splits for causal reasoning cases."""

from __future__ import annotations

from typing import Any


def split(index: int, count: int) -> str:
    """Assign one deterministic 50/25/25 train, validation, or test split."""
    fraction = index / count
    if fraction < 0.5:
        return "train"
    if fraction < 0.75:
        return "validation"
    return "test"


def experiment_row(
    *,
    experiment: str,
    index: int,
    count: int,
    prompts: dict[str, dict[str, Any]],
    evaluations: list[dict[str, Any]],
    representation_pairs: list[dict[str, str]],
    labels: dict[str, Any],
    candidates: list[str] | None = None,
    feature_prompt: str | None = None,
) -> dict[str, Any]:
    """Build the common deterministic case envelope."""
    row = {
        "schema_version": 2,
        "id": f"{experiment}_{index:04d}",
        "experiment": experiment,
        "group": f"{experiment}_case_{index:04d}",
        "split": split(index, count),
        "candidate_symbols": candidates
        or [str(value) for value in range(10)],
        "prompts": prompts,
        "evaluations": evaluations,
        "representation_pairs": representation_pairs,
        "labels": labels,
    }
    if feature_prompt is not None:
        row["feature_prompt"] = feature_prompt
    return row


def marker_prompt(
    lines: list[str], marker: str = "<<STATE>>"
) -> dict[str, Any]:
    """Join lines and mark the final exact occurrence of a visible boundary."""
    text = "\n".join(lines)
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(f"Prompt lacks checkpoint marker {marker!r}")
    return {
        "text": text,
        "checkpoint_start": start,
        "checkpoint_end": start + len(marker),
    }


def value_prompt(prefix: str, value: str | int, suffix: str) -> dict[str, Any]:
    """Mark the exact inserted value without adding a synthetic marker token."""
    rendered = str(value)
    text = prefix + rendered + suffix
    return {
        "text": text,
        "checkpoint_start": len(prefix),
        "checkpoint_end": len(prefix) + len(rendered),
    }


def delimited_span_prompt(
    text: str,
    *,
    start_delimiter: str = "[TRACE START]",
    end_delimiter: str = "[TRACE END]",
) -> dict[str, Any]:
    """Mark all text strictly between two visible trace delimiters."""
    start = text.index(start_delimiter) + len(start_delimiter)
    end = text.index(end_delimiter, start)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return {
        "text": text,
        "checkpoint_start": start,
        "checkpoint_end": end,
    }
