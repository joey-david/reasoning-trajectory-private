"""Reconcile overlapping token-window semantic labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from src.experiments.token_segmentation.data import TraceKey


@dataclass(frozen=True, slots=True)
class SemanticSpan:
    """One globally aligned semantic interval."""

    sample_id: str
    seed: int
    token_start: int
    token_end: int
    label: str
    confidence: float
    text: str
    record_id: str

    @property
    def key(self) -> TraceKey:
        """Return the source trace key."""
        return self.sample_id, self.seed


@dataclass(slots=True)
class SemanticTrace:
    """Reconciled semantic spans and boundaries for one trace."""

    sample_id: str
    seed: int
    token_count: int
    spans: list[SemanticSpan]
    boundaries: np.ndarray

    @property
    def key(self) -> TraceKey:
        """Return the source trace key."""
        return self.sample_id, self.seed


def load_semantic_traces(
    windows_path: Path,
    labels_path: Path,
    *,
    edge_margin: int = 2,
    boundary_tolerance: int = 2,
) -> tuple[dict[TraceKey, SemanticTrace], dict[str, Any]]:
    """Load, edge-filter, deduplicate, and audit semantic annotations."""
    windows = _read_jsonl(windows_path)
    labels = _read_jsonl(labels_path)
    window_map = {str(row["record_id"]): row for row in windows}
    trace_lengths: dict[TraceKey, int] = {}
    for row in windows:
        key = str(row["sample_id"]), int(row["seed"])
        trace_lengths[key] = max(trace_lengths.get(key, 0), int(row["token_end"]))

    observations: list[SemanticSpan] = []
    edge_dropped = 0
    accepted = 0
    for label_row in labels:
        if not label_row.get("accepted"):
            continue
        accepted += 1
        record_id = str(label_row["record_id"])
        window = window_map[record_id]
        key = str(window["sample_id"]), int(window["seed"])
        window_start = int(window["token_start"])
        window_end = int(window["token_end"])
        trace_end = trace_lengths[key]
        for span in (label_row.get("silver_label") or {}).get("spans", []):
            if "token_start" not in span or "token_end" not in span:
                continue
            token_start = int(span["token_start"])
            token_end = int(span["token_end"])
            touches_internal_edge = (
                window_start > 0 and token_start <= window_start + edge_margin
            ) or (
                window_end < trace_end and token_end >= window_end - 1 - edge_margin
            )
            if touches_internal_edge:
                edge_dropped += 1
                continue
            observations.append(
                SemanticSpan(
                    sample_id=key[0],
                    seed=key[1],
                    token_start=token_start,
                    token_end=token_end,
                    label=str(span["label"]),
                    confidence=float(span["confidence"]),
                    text=str(span["text"]),
                    record_id=record_id,
                )
            )

    grouped: dict[tuple[TraceKey, int, int], list[SemanticSpan]] = defaultdict(list)
    for span in observations:
        grouped[(span.key, span.token_start, span.token_end)].append(span)
    duplicate_groups = [items for items in grouped.values() if len(items) > 1]
    reconciled = [_vote_exact_interval(items) for items in grouped.values()]

    by_trace: dict[TraceKey, list[SemanticSpan]] = defaultdict(list)
    for span in reconciled:
        by_trace[span.key].append(span)
    traces: dict[TraceKey, SemanticTrace] = {}
    for key, spans in by_trace.items():
        spans.sort(key=lambda item: (item.token_start, item.token_end))
        clusters = _boundary_clusters(spans, boundary_tolerance)
        boundaries = np.asarray(
            sorted(
                {
                    round(median(item.token_end for item in cluster))
                    for cluster in clusters
                }
            ),
            dtype=np.int32,
        )
        traces[key] = SemanticTrace(
            sample_id=key[0],
            seed=key[1],
            token_count=trace_lengths[key],
            spans=spans,
            boundaries=boundaries,
        )

    lengths = [span.token_end - span.token_start + 1 for span in reconciled]
    covered = sum(_covered_tokens(trace.spans) for trace in traces.values())
    total_tokens = sum(trace.token_count for trace in traces.values())
    audit = {
        "windows": len(windows),
        "accepted_windows": accepted,
        "rejected_windows": len(labels) - accepted,
        "raw_spans": sum(
            len((row.get("silver_label") or {}).get("spans", [])) for row in labels
        ),
        "edge_spans_dropped": edge_dropped,
        "aligned_observations": len(observations),
        "unique_exact_intervals": len(reconciled),
        "traces": len(traces),
        "duplicate_interval_groups": len(duplicate_groups),
        "duplicate_interval_label_agreement": _unanimous_rate(duplicate_groups),
        "token_coverage": covered / max(total_tokens, 1),
        "span_length_tokens": {
            "median": float(np.median(lengths)) if lengths else None,
            "mean": float(np.mean(lengths)) if lengths else None,
            "p90": float(np.quantile(lengths, 0.9)) if lengths else None,
        },
        "labels": dict(Counter(span.label for span in reconciled).most_common()),
    }
    return traces, audit


def semantic_rows(traces: dict[TraceKey, SemanticTrace]) -> list[dict[str, Any]]:
    """Flatten reconciled spans for an inspectable JSONL artifact."""
    return [
        asdict(span)
        for trace in traces.values()
        for span in trace.spans
    ]


def _vote_exact_interval(items: list[SemanticSpan]) -> SemanticSpan:
    """Choose the confidence-weighted label for one exact interval."""
    votes: dict[str, float] = defaultdict(float)
    for item in items:
        votes[item.label] += item.confidence
    label = max(votes, key=votes.get)
    candidates = [item for item in items if item.label == label]
    return max(candidates, key=lambda item: item.confidence)


def _boundary_clusters(
    spans: list[SemanticSpan],
    tolerance: int,
) -> list[list[SemanticSpan]]:
    """Cluster nearby end-token observations without merging distant edits."""
    ordered = sorted(spans, key=lambda item: item.token_end)
    clusters: list[list[SemanticSpan]] = []
    for span in ordered:
        if not clusters or span.token_end - clusters[-1][-1].token_end > tolerance:
            clusters.append([span])
        else:
            clusters[-1].append(span)
    return clusters


def _covered_tokens(spans: list[SemanticSpan]) -> int:
    """Count the union of annotated token positions."""
    intervals = sorted((span.token_start, span.token_end + 1) for span in spans)
    covered = 0
    current_start = current_end = -1
    for start, end in intervals:
        if start > current_end:
            covered += max(0, current_end - current_start)
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return covered + max(0, current_end - current_start)


def _unanimous_rate(groups: list[list[SemanticSpan]]) -> float | None:
    """Return the share of repeated observations with one unanimous label."""
    if not groups:
        return None
    return float(
        np.mean([len({item.label for item in group}) == 1 for group in groups])
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact into dictionaries."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
