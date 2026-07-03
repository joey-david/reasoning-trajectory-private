"""Streaming access to token activations and token-aligned annotations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.step_classification.segmentation import sentence_spans
from src.analysis.token_alignment import build_token_spans, token_range_for_chars
from src.experiments.common import balanced_generation_rows
from src.runtime.artifact_store import load_hidden_states_npz


TraceKey = tuple[str, int]


@dataclass(slots=True)
class TraceMeta:
    """Metadata needed to score candidate token boundaries."""

    sample_id: str
    seed: int
    is_correct: bool
    hidden_states_file: str
    token_count: int
    text: str
    object_boundaries: np.ndarray
    sentence_boundaries: np.ndarray
    token_char_ends: np.ndarray
    train: bool

    @property
    def key(self) -> TraceKey:
        """Return the stable trace identifier."""
        return self.sample_id, self.seed


def prepare_traces(
    run_path: Path,
    updates_path: Path,
    *,
    per_sample: int = 10,
) -> list[TraceMeta]:
    """Build lightweight token-aligned metadata without retaining activations."""
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    update_map = load_object_boundaries(updates_path)
    test_ids = held_out_questions(rows)
    token_spans = build_token_spans(run_path, rows)
    traces: list[TraceMeta] = []
    for row, spans in zip(rows, token_spans, strict=True):
        token_count = len(row.get("generated_token_ids", []))
        if token_count < 3:
            continue
        valid_last = token_count - 2
        sentence_ends: list[int] = []
        for char_start, char_end in sentence_spans(str(row.get("produced_text", ""))):
            token_range = token_range_for_chars(spans, char_start, char_end)
            if token_range is None or token_range[1] > valid_last:
                continue
            sentence_ends.append(token_range[1])
        key = (str(row["sample_id"]), int(row["seed"]))
        object_boundaries = [
            boundary
            for boundary in update_map.get(key, ())
            if 0 <= boundary <= valid_last
        ]
        char_ends = np.asarray(
            [span[1] if span is not None else -1 for span in spans], dtype=np.int32
        )
        traces.append(
            TraceMeta(
                sample_id=key[0],
                seed=key[1],
                is_correct=bool(row["is_correct"]),
                hidden_states_file=str(row["hidden_states_file"]),
                token_count=token_count,
                text=str(row.get("produced_text", "")),
                object_boundaries=np.asarray(
                    sorted(set(object_boundaries)), dtype=np.int32
                ),
                sentence_boundaries=np.asarray(
                    sorted(set(sentence_ends)), dtype=np.int32
                ),
                token_char_ends=char_ends,
                train=key[0] not in test_ids,
            )
        )
    return traces


def load_states(run_path: Path, trace: TraceMeta, layer: int = -1) -> np.ndarray:
    """Load one trace's selected hidden-state layer as float32."""
    states, layers = load_hidden_states_npz(run_path / trace.hidden_states_file)
    try:
        layer_index = layers.index(layer)
    except ValueError as error:
        raise ValueError(f"Layer {layer} absent from {trace.hidden_states_file}") from error
    count = min(trace.token_count, states.shape[0])
    return np.asarray(states[:count, layer_index], dtype=np.float32)


def load_gold_targets(gold_run: Path, layer: int = -1) -> dict[str, np.ndarray]:
    """Load one mean teacher-forced gold-solution state per question."""
    manifest = gold_run / "gold_answers" / "manifest.jsonl"
    targets: dict[str, np.ndarray] = {}
    if not manifest.exists():
        return targets
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            states, layers = load_hidden_states_npz(
                gold_run / str(row["hidden_states_file"])
            )
            if layer not in layers or not len(states):
                continue
            targets[str(row["sample_id"])] = np.asarray(
                states[:, layers.index(layer)].mean(axis=0), dtype=np.float32
            )
    return targets


def load_object_boundaries(path: Path) -> dict[TraceKey, list[int]]:
    """Load verified symbolic-update completion tokens from H2 output."""
    grouped: dict[TraceKey, list[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["sample_id"]), int(row["seed"])
            grouped.setdefault(key, []).append(int(row["token_end"]))
    return grouped


def held_out_questions(rows: list[dict[str, Any]]) -> set[str]:
    """Select every fifth sorted question for a deterministic disjoint test set."""
    questions = sorted({str(row["sample_id"]) for row in rows})
    return set(questions[::5])


def boundary_snippet(trace: TraceMeta, boundary: int, radius: int = 90) -> str:
    """Return text around a token boundary when exact character alignment exists."""
    if boundary < 0 or boundary >= len(trace.token_char_ends):
        return ""
    offset = int(trace.token_char_ends[boundary])
    if offset < 0:
        return ""
    return trace.text[max(0, offset - radius) : min(len(trace.text), offset + radius)]
