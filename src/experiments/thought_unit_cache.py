"""Build and load compact sentence-lattice feature caches."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.analysis.step_classification.segmentation import build_segments
from src.analysis.token_alignment import build_token_spans
from src.experiments.common import balanced_generation_rows
from src.experiments.symbolic import extract_symbolic_updates
from src.experiments.thought_unit_features import (
    accumulated_gram_spectra,
    cross_rollout_answer_scores,
    fit_sentence_pca,
    load_h4_projection,
    normalize_rows,
    question_split,
    raw_geometry,
    sentence_means,
    sentence_update_counts,
    terminal_answer_sentence,
)
from src.experiments.thought_unit_types import TraceSpec, TraceView
from src.runtime.artifact_store import load_hidden_states_npz
from src.runtime.data import write_jsonl

_LIST_MARKER_RE = re.compile(r"^(?:\d+|[A-Za-z])[.)]$")


def build_feature_cache(
    run_path: Path,
    out_dir: Path,
    *,
    projection_path: Path | None,
    per_sample: int,
    pca_dim: int,
    gram_dim: int,
) -> None:
    """Stream activations into compact sentence-level projection features.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        out_dir: Directory in which to write the results.
        projection_path: Path to a saved projection artifact.
        per_sample: Maximum number of trajectories retained per sample.
        pca_dim: Maximum PCA output dimension.
        gram_dim: Number of Gram-spectrum dimensions to retain.

    Returns:
        None.
    """
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    token_spans = build_token_spans(run_path, rows)
    train_ids = question_split(rows)
    specs: list[TraceSpec] = []
    for row, spans in zip(rows, token_spans):
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        updates = extract_symbolic_updates(
            str(row.get("produced_text", "")),
            spans,
            token_count=len(row.get("generated_token_ids", [])),
        )
        if len(segments) >= 3:
            specs.append(
                TraceSpec(
                    row=row,
                    segments=segments,
                    updates=updates,
                    train=str(row["sample_id"]) in train_ids,
                )
            )
    pca = fit_sentence_pca(run_path, specs, pca_dim=pca_dim)
    h4_weight, resolved_projection = load_h4_projection(
        run_path, projection_path=projection_path
    )

    raw_rows: list[np.ndarray] = []
    pca_rows: list[np.ndarray] = []
    h4_rows: list[np.ndarray] = []
    gram_rows: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    update_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    offsets = [0]
    parser_counts: Counter[str] = Counter()

    for spec in specs:
        states, layers = load_hidden_states_npz(
            run_path / spec.row["hidden_states_file"]
        )
        if -1 not in layers:
            raise ValueError(f"Last layer missing for {spec.row['sample_id']}")
        layer_states = states[:, layers.index(-1)].astype(np.float32)
        means, token_counts = sentence_means(layer_states, spec.segments)
        pca_means = pca.transform(means).astype(np.float32)
        h4_means = normalize_rows(means @ h4_weight.T)
        gram = accumulated_gram_spectra(
            layer_states,
            spec.segments,
            pca,
            dimension=gram_dim,
        )
        geometry = raw_geometry(means)
        answer_index = terminal_answer_sentence(spec)
        update_counts = sentence_update_counts(spec.segments, spec.updates)

        raw_rows.append(means.astype(np.float16))
        pca_rows.append(pca_means.astype(np.float16))
        h4_rows.append(h4_means.astype(np.float16))
        gram_rows.append(gram.astype(np.float16))
        geometry_rows.append(geometry.astype(np.float32))
        update_rows.append(update_counts.astype(np.int16))
        token_rows.append(token_counts.astype(np.int16))
        offsets.append(offsets[-1] + len(means))

        fragment_count = int(np.sum(token_counts <= 2))
        marker_count = sum(
            bool(_LIST_MARKER_RE.fullmatch(segment.text.strip()))
            for segment in spec.segments
        )
        parser_counts.update(
            {
                "sentences": len(means),
                "short_fragments": fragment_count,
                "list_markers": marker_count,
            }
        )
        records.append(
            {
                "sample_id": str(spec.row["sample_id"]),
                "seed": int(spec.row["seed"]),
                "is_correct": bool(spec.row["is_correct"]),
                "train": spec.train,
                "sentences": len(means),
                "updates": int(update_counts.sum()),
                "answer_sentence": answer_index,
                "short_fragments": fragment_count,
                "list_markers": marker_count,
            }
        )

    answer_rows, answer_target_counts = cross_rollout_answer_scores(raw_rows, records)
    np.savez_compressed(
        out_dir / "features.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        raw=np.concatenate(raw_rows).astype(np.float16),
        pca=np.concatenate(pca_rows).astype(np.float16),
        h4=np.concatenate(h4_rows).astype(np.float16),
        gram=np.concatenate(gram_rows).astype(np.float16),
        raw_geometry=np.concatenate(geometry_rows).astype(np.float32),
        answer_score=np.concatenate(answer_rows).astype(np.float32),
        update_count=np.concatenate(update_rows).astype(np.int16),
        token_count=np.concatenate(token_rows).astype(np.int16),
    )
    write_jsonl(out_dir / "traces.jsonl", records)
    metadata = {
        "source_run": run_path.as_posix(),
        "layer": -1,
        "pca": {
            "dimensions": int(pca.n_components_),
            "fit": "at most 24 evenly spaced sentences per training trajectory",
            "questions_disjoint": True,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        },
        "gram": {
            "dimensions": gram_dim,
            "definition": (
                "top eigenvalues of the accumulated token Gram matrix after "
                "training-only PCA whitening"
            ),
        },
        "h4_projection": resolved_projection.as_posix(),
        "parser_audit": {
            **dict(parser_counts),
            "short_fragment_rate": parser_counts["short_fragments"]
            / max(parser_counts["sentences"], 1),
            "list_marker_rate": parser_counts["list_markers"]
            / max(parser_counts["sentences"], 1),
            "policy": (
                "Preserve the repository sentence parser exactly; fragments are "
                "reported rather than silently merged."
            ),
        },
        "answer_targets": {
            **answer_target_counts,
            "policy": (
                "Prefer a different correct rollout of the same question; use "
                "the trace itself only when no correct donor exists."
            ),
        },
    }
    (out_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def load_feature_cache(feature_path: Path, trace_path: Path) -> dict[str, Any]:
    """Load compact arrays and trace metadata into memory.

    Args:
        feature_path: Path to the cached feature arrays.
        trace_path: Path to the cached trace metadata.

    Returns:
        The resulting keyed records or metrics.
    """
    with np.load(feature_path) as data:
        cache = {key: data[key].copy() for key in data.files}
    with trace_path.open(encoding="utf-8") as handle:
        cache["records"] = [json.loads(line) for line in handle if line.strip()]
    return cache


def load_partitions(path: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Load persisted primary partitions keyed by trajectory identity.

    Args:
        path: Filesystem path to read from or write to.

    Returns:
        The resulting keyed records or metrics.
    """
    output: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output[(str(row["sample_id"]), int(row["seed"]))] = {
                method: np.asarray(boundaries, dtype=np.int32)
                for method, boundaries in row["methods"].items()
            }
    return output


def trace_view(cache: dict[str, Any], index: int) -> TraceView:
    """Slice concatenated cache arrays for one trajectory.

    Args:
        cache: Cached arrays or records used by the computation.
        index: Trace index in the feature cache.

    Returns:
        A view of one cached trace and its aligned sentence features.
    """
    start, end = cache["offsets"][index : index + 2]
    record = cache["records"][index]
    section = slice(int(start), int(end))
    return TraceView(
        sample_id=str(record["sample_id"]),
        seed=int(record["seed"]),
        is_correct=bool(record["is_correct"]),
        train=bool(record["train"]),
        raw=cache["raw"][section].astype(np.float32),
        pca=cache["pca"][section].astype(np.float32),
        h4=cache["h4"][section].astype(np.float32),
        gram=cache["gram"][section].astype(np.float32),
        raw_geometry=cache["raw_geometry"][section].astype(np.float32),
        answer_score=cache["answer_score"][section].astype(np.float32),
        update_count=cache["update_count"][section].astype(np.float32),
        token_count=cache["token_count"][section].astype(np.float32),
    )


def evenly_select_traces(records: list[dict[str, Any]], limit: int) -> list[int]:
    """Retain train and test traces across as many questions as possible.

    Args:
        records: Aligned records to analyze or annotate.
        limit: Maximum number of traces to retain.

    Returns:
        The resulting ordered records or values.
    """
    by_question: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_question[str(record["sample_id"])].append(index)
    selected: list[int] = []
    cursor = 0
    questions = sorted(by_question)
    while len(selected) < limit:
        added = False
        for question in questions:
            rows = by_question[question]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        cursor += 1
    return sorted(selected)
