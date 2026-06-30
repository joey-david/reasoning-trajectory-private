"""Compare natural and prompted text boundaries with latent and symbolic events."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    pairwise_distances,
    silhouette_score,
)

from src.analysis.step_classification.segmentation import (
    paragraph_spans,
    sentence_spans,
)
from src.analysis.token_alignment import (
    TokenSpan,
    build_token_spans,
    token_range_for_chars,
)
from src.experiments.common import (
    balanced_generation_rows,
    latent_deltas,
    percentile_rank,
    robust_spike_indices,
)
from src.experiments.symbolic import extract_symbolic_updates
from src.runtime.artifact_store import load_hidden_states_npz
from src.runtime.config import load_config


def run_boundary_comparison(
    run_paths: list[Path],
    *,
    per_sample: int = 5,
    window: int = 2,
) -> Path:
    """Run H1 after the matched freeform and prompted runs are available."""
    if len(run_paths) < 2:
        raise ValueError("H1 requires one freeform and at least one prompted run")
    condition_rows: dict[
        str, tuple[Path, list[dict[str, Any]], list[list[TokenSpan]]]
    ] = {}
    matched_ids: set[str] | None = None
    for index, run_path in enumerate(run_paths):
        config = load_config(run_path)
        condition = str(
            config.get("analysis", {})
            .get("experiment", {})
            .get("condition", "freeform" if index == 0 else run_path.name)
        )
        rows = balanced_generation_rows(
            run_path,
            per_sample=per_sample,
            require_scored=False,
        )
        if index > 0 and matched_ids is None:
            matched_ids = {str(row["sample_id"]) for row in rows}
        condition_rows[condition] = (run_path, rows, build_token_spans(run_path, rows))

    if matched_ids:
        run_path, rows, spans = condition_rows["freeform"]
        keep = [
            idx for idx, row in enumerate(rows) if str(row["sample_id"]) in matched_ids
        ]
        condition_rows["freeform"] = (
            run_path,
            [rows[idx] for idx in keep],
            [spans[idx] for idx in keep],
        )

    trace_records: list[dict[str, Any]] = []
    aggregates: defaultdict[tuple[str, int, str], list[dict[str, float]]] = defaultdict(
        list
    )
    compliance: defaultdict[str, list[float]] = defaultdict(list)
    for condition, (run_path, rows, spans_by_row) in condition_rows.items():
        for row, token_spans in zip(rows, spans_by_row):
            states, layers = load_hidden_states_npz(
                run_path / row["hidden_states_file"]
            )
            count = min(len(row.get("generated_token_ids", [])), len(states))
            if count < 3:
                continue
            text = str(row.get("produced_text", ""))
            boundaries = text_boundary_indices(text, token_spans, count)
            symbolic = extract_symbolic_updates(text, token_spans, token_count=count)
            symbolic_indices = np.asarray(
                sorted(
                    {
                        min(max(update.token_end + 1, 1), count - 1)
                        for update in symbolic
                    }
                ),
                dtype=np.int32,
            )
            compliance[condition].append(format_compliance(text, condition))
            for layer_col, layer in enumerate(layers):
                layer_states = states[:count, layer_col].astype(np.float32)
                signals = latent_boundary_signals(row, layer_states, layer_col)
                magnitude_spikes = robust_spike_indices(signals["magnitude"])
                quality_sample = segment_quality_sample(layer_states)
                candidates = {
                    **{
                        name: (indices, signals["magnitude"])
                        for name, indices in boundaries.items()
                    },
                    **{
                        f"latent_{name}": (
                            robust_spike_indices(values),
                            values,
                        )
                        for name, values in signals.items()
                    },
                }
                for boundary_name, (indices, signal_values) in candidates.items():
                    metrics = boundary_metrics(
                        indices=np.asarray(indices, dtype=np.int32),
                        symbolic=symbolic_indices,
                        spikes=magnitude_spikes,
                        magnitudes=signals["magnitude"],
                        signal_values=signal_values,
                        states=layer_states,
                        quality_sample=quality_sample,
                        window=window,
                    )
                    aggregates[(condition, layer, boundary_name)].append(metrics)
                    trace_records.append(
                        {
                            "condition": condition,
                            "sample_id": row["sample_id"],
                            "seed": row["seed"],
                            "layer": layer,
                            "boundary": boundary_name,
                            "count": len(indices),
                            "symbolic_updates": len(symbolic_indices),
                            **metrics,
                        }
                    )

    results = []
    for (condition, layer, boundary), metrics in sorted(aggregates.items()):
        results.append(
            {
                "condition": condition,
                "layer": layer,
                "boundary": boundary,
                "traces": len(metrics),
                **{
                    key: mean_finite([record[key] for record in metrics])
                    for key in metrics[0]
                },
            }
        )
    out_dir = run_paths[0] / "analysis" / "experiments" / "h1_boundaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "hypothesis": "H1_text_boundaries_vs_natural_latent_boundaries",
        "runs": {
            condition: run_path.as_posix()
            for condition, (run_path, _, _) in condition_rows.items()
        },
        "evaluation": {
            "boundary_window_tokens": window,
            "independent_target": "restricted-AST-verified symbolic updates",
            "latent_spike_role": "geometric candidate and upper-bound segmentation",
        },
        "format_compliance": {
            condition: float(np.mean(values))
            for condition, values in compliance.items()
        },
        "results": results,
        "traces": trace_records,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def text_boundary_indices(
    text: str,
    token_spans: list[TokenSpan],
    token_count: int,
) -> dict[str, np.ndarray]:
    """Return starts of numbered steps and ends of sentences/paragraphs."""
    char_boundaries = {
        "numbered": [
            match.start() for match in re.finditer(r"(?m)^\s*Step\s+\d+\s*:", text)
        ],
        "sentence": [end for _, end in sentence_spans(text)],
        "paragraph": [end for _, end in paragraph_spans(text)],
    }
    result: dict[str, np.ndarray] = {}
    for name, positions in char_boundaries.items():
        indices: set[int] = set()
        for position in positions:
            token_range = token_range_for_chars(
                token_spans, max(position - 1, 0), min(position + 1, len(text))
            )
            if token_range is not None:
                indices.add(min(max(token_range[-1], 1), token_count - 1))
            elif text:
                indices.add(
                    min(
                        max(int(position / len(text) * token_count), 1), token_count - 1
                    )
                )
        result[name] = np.asarray(sorted(indices), dtype=np.int32)
    return result


def boundary_metrics(
    *,
    indices: np.ndarray,
    symbolic: np.ndarray,
    spikes: np.ndarray,
    magnitudes: np.ndarray,
    signal_values: np.ndarray,
    states: np.ndarray,
    quality_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
    window: int,
) -> dict[str, float]:
    silhouette, calinski_harabasz = segment_cluster_quality(indices, quality_sample)
    return {
        "symbolic_recall": overlap_fraction(symbolic, indices, window),
        "symbolic_precision": overlap_fraction(indices, symbolic, window),
        "spike_agreement": overlap_fraction(indices, spikes, window),
        "mean_delta_percentile": mean_finite(
            [percentile_rank(magnitudes[1:], magnitudes[index]) for index in indices]
        ),
        "mean_signal_percentile": mean_finite(
            [
                percentile_rank(signal_values[1:], signal_values[index])
                for index in indices
            ]
        ),
        "separation_ratio": segment_separation(states, indices),
        "silhouette": silhouette,
        "calinski_harabasz": calinski_harabasz,
    }


def latent_boundary_signals(
    row: dict[str, Any],
    states: np.ndarray,
    layer_col: int,
) -> dict[str, np.ndarray]:
    """Compute all H1 latent and token-diagnostic boundary scores."""
    deltas = latent_deltas(states)
    magnitude = np.linalg.norm(deltas, axis=1)
    curvature = np.zeros(len(states), dtype=np.float32)
    direction_change = np.zeros(len(states), dtype=np.float32)
    if len(states) > 2:
        curvature[2:] = np.linalg.norm(deltas[2:] - deltas[1:-1], axis=1)
        left = deltas[1:-1]
        right = deltas[2:]
        denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-8)
        direction_change[2:] = 1.0 - np.clip(cosine, -1.0, 1.0)
    signals = {
        "magnitude": magnitude,
        "curvature": curvature,
        "direction_change": direction_change,
    }
    timesteps = row.get("timesteps", [])
    if len(timesteps) >= len(states):
        for field, name in (("entropy", "entropy"), ("ce_next_token", "surprisal")):
            values = [
                timestep.get(field, [None])[layer_col]
                if timestep.get(field) and len(timestep[field]) > layer_col
                else None
                for timestep in timesteps[: len(states)]
            ]
            if all(value is not None for value in values):
                signals[name] = np.asarray(values, dtype=np.float32)
    return signals


def overlap_fraction(source: np.ndarray, target: np.ndarray, window: int) -> float:
    if not len(source):
        return float("nan")
    if not len(target):
        return 0.0
    return float(
        np.mean([np.min(np.abs(target - index)) <= window for index in source])
    )


def segment_separation(states: np.ndarray, boundaries: np.ndarray) -> float:
    """Compare adjacent segment-mean distance with within-segment variance."""
    cuts = [
        0,
        *[int(index) for index in boundaries if 0 < index < len(states)],
        len(states),
    ]
    segments = [
        states[start:end] for start, end in zip(cuts, cuts[1:]) if end - start >= 2
    ]
    if len(segments) < 2:
        return float("nan")
    means = [segment.mean(axis=0) for segment in segments]
    within = np.mean(
        [
            np.mean(np.sum((segment - mean) ** 2, axis=1))
            for segment, mean in zip(segments, means)
        ]
    )
    between = np.mean(
        [np.sum((right - left) ** 2) for left, right in zip(means, means[1:])]
    )
    return float(between / max(float(within), 1e-8))


def segment_quality_sample(
    states: np.ndarray,
    *,
    max_tokens: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute one bounded token sample and its pairwise distances."""
    positions = np.linspace(
        0,
        len(states) - 1,
        min(len(states), max_tokens),
        dtype=np.int32,
    )
    sampled = states[positions].astype(np.float32)
    distances = pairwise_distances(sampled, metric="euclidean")
    return positions, sampled, distances


def segment_cluster_quality(
    boundaries: np.ndarray,
    quality_sample: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[float, float]:
    positions, sampled, distances = quality_sample
    labels = np.searchsorted(np.sort(boundaries), positions, side="right")
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan"), float("nan")
    return (
        float(silhouette_score(distances, labels, metric="precomputed")),
        float(calinski_harabasz_score(sampled, labels)),
    )


def format_compliance(text: str, condition: str) -> float:
    if condition == "numbered":
        return float(bool(re.search(r"(?m)^\s*Step\s+1\s*:", text)))
    paragraphs = paragraph_spans(text)
    if condition == "sentence_separated":
        if not paragraphs:
            return 0.0
        return float(
            np.mean(
                [len(sentence_spans(text[start:end])) == 1 for start, end in paragraphs]
            )
        )
    if condition == "paragraph_separated":
        multi_sentence = [
            len(sentence_spans(text[start:end])) >= 2 for start, end in paragraphs
        ]
        return float(np.mean(multi_sentence)) if multi_sentence else 0.0
    return 1.0


def mean_finite(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.mean(finite)) if len(finite) else float("nan")
