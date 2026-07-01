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
from src.runtime.data import load_samples


INTERVAL_EFFECT_FIELDS = (
    "path_length_control_percentile",
    "net_displacement_control_percentile",
    "peak_share",
    "effective_width_fraction",
    "net_to_path_ratio",
)


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
    condition_summary = {
        condition: summarize_condition(run_path, rows)
        for condition, (run_path, rows, _) in condition_rows.items()
    }
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
        "condition_summary": condition_summary,
        "matched_behavior_effects": matched_behavior_effects(condition_rows),
        "matched_interval_effects": matched_interval_effects(condition_rows),
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
    """Score one boundary set against symbolic, latent, and cluster criteria."""
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
    """Return the fraction of source indices near any target index."""
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
    """Measure how well boundaries partition a bounded sample of latent states."""
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
    """Measure compliance with the requested reasoning format."""
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
    """Return the mean of finite values or NaN when none exist."""
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(np.mean(finite)) if len(finite) else float("nan")


def summarize_condition(
    run_path: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize completion, accuracy, and length for one H1 condition."""
    lengths = np.asarray(
        [len(row.get("generated_token_ids", [])) for row in rows],
        dtype=np.int32,
    )
    scored = [
        bool(row["is_correct"]) for row in rows if row.get("is_correct") is not None
    ]
    expected = expected_trajectories(run_path)
    return {
        "trajectories": len(rows),
        "expected_trajectories": expected,
        "completion_fraction": len(rows) / expected if expected else None,
        "questions": len({str(row["sample_id"]) for row in rows}),
        "accuracy": float(np.mean(scored)) if scored else None,
        "mean_tokens": float(lengths.mean()) if len(lengths) else None,
        "median_tokens": float(np.median(lengths)) if len(lengths) else None,
    }


def expected_trajectories(run_path: Path) -> int | None:
    """Infer the configured trajectory count when the run schema permits it."""
    config = load_config(run_path)
    if "replay" in config:
        maximum = int(config["replay"].get("max_trajectories", 0))
        return maximum or None
    dataset_path = run_path / "dataset.jsonl"
    if "generation" in config and dataset_path.exists():
        rows = sum(1 for line in dataset_path.open() if line.strip())
        return rows * int(config["generation"].get("num_samples_per_item", 1))
    return None


def matched_behavior_effects(
    condition_rows: dict[
        str,
        tuple[Path, list[dict[str, Any]], list[list[TokenSpan]]],
    ],
) -> list[dict[str, Any]]:
    """Compare prompted conditions with freeform using matched question/seed rows."""
    if "freeform" not in condition_rows:
        return []
    baseline = {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in condition_rows["freeform"][1]
    }
    effects: list[dict[str, Any]] = []
    for condition, (_, rows, _) in condition_rows.items():
        if condition == "freeform":
            continue
        prompted = {(str(row["sample_id"]), int(row["seed"])): row for row in rows}
        keys = sorted(set(baseline) & set(prompted))
        groups = sorted({sample_id for sample_id, _ in keys})
        if not keys:
            continue
        accuracy_by_group: dict[str, list[float]] = defaultdict(list)
        token_ratio_by_group: dict[str, list[float]] = defaultdict(list)
        for key in keys:
            base_row = baseline[key]
            prompted_row = prompted[key]
            if (
                base_row.get("is_correct") is not None
                and prompted_row.get("is_correct") is not None
            ):
                accuracy_by_group[key[0]].append(
                    float(bool(prompted_row["is_correct"]))
                    - float(bool(base_row["is_correct"]))
                )
            base_tokens = max(len(base_row.get("generated_token_ids", [])), 1)
            token_ratio_by_group[key[0]].append(
                len(prompted_row.get("generated_token_ids", [])) / base_tokens
            )
        accuracy_values = np.asarray(
            [
                np.mean(accuracy_by_group[group])
                for group in groups
                if accuracy_by_group[group]
            ]
        )
        token_values = np.asarray(
            [np.mean(token_ratio_by_group[group]) for group in groups]
        )
        effects.append(
            {
                "condition": condition,
                "matched_trajectories": len(keys),
                "matched_questions": len(groups),
                "accuracy_difference": (
                    float(accuracy_values.mean()) if len(accuracy_values) else None
                ),
                "accuracy_difference_95ci": grouped_bootstrap_interval(accuracy_values),
                "token_ratio": float(token_values.mean()),
                "token_ratio_95ci": grouped_bootstrap_interval(token_values),
            }
        )
    return effects


def grouped_bootstrap_interval(
    values: np.ndarray,
    *,
    draws: int = 1000,
) -> list[float] | None:
    """Bootstrap a 95% interval over already grouped values."""
    if not len(values):
        return None
    rng = np.random.default_rng(42)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(draws)
    ]
    return np.quantile(means, [0.025, 0.975]).tolist()


def matched_interval_effects(
    condition_rows: dict[
        str,
        tuple[Path, list[dict[str, Any]], list[list[TokenSpan]]],
    ],
) -> list[dict[str, Any]]:
    """Compare question-balanced H2 interval metrics with matched freeform rows."""
    if "freeform" not in condition_rows:
        return []
    baseline = load_interval_trace_metrics(condition_rows["freeform"][0])
    effects = []
    for condition, (run_path, _, _) in condition_rows.items():
        if condition == "freeform":
            continue
        prompted = load_interval_trace_metrics(run_path)
        keys = sorted(set(baseline) & set(prompted))
        if not keys:
            continue
        metrics = {}
        for field in INTERVAL_EFFECT_FIELDS:
            differences: defaultdict[str, list[float]] = defaultdict(list)
            for key in keys:
                differences[key[0]].append(prompted[key][field] - baseline[key][field])
            question_values = np.asarray(
                [np.mean(values) for values in differences.values()],
                dtype=np.float64,
            )
            metrics[field] = {
                "question_mean_difference": float(question_values.mean()),
                "question_bootstrap_95ci": grouped_bootstrap_interval(question_values),
            }
        effects.append(
            {
                "condition": condition,
                "matched_trajectories": len(keys),
                "matched_questions": len({key[0] for key in keys}),
                "metrics": metrics,
            }
        )
    return effects


def load_interval_trace_metrics(
    run_path: Path,
) -> dict[tuple[str, int], dict[str, float]]:
    """Load mean final-layer interval metrics for each trace."""
    path = (
        run_path / "analysis" / "experiments" / "h2_localized_updates" / "updates.jsonl"
    )
    if not path.exists():
        return {}
    values: defaultdict[
        tuple[str, int],
        defaultdict[str, list[float]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in load_samples(path.resolve()):
        if int(row["layer"]) != -1:
            continue
        key = (str(row["sample_id"]), int(row["seed"]))
        for field in INTERVAL_EFFECT_FIELDS:
            value = row.get(field)
            if value is not None:
                values[key][field].append(float(value))
    return {
        key: {
            field: float(np.mean(fields[field]))
            for field in INTERVAL_EFFECT_FIELDS
            if fields[field]
        }
        for key, fields in values.items()
        if all(fields[field] for field in INTERVAL_EFFECT_FIELDS)
    }
