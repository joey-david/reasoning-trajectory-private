"""Measure latent dynamics across symbolically verified update intervals."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.analysis.token_alignment import build_token_spans
from src.experiments.common import (
    balanced_generation_rows,
    control_percentile,
    interval_dynamics,
    latent_deltas,
    matched_control_dynamics,
    nearest_distance,
    percentile_rank,
    robust_spike_indices,
    update_phase_bounds,
)
from src.experiments.symbolic import extract_symbolic_updates
from src.runtime.artifact_store import load_hidden_states_npz
from src.runtime.data import write_jsonl


def run_localized_update_analysis(
    run_path: Path,
    *,
    per_sample: int = 10,
    spike_z: float = 3.0,
    window: int = 2,
) -> Path:
    """Run H2 on existing captured states and write reusable update vectors."""
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    spans_by_row = build_token_spans(run_path, rows)
    out_dir = run_path / "analysis" / "experiments" / "h2_localized_updates"
    out_dir.mkdir(parents=True, exist_ok=True)

    update_records: list[dict[str, Any]] = []
    vectors_by_layer: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    trace_summaries: list[dict[str, Any]] = []
    layer_stats: defaultdict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row_index, (row, token_spans) in enumerate(zip(rows, spans_by_row)):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        count = min(len(row.get("generated_token_ids", [])), states.shape[0])
        if count < 3:
            continue
        updates = extract_symbolic_updates(
            str(row.get("produced_text", "")),
            token_spans,
            token_count=count,
        )
        phase_bounds = [
            update_phase_bounds(update.token_start, update.token_end, count)
            for update in updates
        ]
        trace_record = {
            "sample_id": row["sample_id"],
            "seed": row["seed"],
            "is_correct": row["is_correct"],
            "tokens": count,
            "updates": len(updates),
            "layers": {},
        }
        for layer_col, layer in enumerate(layers):
            layer_states = states[:count, layer_col].astype(np.float32)
            deltas = latent_deltas(layer_states)
            magnitudes = np.linalg.norm(deltas, axis=1)
            spikes = robust_spike_indices(magnitudes, z_threshold=spike_z)
            controls_by_duration = {}
            endpoint_labels = np.zeros(count, dtype=np.int8)
            state_indices = [phase_end for _, phase_end in phase_bounds]
            endpoint_labels[state_indices] = 1
            auc = (
                float(roc_auc_score(endpoint_labels, magnitudes))
                if 0 < endpoint_labels.sum() < count
                else None
            )
            hit_rate = (
                float(
                    np.mean(
                        [
                            nearest_distance(spikes, endpoint) is not None
                            and nearest_distance(spikes, endpoint) <= window
                            for endpoint in state_indices
                        ]
                    )
                )
                if state_indices
                else None
            )
            null_hit_rate = shifted_null_hit_rate(
                endpoints=state_indices,
                spikes=spikes,
                token_count=count,
                window=window,
            )
            trace_record["layers"][str(layer)] = {
                "spikes": len(spikes),
                "update_auc": auc,
                "update_hit_rate": hit_rate,
                "shifted_null_hit_rate": null_hit_rate,
            }
            if auc is not None:
                layer_stats[layer]["auc"].append(auc)
            if hit_rate is not None:
                layer_stats[layer]["hit_rate"].append(hit_rate)
                layer_stats[layer]["null_hit_rate"].append(null_hit_rate)

            interval_records: list[dict[str, float | int | None]] = []
            for update_index, (update, (phase_start, phase_end)) in enumerate(
                zip(updates, phase_bounds)
            ):
                state_index = phase_end
                distance = nearest_distance(spikes, state_index)
                dynamics = interval_dynamics(
                    layer_states,
                    phase_start,
                    phase_end,
                )
                if dynamics.token_count not in controls_by_duration:
                    controls_by_duration[dynamics.token_count] = (
                        matched_control_dynamics(
                            layer_states,
                            duration=dynamics.token_count,
                            excluded=phase_bounds,
                        )
                    )
                controls = controls_by_duration[dynamics.token_count]
                interval_metrics = {
                    **dynamics.scalar_record(),
                    "path_length_control_percentile": control_percentile(
                        dynamics.path_length,
                        controls,
                        "path_length",
                    ),
                    "net_displacement_control_percentile": control_percentile(
                        dynamics.net_displacement,
                        controls,
                        "net_displacement",
                    ),
                    "effective_width_control_percentile": control_percentile(
                        dynamics.effective_width_fraction,
                        controls,
                        "effective_width_fraction",
                    ),
                    "peak_share_control_percentile": control_percentile(
                        dynamics.peak_share,
                        controls,
                        "peak_share",
                    ),
                }
                record = {
                    "feature_row": len(vectors_by_layer[layer]),
                    "sample_id": row["sample_id"],
                    "seed": row["seed"],
                    "trajectory_id": f"{row['sample_id']}::{row['seed']}",
                    "is_correct": row["is_correct"],
                    "layer": layer,
                    "update_index": update_index,
                    **update.to_record(),
                    "phase_start_state_index": phase_start,
                    "phase_end_state_index": phase_end,
                    "state_index": state_index,
                    **interval_metrics,
                    "delta_norm": float(magnitudes[state_index]),
                    "delta_percentile": percentile_rank(
                        magnitudes[1:], float(magnitudes[state_index])
                    ),
                    "nearest_spike_distance": distance,
                    "is_spike_nearby": distance is not None and distance <= window,
                }
                update_records.append(record)
                vectors_by_layer[layer].append(dynamics.net_vector.copy())
                interval_records.append(interval_metrics)
                for metric in (
                    "path_length_control_percentile",
                    "net_displacement_control_percentile",
                    "effective_width_control_percentile",
                    "peak_share",
                    "effective_width_fraction",
                    "net_to_path_ratio",
                ):
                    value = interval_metrics[metric]
                    if value is not None:
                        layer_stats[layer][metric].append(float(value))
            trace_record["layers"][str(layer)].update(
                {
                    "mean_interval_path_percentile": record_mean(
                        interval_records,
                        "path_length_control_percentile",
                    ),
                    "mean_interval_effective_width_fraction": record_mean(
                        interval_records,
                        "effective_width_fraction",
                    ),
                    "mean_interval_peak_share": record_mean(
                        interval_records,
                        "peak_share",
                    ),
                }
            )
        trace_summaries.append(trace_record)

    write_jsonl(out_dir / "updates.jsonl", update_records)
    write_jsonl(out_dir / "traces.jsonl", trace_summaries)
    for layer, vectors in vectors_by_layer.items():
        np.savez_compressed(
            out_dir / f"layer{layer}_update_vectors.npz",
            net_update_vectors=np.stack(vectors).astype(np.float16)
            if vectors
            else np.empty((0, 0), dtype=np.float16),
        )

    report = build_report(
        run_path=run_path,
        rows=rows,
        update_records=update_records,
        layer_stats=layer_stats,
        spike_z=spike_z,
        window=window,
    )
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def shifted_null_hit_rate(
    *,
    endpoints: list[int],
    spikes: np.ndarray,
    token_count: int,
    window: int,
    shifts: int = 31,
) -> float:
    """Estimate chance overlap by circularly shifting all update endpoints."""
    if not endpoints or token_count < 2:
        return 0.0
    offsets = np.linspace(1, token_count - 1, min(shifts, token_count - 1), dtype=int)
    hits: list[float] = []
    for offset in offsets:
        shifted = [
            1 + ((endpoint - 1 + int(offset)) % (token_count - 1))
            for endpoint in endpoints
        ]
        hits.append(
            float(
                np.mean(
                    [
                        nearest_distance(spikes, endpoint) is not None
                        and nearest_distance(spikes, endpoint) <= window
                        for endpoint in shifted
                    ]
                )
            )
        )
    return float(np.mean(hits))


def build_report(
    *,
    run_path: Path,
    rows: list[dict[str, Any]],
    update_records: list[dict[str, Any]],
    layer_stats: dict[int, dict[str, list[float]]],
    spike_z: float,
    window: int,
) -> dict[str, Any]:
    """Build the H2 report from trace, update, and layer statistics."""
    operator_counts = Counter(record["operator"] for record in update_records)
    signatures = Counter(record["operation_signature"] for record in update_records)
    layers: dict[str, Any] = {}
    for layer, stats in layer_stats.items():
        hit_rates = np.asarray(stats.get("hit_rate", []), dtype=np.float32)
        null_rates = np.asarray(stats.get("null_hit_rate", []), dtype=np.float32)
        layers[str(layer)] = {
            "traces_with_updates": len(hit_rates),
            "mean_update_auc": mean_or_none(stats.get("auc", [])),
            "mean_update_hit_rate": mean_or_none(hit_rates),
            "mean_shifted_null_hit_rate": mean_or_none(null_rates),
            "mean_hit_rate_lift": mean_or_none(hit_rates - null_rates),
            "mean_interval_path_percentile": mean_or_none(
                stats.get("path_length_control_percentile", [])
            ),
            "mean_interval_net_displacement_percentile": mean_or_none(
                stats.get("net_displacement_control_percentile", [])
            ),
            "mean_interval_effective_width_percentile": mean_or_none(
                stats.get("effective_width_control_percentile", [])
            ),
            "mean_peak_share": mean_or_none(stats.get("peak_share", [])),
            "mean_effective_width_fraction": mean_or_none(
                stats.get("effective_width_fraction", [])
            ),
            "mean_net_to_path_ratio": mean_or_none(stats.get("net_to_path_ratio", [])),
        }
    return {
        "hypothesis": "H2_localized_solution_object_updates",
        "source_run": run_path.as_posix(),
        "selection": {
            "trajectories": len(rows),
            "questions": len({row["sample_id"] for row in rows}),
            "per_sample_cap": max(
                Counter(row["sample_id"] for row in rows).values(), default=0
            ),
        },
        "definition": {
            "spike_z_mad": spike_z,
            "boundary_window_tokens": window,
            "symbolic_updates": "restricted-AST-verified arithmetic relations",
            "interval_metric": (
                "integrated activation path from token_start pre-state through "
                "token_end completed-state"
            ),
            "update_vector": "net state displacement across the full interval",
        },
        "updates": len(update_records),
        "operator_counts": dict(operator_counts),
        "operation_signatures": dict(signatures.most_common()),
        "layers": layers,
        "question_grouped_interval_metrics": grouped_interval_metrics(update_records),
        "operator_interval_metrics": {
            operator: {
                "updates": count,
                "question_grouped": grouped_interval_metrics(
                    [
                        record
                        for record in update_records
                        if record["operator"] == operator
                    ]
                ),
            }
            for operator, count in sorted(operator_counts.items())
        },
        "interpretation": h2_interpretation(layers),
    }


def mean_or_none(values: Any) -> float | None:
    """Return a numeric mean or None for an empty collection."""
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)) if array.size else None


def record_mean(
    records: list[dict[str, float | int | None]],
    field: str,
) -> float | None:
    """Average one populated field across interval records."""
    return mean_or_none(
        [record[field] for record in records if record.get(field) is not None]
    )


def grouped_interval_metrics(
    records: list[dict[str, Any]],
    *,
    draws: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Bootstrap interval metrics over questions rather than update rows."""
    fields = (
        "path_length_control_percentile",
        "net_displacement_control_percentile",
        "effective_width_control_percentile",
        "peak_share",
        "effective_width_fraction",
        "net_to_path_ratio",
    )
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        for field in fields:
            value = record.get(field)
            if value is not None:
                grouped[str(record["sample_id"])][field].append(float(value))
    rng = np.random.default_rng(42)
    summary: dict[str, dict[str, Any]] = {}
    for field in fields:
        values = np.asarray(
            [
                np.mean(question[field])
                for question in grouped.values()
                if question[field]
            ],
            dtype=np.float64,
        )
        if not len(values):
            continue
        bootstraps = np.asarray(
            [
                np.mean(rng.choice(values, size=len(values), replace=True))
                for _ in range(draws)
            ]
        )
        summary[field] = {
            "question_mean": float(values.mean()),
            "question_bootstrap_95ci": np.quantile(bootstraps, [0.025, 0.975]).tolist(),
            "questions": len(values),
        }
    return summary


def h2_interpretation(layers: dict[str, Any]) -> str:
    """Distinguish elevated update magnitude from genuinely sharp localization."""
    if not layers:
        return "insufficient_data"
    strongest = max(
        layers.values(),
        key=lambda record: record.get("mean_update_auc") or 0.0,
    )
    auc = strongest.get("mean_update_auc") or 0.0
    lift = strongest.get("mean_hit_rate_lift") or 0.0
    path_percentile = strongest.get("mean_interval_path_percentile") or 0.0
    width = strongest.get("mean_effective_width_fraction") or 0.0
    peak_share = strongest.get("mean_peak_share") or 1.0
    if path_percentile >= 0.6 and width >= 0.5 and peak_share <= 0.5 and lift < 0.05:
        return "distributed_transition_band_supported"
    if auc >= 0.6 and lift >= 0.05:
        return "sharp_localization_supported"
    if auc >= 0.6 and lift > 0.0:
        return "elevated_change_without_strong_sharp_localization"
    return "no_localization_support"
