"""Measure distributed symbolic updates in residual, MLP, and attention paths."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import read_generation_rows
from src.experiments.common import (
    control_percentile,
    interval_dynamics,
    matched_control_dynamics,
    update_phase_bounds,
)
from src.experiments.localized_updates import mean_or_none
from src.runtime.artifact_store import (
    load_component_states_npz,
    load_hidden_states_npz,
)
from src.runtime.data import load_samples, write_jsonl


COMPONENTS = ("residual", "mlp_output", "attention_output")
CONTROL_FIELDS = (
    "integrated_vector_norm",
    "path_length",
    "net_displacement",
    "cumulative_state_cosine_distance",
    "cumulative_derivative_cosine_distance",
    "effective_width_fraction",
    "peak_share",
)


def run_component_localization(
    replay_run: Path,
    h2_dir: Path,
) -> Path:
    """Integrate component dynamics across each verified symbolic interval.

    Args:
        replay_run: Run directory containing teacher-forced replay artifacts.
        h2_dir: Directory containing H2 update-analysis artifacts.

    Returns:
        The path of the written or discovered artifact.
    """
    updates = load_samples((h2_dir / "updates.jsonl").resolve())
    by_trace: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        by_trace[(str(update["sample_id"]), int(update["seed"]))].append(update)

    rows = [
        row for row in read_generation_rows(replay_run) if row.get("hidden_states_file")
    ]
    aggregate: defaultdict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    interval_records: list[dict[str, Any]] = []
    for row in rows:
        trace_updates = by_trace[(str(row["sample_id"]), int(row["seed"]))]
        if not trace_updates:
            continue
        artifact = replay_run / row["hidden_states_file"]
        for component in COMPONENTS:
            loaded = load_component(artifact, component)
            if loaded is None:
                continue
            states, layers = loaded
            phase_bounds = [
                update_phase_bounds(
                    update["token_start"],
                    update["token_end"],
                    len(states),
                )
                for update in trace_updates
            ]
            for layer_col, layer in enumerate(layers):
                component_states = states[:, layer_col].astype(np.float32)
                controls_by_duration = {}
                # Reuse same-duration null windows within a trace/component/layer
                # so update intervals are compared against an identical baseline.
                for update, (phase_start, phase_end) in zip(
                    trace_updates, phase_bounds
                ):
                    dynamics = interval_dynamics(
                        component_states,
                        phase_start,
                        phase_end,
                    )
                    if dynamics.token_count not in controls_by_duration:
                        controls_by_duration[dynamics.token_count] = (
                            matched_control_dynamics(
                                component_states,
                                duration=dynamics.token_count,
                                excluded=phase_bounds,
                            )
                        )
                    controls = controls_by_duration[dynamics.token_count]
                    percentiles = {
                        f"{field}_control_percentile": control_percentile(
                            float(getattr(dynamics, field)),
                            controls,
                            field,
                        )
                        for field in CONTROL_FIELDS
                    }
                    record = {
                        "sample_id": row["sample_id"],
                        "seed": row["seed"],
                        "trajectory_id": f"{row['sample_id']}::{row['seed']}",
                        "component": component,
                        "layer": layer,
                        "operator": update["operator"],
                        "operation_signature": update["operation_signature"],
                        "token_start": update["token_start"],
                        "token_end": update["token_end"],
                        "phase_start_state_index": phase_start,
                        "phase_end_state_index": phase_end,
                        **dynamics.scalar_record(),
                        **percentiles,
                    }
                    interval_records.append(record)
                    values = aggregate[(component, layer)]
                    for field, value in dynamics.scalar_record().items():
                        values[field].append(float(value))
                    for field, value in percentiles.items():
                        if value is not None:
                            values[field].append(float(value))

    results = []
    for (component, layer), values in aggregate.items():
        result = summarize_component(component, layer, values)
        result["question_grouped"] = grouped_component_metrics(
            interval_records,
            component=component,
            layer=layer,
        )
        score_fields = (
            "path_length_control_percentile",
            "net_displacement_control_percentile",
            "cumulative_state_cosine_distance_control_percentile",
        )
        result["question_interval_signal_score"] = mean_or_none(
            [
                result["question_grouped"][field]["question_mean"]
                for field in score_fields
            ]
        )
        results.append(result)
    results.sort(
        key=lambda record: record["question_interval_signal_score"] or 0.0,
        reverse=True,
    )
    patch_candidates = [
        result for result in results if result["component"] != "residual"
    ]
    report = {
        "hypothesis": "H2_distributed_component_updates",
        "replay_run": replay_run.as_posix(),
        "source_updates": h2_dir.as_posix(),
        "traces": len(rows),
        "localization_model": (
            "integrated activation path over [token_start, token_end], "
            "compared with same-length non-update windows"
        ),
        "results": results,
        "recommended_patch_target": (
            {
                "component": patch_candidates[0]["component"],
                "layer": patch_candidates[0]["layer"],
                "alignment": "symbolic_step_end",
            }
            if patch_candidates
            else None
        ),
        "interpretation": component_interpretation(patch_candidates),
    }
    out_dir = replay_run / "analysis" / "experiments" / "h2_component_localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "intervals.jsonl", interval_records)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def load_component(
    artifact: Path,
    component: str,
) -> tuple[np.ndarray, list[int]] | None:
    """Load residual or component activations from an artifact.

    Args:
        artifact: NPZ activation artifact to load.
        component: Activation component name.

    Returns:
        The computed aligned values described above.
    """
    try:
        if component == "residual":
            return load_hidden_states_npz(artifact)
        return load_component_states_npz(artifact, component)
    except KeyError:
        return None


def summarize_component(
    component: str,
    layer: int,
    values: dict[str, list[float]],
) -> dict[str, Any]:
    """Aggregate interval-localization metrics for one component layer.

    Args:
        component: Activation component name.
        layer: Model layer index.
        values: Values to summarize or transform.

    Returns:
        The resulting keyed records or metrics.
    """
    path_percentile = mean_or_none(values.get("path_length_control_percentile", []))
    cosine_percentile = mean_or_none(
        values.get("cumulative_state_cosine_distance_control_percentile", [])
    )
    net_percentile = mean_or_none(values.get("net_displacement_control_percentile", []))
    score = mean_or_none(
        [
            value
            for value in (path_percentile, cosine_percentile, net_percentile)
            if value is not None
        ]
    )
    return {
        "component": component,
        "layer": layer,
        "intervals": len(values.get("path_length", [])),
        "mean_interval_signal_score": score,
        "mean_integrated_output_norm_percentile": mean_or_none(
            values.get("integrated_vector_norm_control_percentile", [])
        ),
        "mean_path_length_percentile": path_percentile,
        "mean_net_displacement_percentile": net_percentile,
        "mean_state_cosine_distance_percentile": cosine_percentile,
        "mean_derivative_cosine_distance_percentile": mean_or_none(
            values.get(
                "cumulative_derivative_cosine_distance_control_percentile",
                [],
            )
        ),
        "mean_peak_share": mean_or_none(values.get("peak_share", [])),
        "mean_effective_width_tokens": mean_or_none(
            values.get("effective_width_tokens", [])
        ),
        "mean_effective_width_fraction": mean_or_none(
            values.get("effective_width_fraction", [])
        ),
        "mean_effective_width_percentile": mean_or_none(
            values.get("effective_width_fraction_control_percentile", [])
        ),
        "mean_net_to_path_ratio": mean_or_none(values.get("net_to_path_ratio", [])),
        "mean_temporal_centroid": mean_or_none(values.get("temporal_centroid", [])),
    }


def component_interpretation(results: list[dict[str, Any]]) -> str:
    """Classify the strongest component-localization pattern.

    Args:
        results: Ranked component-layer summaries.

    Returns:
        The resulting text or classification label.
    """
    if not results:
        return "insufficient_data"
    strongest = results[0]
    score = (
        strongest.get("question_interval_signal_score")
        or strongest["mean_interval_signal_score"]
        or 0.0
    )
    width = strongest["mean_effective_width_fraction"] or 0.0
    peak_share = strongest["mean_peak_share"] or 1.0
    if score >= 0.6 and width >= 0.5 and peak_share <= 0.5:
        return "distributed_component_transition_supported"
    if score >= 0.6:
        return "interval_elevation_without_clear_wave_shape"
    return "no_component_localization_support"


def grouped_component_metrics(
    records: list[dict[str, Any]],
    *,
    component: str,
    layer: int,
    draws: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Balance component metrics over questions and bootstrap their means.

    Args:
        records: Aligned records to analyze or annotate.
        component: Activation component name.
        layer: Model layer index.
        draws: Number of bootstrap resamples.

    Returns:
        The resulting keyed records or metrics.
    """
    fields = (
        "integrated_vector_norm_control_percentile",
        "path_length_control_percentile",
        "net_displacement_control_percentile",
        "cumulative_state_cosine_distance_control_percentile",
        "cumulative_derivative_cosine_distance_control_percentile",
        "peak_share",
        "effective_width_fraction",
        "net_to_path_ratio",
        "temporal_centroid",
    )
    grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        if record["component"] != component or int(record["layer"]) != layer:
            continue
        for field in fields:
            value = record.get(field)
            if value is not None:
                grouped[str(record["sample_id"])][field].append(float(value))
    rng = np.random.default_rng(42)
    summary = {}
    for field in fields:
        values = np.asarray(
            [
                np.mean(question[field])
                for question in grouped.values()
                if question[field]
            ],
            dtype=np.float64,
        )
        bootstraps = np.asarray(
            [
                np.mean(rng.choice(values, size=len(values), replace=True))
                for _ in range(draws)
            ]
        )
        summary[field] = {
            "question_mean": float(values.mean()),
            "question_bootstrap_95ci": np.quantile(
                bootstraps,
                [0.025, 0.975],
            ).tolist(),
            "questions": len(values),
        }
    return summary
