#!/usr/bin/env python3
"""Retest H2 peak localization on finer token-level activation captures."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reasoning_trajectory.token_alignment import build_token_spans
from src.experiments.common import (
    balanced_generation_rows,
    latent_deltas,
    percentile_rank,
    update_phase_bounds,
)
from src.experiments.symbolic import extract_symbolic_updates
from src.runtime.artifact_store import load_hidden_states_npz
from src.runtime.data import write_jsonl


DEFAULT_RUN = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/"
    "gsm_symb_40_60_full_token_mlx"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", nargs="?", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--per-sample", type=int, default=8)
    parser.add_argument("--radius", type=int, default=16)
    args = parser.parse_args()
    report_path = run_retest(args.run_path, per_sample=args.per_sample, radius=args.radius)
    print(report_path)
    return 0


def run_retest(run_path: Path, *, per_sample: int, radius: int) -> Path:
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    spans_by_row = build_token_spans(run_path, rows)
    out_dir = run_path / "analysis" / "experiments" / "h2_fine_peak_retest"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for row, token_spans in zip(rows, spans_by_row):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        count = min(len(row.get("generated_token_ids", [])), states.shape[0])
        if count < 3:
            continue
        updates = extract_symbolic_updates(
            str(row.get("produced_text", "")),
            token_spans,
            token_count=count,
        )
        if not updates:
            continue
        for layer_col, layer in enumerate(layers):
            layer_states = states[:count, layer_col].astype(np.float32)
            magnitudes = np.linalg.norm(latent_deltas(layer_states), axis=1)
            baseline = magnitudes[1:] if len(magnitudes) > 1 else magnitudes
            for update_index, update in enumerate(updates):
                phase_start, phase_end = update_phase_bounds(
                    update.token_start,
                    update.token_end,
                    count,
                )
                window_start = max(1, phase_start - radius)
                window_end = min(count - 1, phase_end + radius)
                if window_end < window_start:
                    continue
                window_values = magnitudes[window_start : window_end + 1]
                peak_index = int(window_start + np.argmax(window_values))
                interval_start = max(1, phase_start)
                interval_end = max(interval_start, phase_end)
                interval_values = magnitudes[interval_start : interval_end + 1]
                interval_peak = int(interval_start + np.argmax(interval_values))
                endpoint_value = float(magnitudes[phase_end])
                peak_value = float(magnitudes[peak_index])
                records.append(
                    {
                        "sample_id": row["sample_id"],
                        "seed": row["seed"],
                        "trajectory_id": f"{row['sample_id']}::{row['seed']}",
                        "is_correct": row.get("is_correct"),
                        "layer": layer,
                        "update_index": update_index,
                        **update.to_record(),
                        "phase_start_state_index": phase_start,
                        "phase_end_state_index": phase_end,
                        "radius": radius,
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_peak_state_index": peak_index,
                        "window_peak_offset_from_end": peak_index - phase_end,
                        "window_peak_inside_update": phase_start <= peak_index <= phase_end,
                        "interval_peak_state_index": interval_peak,
                        "interval_peak_offset_from_end": interval_peak - phase_end,
                        "endpoint_delta_norm": endpoint_value,
                        "window_peak_delta_norm": peak_value,
                        "interval_peak_delta_norm": float(magnitudes[interval_peak]),
                        "window_peak_to_endpoint_ratio": peak_value / max(endpoint_value, 1e-8),
                        "endpoint_delta_percentile": percentile_rank(baseline, endpoint_value),
                        "window_peak_delta_percentile": percentile_rank(baseline, peak_value),
                    }
                )

    write_jsonl(out_dir / "peaks.jsonl", records)
    report = build_report(run_path, rows, records, per_sample=per_sample, radius=radius)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def build_report(
    run_path: Path,
    rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    per_sample: int,
    radius: int,
) -> dict[str, Any]:
    by_layer: dict[str, dict[str, Any]] = {}
    for layer in sorted({record["layer"] for record in records}):
        layer_records = [record for record in records if record["layer"] == layer]
        offsets = np.asarray(
            [record["window_peak_offset_from_end"] for record in layer_records],
            dtype=np.float32,
        )
        by_layer[str(layer)] = {
            "updates": len(layer_records),
            "mean_peak_offset_from_update_end": mean_or_none(offsets),
            "median_peak_offset_from_update_end": median_or_none(offsets),
            "mean_abs_peak_offset_from_update_end": mean_or_none(np.abs(offsets)),
            "fraction_window_peak_inside_update": mean_or_none(
                [record["window_peak_inside_update"] for record in layer_records]
            ),
            "mean_window_peak_to_endpoint_ratio": mean_or_none(
                [record["window_peak_to_endpoint_ratio"] for record in layer_records]
            ),
            "mean_endpoint_delta_percentile": mean_or_none(
                [record["endpoint_delta_percentile"] for record in layer_records]
            ),
            "mean_window_peak_delta_percentile": mean_or_none(
                [record["window_peak_delta_percentile"] for record in layer_records]
            ),
            "offset_histogram": dict(Counter(int(value) for value in offsets)),
            "by_correctness": correctness_summary(layer_records),
        }
    return {
        "hypothesis": "H2_fine_peak_retest",
        "source_run": run_path.as_posix(),
        "selection": {
            "trajectories": len(rows),
            "questions": len({row["sample_id"] for row in rows}),
            "per_sample_cap": per_sample,
        },
        "definition": {
            "radius_tokens": radius,
            "peak": "max token-to-token activation-change magnitude in update_end +/- radius",
            "activation_sampling": "one final-layer state per generated token",
        },
        "updates": len(records),
        "operator_counts": dict(Counter(record["operator"] for record in records)),
        "layers": by_layer,
    }


def correctness_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("is_correct"))].append(record)
    return {
        key: {
            "updates": len(items),
            "mean_peak_offset_from_update_end": mean_or_none(
                [item["window_peak_offset_from_end"] for item in items]
            ),
            "fraction_window_peak_inside_update": mean_or_none(
                [item["window_peak_inside_update"] for item in items]
            ),
            "mean_window_peak_to_endpoint_ratio": mean_or_none(
                [item["window_peak_to_endpoint_ratio"] for item in items]
            ),
        }
        for key, items in grouped.items()
    }


def mean_or_none(values: Any) -> float | None:
    array = np.asarray(values, dtype=np.float32)
    return float(array.mean()) if array.size else None


def median_or_none(values: Any) -> float | None:
    array = np.asarray(values, dtype=np.float32)
    return float(np.median(array)) if array.size else None


if __name__ == "__main__":
    raise SystemExit(main())
