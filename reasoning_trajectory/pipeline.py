"""Run the reusable trajectory-analysis bundle over one completed run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from reasoning_trajectory.bank import load_trajectory_bank
from reasoning_trajectory.metrics.alignment import alignment_summary, trajectory_pairs
from reasoning_trajectory.metrics.diagnostics import (
    basin_summary,
    compression_curve,
    failure_autopsies,
)
from reasoning_trajectory.metrics.geometry import trajectory_geometry


def analyze_trajectories(
    run_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute and persist the complete bounded latent-trajectory report."""
    run_path = Path(run_path)
    config = config or {}
    paths = load_trajectory_bank(run_path, config)
    output_path = run_path / "analysis" / "trajectory_metrics.json"
    if len(paths) < 2:
        if output_path.exists():
            output_path.unlink()
        return {}

    geometry = [trajectory_geometry(path) for path in paths]
    pairs = trajectory_pairs(paths, int(config.get("max_pairs", 200)))
    alignments = [
        {
            "a": a.trajectory_id,
            "b": b.trajectory_id,
            "sample_id": a.sample_id if a.sample_id == b.sample_id else None,
            "same_outcome": a.is_correct == b.is_correct,
            **alignment_summary(a.states, b.states),
        }
        for a, b in pairs
    ]
    dimensions = [
        int(value) for value in config.get("compression_dimensions", [2, 3, 8, 16])
    ]
    report = {
        "schema_version": 1,
        "layer": paths[0].layer,
        "sampling": {
            "trajectories": len(paths),
            "max_trajectories": int(config.get("max_trajectories", 80)),
            "max_tokens_per_trajectory": int(
                config.get("max_tokens_per_trajectory", 64)
            ),
            "pair_count": len(pairs),
        },
        "summary": _summary(paths, geometry, alignments),
        "geometry": geometry,
        "alignment": {
            "pairs": alignments,
            "metrics": _alignment_aggregates(alignments),
        },
        "compression": compression_curve(paths, dimensions),
        "basins": basin_summary(paths, int(config.get("basin_clusters", 4))),
        "failures": failure_autopsies(paths),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _summary(paths: list, geometry: list[dict], alignments: list[dict]) -> dict:
    """Build compact headline values for the UI."""
    return {
        "trajectories": len(paths),
        "correct": sum(path.is_correct is True for path in paths),
        "incorrect": sum(path.is_correct is False for path in paths),
        "median_path_length": median(row["path_length"] for row in geometry),
        "median_net_path_ratio": median(row["net_path_ratio"] for row in geometry),
        "median_curvature": median(row["mean_curvature"] for row in geometry),
        "median_cka": (
            median(row["cka"] for row in alignments) if alignments else None
        ),
    }


def _alignment_aggregates(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate pairwise metrics overall and by outcome agreement."""
    metrics = ["dtw", "frechet", "cosine_path_similarity", "procrustes", "cka", "rsa"]
    result: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        same = np.asarray(
            [row[metric] for row in rows if row["same_outcome"]], dtype=float
        )
        different = np.asarray(
            [row[metric] for row in rows if not row["same_outcome"]], dtype=float
        )
        result[metric] = {
            "median": float(np.median(values)) if len(values) else 0.0,
            "same_outcome": float(np.median(same)) if len(same) else 0.0,
            "different_outcome": float(np.median(different)) if len(different) else 0.0,
        }
    return result
