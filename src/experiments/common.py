"""Shared rollout selection, latent statistics, and grouped evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import read_generation_rows


def balanced_generation_rows(
    run_path: Path,
    *,
    per_sample: int = 10,
    require_hidden_states: bool = True,
    require_scored: bool = True,
) -> list[dict[str, Any]]:
    """Select a deterministic, uniformly capped set without changing raw data."""
    selected: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    rows = sorted(
        read_generation_rows(run_path),
        key=lambda row: (str(row.get("sample_id")), int(row.get("seed", 0))),
    )
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if counts[sample_id] >= per_sample:
            continue
        if require_hidden_states and not row.get("hidden_states_file"):
            continue
        if require_scored and row.get("is_correct") is None:
            continue
        selected.append(row)
        counts[sample_id] += 1
    return selected


def latent_deltas(states: np.ndarray) -> np.ndarray:
    """Return token-to-token state changes with a zero vector at token zero."""
    states = np.asarray(states, dtype=np.float32)
    deltas = np.zeros_like(states)
    if len(states) > 1:
        deltas[1:] = states[1:] - states[:-1]
    return deltas


def robust_spike_indices(
    magnitudes: np.ndarray,
    *,
    z_threshold: float = 3.0,
    min_distance: int = 3,
) -> np.ndarray:
    """Find separated local maxima above a median absolute-deviation threshold."""
    values = np.asarray(magnitudes, dtype=np.float32)
    if len(values) < 3:
        return np.empty(0, dtype=np.int32)
    core = values[1:]
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)))
    scale = max(1.4826 * mad, float(np.std(core)) * 0.1, 1e-8)
    threshold = median + z_threshold * scale
    candidates = np.flatnonzero(
        (values >= threshold)
        & (values >= np.r_[values[0], values[:-1]])
        & (values >= np.r_[values[1:], values[-1]])
    )
    if not len(candidates):
        return candidates.astype(np.int32)

    retained: list[int] = []
    for idx in candidates[np.argsort(values[candidates])[::-1]]:
        if all(abs(int(idx) - existing) >= min_distance for existing in retained):
            retained.append(int(idx))
    return np.asarray(sorted(retained), dtype=np.int32)


def nearest_distance(indices: np.ndarray, target: int) -> int | None:
    """Return the distance from ``target`` to the nearest index."""
    if not len(indices):
        return None
    return int(np.min(np.abs(indices - int(target))))


def percentile_rank(values: np.ndarray, value: float) -> float:
    """Return the empirical percentile of a value in ``values``."""
    values = np.asarray(values)
    if not len(values):
        return float("nan")
    return float(np.mean(values <= value))


def prefix_checkpoints(length: int) -> dict[int, int]:
    """Map 25/50/75 percent checkpoints to valid zero-based token indices."""
    return {
        percent: min(max(int(np.ceil(length * percent / 100)) - 1, 0), length - 1)
        for percent in (25, 50, 75)
    }
