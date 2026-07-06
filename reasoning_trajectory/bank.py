"""Load bounded token-level paths from completed generation artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_trajectory.artifacts import (
    evenly_capped,
    load_hidden_states_npz,
    read_generation_rows,
)


@dataclass(slots=True)
class TrajectoryPath:
    """A token-level latent path with the metadata needed by every analysis."""

    trajectory_id: str
    sample_id: str
    seed: int
    is_correct: bool | None
    layer: int
    token_indices: list[int]
    states: np.ndarray


def load_trajectory_bank(
    run_path: str | Path,
    config: dict[str, Any] | None = None,
) -> list[TrajectoryPath]:
    """Load a stratified, bounded bank of final-layer token trajectories."""
    run_path = Path(run_path)
    config = config or {}
    rows = [
        row
        for row in read_generation_rows(run_path)
        if row.get("hidden_states_file")
    ]
    rows = _stratified_cap(rows, int(config.get("max_trajectories", 80)))
    max_tokens = max(3, int(config.get("max_tokens_per_trajectory", 64)))
    requested_layer = config.get("layer")
    paths: list[TrajectoryPath] = []

    for row in rows:
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        if states.shape[0] < 2:
            continue
        layer_col = _layer_column(layers, requested_layer)
        indices = np.linspace(
            0,
            states.shape[0] - 1,
            min(max_tokens, states.shape[0]),
            dtype=int,
        )
        indices = np.unique(indices)
        layer = int(layers[layer_col])
        paths.append(
            TrajectoryPath(
                trajectory_id=f"{row.get('sample_id')}::{row.get('seed')}",
                sample_id=str(row.get("sample_id")),
                seed=int(row.get("seed", 0)),
                is_correct=row.get("is_correct"),
                layer=layer,
                token_indices=indices.astype(int).tolist(),
                states=states[indices, layer_col].astype(np.float32),
            )
        )
    return paths


def _stratified_cap(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Retain evenly spaced correct, incorrect, and unscored rows."""
    if limit <= 0 or len(rows) <= limit:
        return rows
    buckets = [
        [row for row in rows if row.get("is_correct") is value]
        for value in (True, False, None)
    ]
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            if bucket and len(selected) < limit:
                index = round((len(selected) / max(limit - 1, 1)) * (len(bucket) - 1))
                selected.append(bucket.pop(index))
    return selected


def _layer_column(layers: list[int], requested: int | None) -> int:
    """Resolve a decoder-layer identifier to an activation array column."""
    if requested is None:
        return len(layers) - 1
    requested = int(requested)
    if requested in layers:
        return layers.index(requested)
    if requested < 0:
        return requested
    raise ValueError(f"Layer {requested} is absent; available layers: {layers}")
