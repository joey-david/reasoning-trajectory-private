"""Original-space geometry summaries for token-level latent paths."""

from __future__ import annotations

import numpy as np

from reasoning_trajectory.bank import TrajectoryPath


def trajectory_geometry(trajectory: TrajectoryPath) -> dict:
    """Summarize speed, turning, spread, and commitment along one path."""
    x = trajectory.states.astype(np.float64)
    deltas = np.diff(x, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    path_length = float(speeds.sum())
    endpoint = float(np.linalg.norm(x[-1] - x[0]))
    curvature = _curvature(deltas)
    direction = x[-1] - x[0]
    direction_norm = np.linalg.norm(direction)
    consistency = (
        deltas @ direction
        / np.maximum(np.linalg.norm(deltas, axis=1) * direction_norm, 1e-12)
        if len(deltas)
        else np.array([1.0])
    )
    return {
        "trajectory_id": trajectory.trajectory_id,
        "sample_id": trajectory.sample_id,
        "seed": trajectory.seed,
        "is_correct": trajectory.is_correct,
        "layer": trajectory.layer,
        "points": len(x),
        "path_length": path_length,
        "endpoint_distance": endpoint,
        "net_path_ratio": endpoint / max(path_length, 1e-12),
        "mean_velocity": float(speeds.mean()) if len(speeds) else 0.0,
        "max_velocity": float(speeds.max()) if len(speeds) else 0.0,
        "peak_share": float(speeds.max() / max(path_length, 1e-12))
        if len(speeds)
        else 0.0,
        "effective_width": _effective_width(speeds),
        "mean_acceleration": _mean_acceleration(deltas),
        "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
        "directional_consistency": float(consistency.mean()),
        "mean_distance_to_final": float(np.linalg.norm(x - x[-1], axis=1).mean()),
        "commitment_fraction": _commitment_fraction(x),
        "looping_score": _looping_score(x),
    }


def _curvature(deltas: np.ndarray) -> np.ndarray:
    """Return one-minus-cosine turning angles between consecutive moves."""
    if len(deltas) < 2:
        return np.array([])
    numerator = np.sum(deltas[:-1] * deltas[1:], axis=1)
    denominator = np.maximum(
        np.linalg.norm(deltas[:-1], axis=1)
        * np.linalg.norm(deltas[1:], axis=1),
        1e-12,
    )
    return 1.0 - numerator / denominator


def _effective_width(speeds: np.ndarray) -> float:
    """Return participation-ratio width as a fraction of path transitions."""
    if not len(speeds) or speeds.sum() == 0:
        return 0.0
    weights = speeds / speeds.sum()
    return float(1.0 / np.sum(weights**2) / len(weights))


def _mean_acceleration(deltas: np.ndarray) -> float:
    """Return mean norm of the second path difference."""
    if len(deltas) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(deltas, axis=0), axis=1).mean())


def _commitment_fraction(x: np.ndarray) -> float:
    """Find the first point after which distance to the endpoint stays halved."""
    if len(x) < 2:
        return 0.0
    distances = np.linalg.norm(x - x[-1], axis=1)
    threshold = distances[0] / 2
    for index in range(len(x)):
        if np.all(distances[index:] <= threshold):
            return index / max(len(x) - 1, 1)
    return 1.0


def _looping_score(x: np.ndarray) -> float:
    """Measure non-adjacent returns near earlier latent states."""
    if len(x) < 3:
        return 0.0
    scale = max(float(np.median(np.linalg.norm(np.diff(x, axis=0), axis=1))), 1e-12)
    hits = total = 0
    for i in range(len(x) - 2):
        distances = np.linalg.norm(x[i + 2 :] - x[i], axis=1)
        hits += int(np.sum(distances < scale))
        total += len(distances)
    return hits / max(total, 1)
