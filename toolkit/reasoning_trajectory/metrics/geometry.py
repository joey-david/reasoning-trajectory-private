from __future__ import annotations

from pathlib import Path

import numpy as np

from reasoning_trajectory.core.storage import load_trajectories, save_table
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.schema import Trajectory


def step_matrix(trajectory: Trajectory, layer: str | None = None) -> np.ndarray:
    rows = []
    for step in trajectory.steps:
        if not step.hidden_states:
            continue
        key = layer or sorted(step.hidden_states.keys(), key=str)[-1]
        rows.append(np.asarray(step.hidden_states[key], dtype=float).reshape(-1))
    if not rows:
        raise ValueError(f"trajectory {trajectory.trajectory_id} has no hidden states")
    width = max(len(r) for r in rows)
    return np.vstack([np.pad(r, (0, width - len(r))) for r in rows])


def trajectory_geometry(trajectory: Trajectory, layer: str | None = None, reference_sets: dict[str, list[np.ndarray]] | None = None) -> dict:
    x = step_matrix(trajectory, layer)
    deltas = np.diff(x, axis=0)
    speeds = np.linalg.norm(deltas, axis=1) if len(x) > 1 else np.array([])
    acc = np.diff(deltas, axis=0) if len(deltas) > 1 else np.empty((0, x.shape[1]))
    curvature = []
    for a, b in zip(deltas, deltas[1:]):
        denom = max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
        curvature.append(float(1 - np.dot(a, b) / denom))
    direction = x[-1] - x[0]
    consistency = [float(np.dot(d, direction) / max(np.linalg.norm(d) * np.linalg.norm(direction), 1e-12)) for d in deltas]
    rows = {
        "trajectory_id": trajectory.trajectory_id,
        "problem_id": trajectory.problem_id,
        "final_correct": trajectory.final_correct,
        "n_steps": len(x),
        "path_length": float(speeds.sum()),
        "endpoint_distance": float(np.linalg.norm(x[-1] - x[0])),
        "mean_velocity": float(speeds.mean()) if len(speeds) else 0.0,
        "max_velocity": float(speeds.max()) if len(speeds) else 0.0,
        "mean_acceleration": float(np.linalg.norm(acc, axis=1).mean()) if len(acc) else 0.0,
        "mean_curvature": float(np.mean(curvature)) if curvature else 0.0,
        "torsion": float(_torsion(x)),
        "directional_consistency": float(np.mean(consistency)) if consistency else 1.0,
        "distance_to_final_state": float(np.linalg.norm(x - x[-1], axis=1).mean()),
        "commitment_time": int(np.argmax(np.linalg.norm(x - x[-1], axis=1) < np.linalg.norm(x[0] - x[-1]) / 2)) if len(x) > 1 else 0,
        "drift_score": float(max(0.0, speeds.sum() - np.linalg.norm(x[-1] - x[0]))),
        "looping_score": float(_looping_score(x)),
        "branch_entropy": 0.0,
    }
    if reference_sets:
        for name, refs in reference_sets.items():
            rows[f"distance_to_nearest_{name}"] = float(min(np.linalg.norm(x[-1] - r[-1]) for r in refs)) if refs else float("nan")
    return rows


def _torsion(x: np.ndarray) -> float:
    if len(x) < 4 or x.shape[1] < 3:
        return 0.0
    vals = []
    for i in range(len(x) - 3):
        a, b, c = x[i + 1] - x[i], x[i + 2] - x[i + 1], x[i + 3] - x[i + 2]
        vals.append(abs(np.linalg.det(np.vstack([a[:3], b[:3], c[:3]]))) / max(np.linalg.norm(np.cross(a[:3], b[:3])) ** 2, 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def _looping_score(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    repeated = 0
    for i in range(len(x)):
        for j in range(i + 2, len(x)):
            repeated += np.linalg.norm(x[i] - x[j]) < 0.1 * max(np.linalg.norm(x[-1] - x[0]), 1e-12)
    return float(repeated)


@tool(
    "geometry",
    "metrics",
    "Export path length, velocity, acceleration, curvature, commitment, drift, and repetition metrics.",
    "rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics",
    "reasoning_trajectory.metrics.geometry.export_geometry",
    "toolkit/docs/tools/geometry.md",
    dashboard=True,
)
def export_geometry(input_path: str | Path, out: str | Path, layer: str | None = None) -> list[dict]:
    trajectories = load_trajectories(input_path)
    correct = [step_matrix(t, layer) for t in trajectories if t.final_correct is True and any(s.hidden_states for s in t.steps)]
    incorrect = [step_matrix(t, layer) for t in trajectories if t.final_correct is False and any(s.hidden_states for s in t.steps)]
    rows = [trajectory_geometry(t, layer, {"correct_trajectory": correct, "incorrect_trajectory": incorrect}) for t in trajectories]
    save_table(rows, out)
    return rows
