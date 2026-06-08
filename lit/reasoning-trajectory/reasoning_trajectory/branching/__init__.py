from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.storage import load_trajectories
from reasoning_trajectory.metrics.geometry import step_matrix
from reasoning_trajectory.metrics import entropy


def cluster_trajectories(endpoints: np.ndarray, n_clusters: int = 3, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=min(n_clusters, len(endpoints)), random_state=seed, n_init="auto").fit_predict(endpoints)


def commitment_time(path: np.ndarray, prototypes: dict[str, np.ndarray]) -> int:
    labels = [min(prototypes, key=lambda k: np.linalg.norm(point - prototypes[k])) for point in path]
    for i, label in enumerate(labels):
        if all(x == label for x in labels[i:]):
            return i
    return len(path) - 1


def basin_summary(labels) -> dict:
    return {"branch_entropy": entropy(labels), "basin_sizes": {str(k): int((np.asarray(labels) == k).sum()) for k in set(labels)}}


def branch_tree(step_labels: list[list[str]]) -> dict:
    root: dict = {}
    for path in step_labels:
        node = root
        for label in path:
            node = node.setdefault(str(label), {})
    return root


def nearest_valid_index(candidate: np.ndarray, valid_paths: list[np.ndarray]) -> int:
    return int(np.argmin([np.linalg.norm(candidate[-1] - path[-1]) for path in valid_paths]))


@tool(
    "basins",
    "branching",
    "Cluster trajectory endpoints, estimate basin sizes, branch entropy, and branch trees.",
    "rt basins --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/basins.json",
    "reasoning_trajectory.branching.export_basins",
    "toolkit/docs/tools/basins.md",
    dashboard=True,
)
def export_basins(input_path: str | Path, out: str | Path, n_clusters: int = 3, layer: str | None = None) -> dict:
    trajectories = load_trajectories(input_path)
    endpoints = np.vstack([step_matrix(t, layer)[-1] for t in trajectories])
    labels = cluster_trajectories(endpoints, n_clusters)
    tree = branch_tree([[step.step_id for step in t.steps] for t in trajectories])
    result = {"labels": labels.astype(int).tolist(), "tree": tree, **basin_summary(labels)}
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


__all__ = ["basin_summary", "branch_tree", "cluster_trajectories", "commitment_time", "export_basins", "nearest_valid_index"]
