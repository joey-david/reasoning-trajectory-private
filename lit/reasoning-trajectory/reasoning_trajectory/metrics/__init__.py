from __future__ import annotations

import numpy as np


def entropy(labels) -> float:
    values, counts = np.unique(labels, return_counts=True)
    p = counts / max(counts.sum(), 1)
    return float(-(p * np.log2(p + 1e-12)).sum())


def cluster_paths(points: np.ndarray, n_clusters: int = 3, seed: int = 0) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(points)


def divergence_detector(reference: np.ndarray, candidate: np.ndarray, threshold: float | None = None) -> dict:
    from .alignment import first_divergence

    idx = first_divergence(reference, candidate, threshold)
    return {"diverged": idx >= 0, "first_divergence": idx}


__all__ = ["cluster_paths", "divergence_detector", "entropy"]
