from __future__ import annotations

from pathlib import Path

import numpy as np

from reasoning_trajectory.core.storage import load_trajectories, save_table
from reasoning_trajectory.core.registry import tool
from .geometry import step_matrix


def compact_pair(a: np.ndarray, b: np.ndarray, max_dims: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Project wide hidden states before pairwise path metrics.

    Step counts are small but hidden widths can be thousands of dimensions; some
    metrics otherwise perform expensive wide SVDs for every pair.
    """
    width = max(a.shape[1], b.shape[1])
    if width <= max_dims:
        return a, b
    rng = np.random.default_rng(0)
    projection = rng.normal(0.0, 1.0 / np.sqrt(max_dims), size=(width, max_dims))
    aa = np.pad(a, ((0, 0), (0, width - a.shape[1])))
    bb = np.pad(b, ((0, 0), (0, width - b.shape[1])))
    return aa @ projection, bb @ projection


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    dp = np.full((len(a) + 1, len(b) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1])


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    ca = np.full((len(a), len(b)), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = np.linalg.norm(a[i] - b[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), d)
        else:
            ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), d)
        return ca[i, j]

    return float(c(len(a) - 1, len(b) - 1))


def cosine_path_similarity(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = np.diff(a, axis=0).reshape(-1), np.diff(b, axis=0).reshape(-1)
    n = min(len(va), len(vb))
    if n == 0:
        return 1.0
    return float(np.dot(va[:n], vb[:n]) / max(np.linalg.norm(va[:n]) * np.linalg.norm(vb[:n]), 1e-12))


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    x, y = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    r = u @ vt
    return float(np.linalg.norm(x @ r - y) / max(n, 1))


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    x, y = a[:n] - a[:n].mean(0), b[:n] - b[:n].mean(0)
    hsic = np.linalg.norm(x.T @ y) ** 2
    return float(hsic / max(np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y), 1e-12))


def rsa_similarity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    da = np.linalg.norm(a[:n, None] - a[None, :n], axis=-1).reshape(-1)
    db = np.linalg.norm(b[:n, None] - b[None, :n], axis=-1).reshape(-1)
    return float(np.corrcoef(da, db)[0, 1]) if da.std() and db.std() else 0.0


def prototype_path(paths: list[np.ndarray]) -> np.ndarray:
    max_len = max(len(p) for p in paths)
    aligned = []
    grid = np.linspace(0, 1, max_len)
    for p in paths:
        src = np.linspace(0, 1, len(p))
        aligned.append(np.vstack([np.interp(grid, src, p[:, d]) for d in range(p.shape[1])]).T)
    return np.mean(aligned, axis=0)


def first_divergence(a: np.ndarray, b: np.ndarray, threshold: float | None = None) -> int:
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n] - b[:n], axis=1)
    threshold = threshold if threshold is not None else float(d.mean() + d.std())
    hits = np.flatnonzero(d > threshold)
    return int(hits[0]) if len(hits) else -1


def alignment_summary(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = compact_pair(a, b)
    return {
        "dtw": dtw_distance(a, b),
        "frechet": frechet_distance(a, b),
        "cosine_path_similarity": cosine_path_similarity(a, b),
        "procrustes": procrustes_distance(a, b),
        "cka": linear_cka(a, b),
        "rsa": rsa_similarity(a, b),
        "first_divergence": first_divergence(a, b),
    }


@tool(
    "alignment",
    "metrics",
    "Compare trajectories with DTW, Frechet, cosine path similarity, Procrustes, CKA, RSA, and divergence points.",
    "rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics",
    "reasoning_trajectory.metrics.alignment.export_alignment",
    "toolkit/docs/tools/alignment.md",
)
def export_alignment(input_path: str | Path, out: str | Path, layer: str | None = None) -> list[dict]:
    trajectories = load_trajectories(input_path)
    matrices = {t.trajectory_id: step_matrix(t, layer) for t in trajectories}
    rows = []
    ids = list(matrices)
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            rows.append({"a": a_id, "b": b_id, **alignment_summary(matrices[a_id], matrices[b_id])})
    save_table(rows, out)
    return rows
