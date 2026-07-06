"""Pairwise path alignment metrics for trajectories of unequal length."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from reasoning_trajectory.bank import TrajectoryPath


def alignment_summary(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compare two paths with complementary geometric similarity measures."""
    a, b = compact_pair(a, b)
    aa, bb = resample_pair(a, b)
    return {
        "dtw": dtw_distance(a, b),
        "frechet": frechet_distance(a, b),
        "cosine_path_similarity": cosine_path_similarity(aa, bb),
        "procrustes": procrustes_distance(aa, bb),
        "cka": linear_cka(aa, bb),
        "rsa": rsa_similarity(aa, bb),
    }


def trajectory_pairs(
    paths: list[TrajectoryPath],
    max_pairs: int = 200,
) -> list[tuple[TrajectoryPath, TrajectoryPath]]:
    """Prefer within-question rollout pairs and cap them deterministically."""
    same_problem = [
        pair for pair in combinations(paths, 2) if pair[0].sample_id == pair[1].sample_id
    ]
    candidates = same_problem or list(combinations(paths, 2))
    if len(candidates) <= max_pairs:
        return candidates
    keep = np.linspace(0, len(candidates) - 1, max_pairs, dtype=int)
    return [candidates[int(index)] for index in keep]


def compact_pair(
    a: np.ndarray,
    b: np.ndarray,
    max_dims: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one deterministic random projection before costly pair metrics."""
    width = max(a.shape[1], b.shape[1])
    aa = np.pad(a, ((0, 0), (0, width - a.shape[1])))
    bb = np.pad(b, ((0, 0), (0, width - b.shape[1])))
    if width <= max_dims:
        return aa, bb
    projection = np.random.default_rng(0).normal(
        0.0, 1.0 / np.sqrt(max_dims), size=(width, max_dims)
    )
    return aa @ projection, bb @ projection


def resample_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample both paths onto a shared normalized timeline."""
    length = max(2, min(max(len(a), len(b)), 64))
    grid = np.linspace(0.0, 1.0, length)
    return _resample(a, grid), _resample(b, grid)


def _resample(path: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate each latent dimension over normalized path progress."""
    source = np.linspace(0.0, 1.0, len(path))
    return np.vstack([np.interp(grid, source, path[:, dim]) for dim in range(path.shape[1])]).T


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized dynamic time-warping distance."""
    dp = np.full((len(a) + 1, len(b) + 1), np.inf)
    dp[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[-1, -1] / max(len(a), len(b), 1))


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute discrete Frechet distance without recursive stack growth."""
    values = np.empty((len(a), len(b)), dtype=float)
    for i in range(len(a)):
        for j in range(len(b)):
            distance = np.linalg.norm(a[i] - b[j])
            if i == 0 and j == 0:
                values[i, j] = distance
            elif i == 0:
                values[i, j] = max(values[i, j - 1], distance)
            elif j == 0:
                values[i, j] = max(values[i - 1, j], distance)
            else:
                values[i, j] = max(
                    min(values[i - 1, j], values[i - 1, j - 1], values[i, j - 1]),
                    distance,
                )
    return float(values[-1, -1])


def cosine_path_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compare flattened movement directions across aligned paths."""
    va, vb = np.diff(a, axis=0).ravel(), np.diff(b, axis=0).ravel()
    return float(np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Measure path mismatch after optimal orthogonal rotation."""
    x, y = a - a.mean(0), b - b.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    return float(np.linalg.norm(x @ (u @ vt) - y) / np.sqrt(max(len(x), 1)))


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Compute centered linear CKA similarity between aligned path states."""
    x, y = a - a.mean(0), b - b.mean(0)
    numerator = np.linalg.norm(x.T @ y) ** 2
    denominator = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    return float(numerator / max(denominator, 1e-12))


def rsa_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlate each path's pairwise-distance structure."""
    da = np.linalg.norm(a[:, None] - a[None, :], axis=-1).ravel()
    db = np.linalg.norm(b[:, None] - b[None, :], axis=-1).ravel()
    return float(np.corrcoef(da, db)[0, 1]) if da.std() and db.std() else 0.0
