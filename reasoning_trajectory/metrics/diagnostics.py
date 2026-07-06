"""Compression, endpoint basin, and nearest-correct failure diagnostics."""

from __future__ import annotations

import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from reasoning_trajectory.bank import TrajectoryPath
from reasoning_trajectory.metrics.alignment import compact_pair, resample_pair
from reasoning_trajectory.projections import projection_diagnostics


def compression_curve(
    paths: list[TrajectoryPath],
    dimensions: list[int],
    max_points: int = 5000,
) -> list[dict]:
    """Fit shared PCA bottlenecks and report retained variance and error."""
    points = np.concatenate([path.states for path in paths])
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        points = points[indices]
    valid_dimensions = sorted(
        {
            min(int(requested), *points.shape)
            for requested in dimensions
            if int(requested) > 0
        }
    )
    if not valid_dimensions:
        return []
    if float(np.var(points, axis=0).sum()) <= 1e-12:
        return [
            {
                "dimensions": dims,
                "explained_variance": 1.0,
                "normalized_reconstruction_error": 0.0,
            }
            for dims in valid_dimensions
        ]
    pca = PCA(n_components=max(valid_dimensions), random_state=0).fit(points)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    rows: list[dict] = []
    for dims in valid_dimensions:
        explained = float(cumulative[dims - 1])
        rows.append(
            {
                "dimensions": dims,
                "explained_variance": explained,
                "normalized_reconstruction_error": max(0.0, 1.0 - explained),
            }
        )
    return rows


def basin_summary(paths: list[TrajectoryPath], clusters: int = 4) -> dict:
    """Cluster endpoints and report size, entropy, and separation."""
    endpoints = np.stack([path.states[-1] for path in paths])
    reduced, projection = _endpoint_projection(endpoints)
    unique_endpoints = len(np.unique(np.round(reduced, decimals=8), axis=0))
    cluster_count = min(max(1, int(clusters)), len(paths), unique_endpoints)
    labels = KMeans(
        n_clusters=cluster_count,
        n_init=10,
        random_state=0,
    ).fit_predict(reduced)
    sizes = np.bincount(labels, minlength=cluster_count)
    probabilities = sizes / max(sizes.sum(), 1)
    entropy = -sum(float(p * math.log2(p)) for p in probabilities if p > 0)
    occupied = len(np.unique(labels))
    silhouette = (
        float(silhouette_score(reduced, labels))
        if 1 < occupied < len(paths)
        else None
    )
    return {
        "cluster_count": cluster_count,
        "sizes": sizes.astype(int).tolist(),
        "entropy_bits": entropy,
        "silhouette": silhouette,
        "projection": projection,
        "assignments": [
            {
                "trajectory_id": path.trajectory_id,
                "sample_id": path.sample_id,
                "is_correct": path.is_correct,
                "cluster": int(label),
            }
            for path, label in zip(paths, labels)
        ],
    }


def failure_autopsies(paths: list[TrajectoryPath]) -> list[dict]:
    """Locate first sustained divergence from each failure's nearest success."""
    correct = [path for path in paths if path.is_correct is True]
    incorrect = [path for path in paths if path.is_correct is False]
    if not correct or not incorrect:
        return []
    endpoints = np.stack([path.states[-1] for path in correct + incorrect])
    endpoint_features = _failure_endpoint_features(endpoints)
    correct_features = endpoint_features[: len(correct)]
    rows: list[dict] = []
    for offset, candidate in enumerate(incorrect):
        eligible = [
            index
            for index, path in enumerate(correct)
            if path.sample_id == candidate.sample_id
        ]
        if not eligible:
            eligible = list(range(len(correct)))
        distances = np.linalg.norm(
            correct_features[eligible] - endpoint_features[len(correct) + offset],
            axis=1,
        )
        reference_index = eligible[int(np.argmin(distances))]
        reference = correct[reference_index]
        point, fraction, divergence = first_divergence(
            reference.states, candidate.states
        )
        rows.append(
            {
                "trajectory_id": candidate.trajectory_id,
                "nearest_correct": reference.trajectory_id,
                "endpoint_distance": float(distances.min()),
                "first_divergence_point": point,
                "first_divergence_fraction": fraction,
                "divergence_score": divergence,
            }
        )
    return rows


def first_divergence(a: np.ndarray, b: np.ndarray) -> tuple[int, float, float]:
    """Find the first of two consecutive above-threshold aligned distances."""
    aa, bb = compact_pair(a, b)
    aa, bb = resample_pair(aa, bb)
    distances = np.linalg.norm(aa - bb, axis=1)
    threshold = float(np.quantile(distances, 0.75))
    for index in range(max(len(distances) - 1, 1)):
        if distances[index] > threshold and (
            index == len(distances) - 1 or distances[index + 1] > threshold
        ):
            return index, index / max(len(distances) - 1, 1), float(distances[index])
    return -1, -1.0, float(distances.max(initial=0.0))


def _endpoint_projection(endpoints: np.ndarray) -> tuple[np.ndarray, dict]:
    """Whiten endpoints in a bounded PCA space and assess a 3D view."""
    if float(np.var(endpoints, axis=0).sum()) <= 1e-12:
        return np.zeros((len(endpoints), 1)), {
            "trustworthiness": 1.0,
            "explained_variance": [1.0],
            "warning": "All sampled endpoints are identical.",
        }
    components = min(16, *endpoints.shape)
    pca = PCA(n_components=components, whiten=True, random_state=0)
    reduced = pca.fit_transform(endpoints)
    visible = reduced[:, : min(3, reduced.shape[1])]
    diagnostics = projection_diagnostics(
        endpoints,
        visible,
        explained_variance=pca.explained_variance_ratio_[:3].astype(float).tolist(),
    )
    return reduced, diagnostics


def _failure_endpoint_features(endpoints: np.ndarray) -> np.ndarray:
    """Produce stable endpoint features even when all endpoints coincide."""
    if float(np.var(endpoints, axis=0).sum()) <= 1e-12:
        return np.zeros((len(endpoints), 1))
    reduced = PCA(
        n_components=min(16, *endpoints.shape),
        random_state=0,
    ).fit_transform(endpoints)
    return StandardScaler().fit_transform(reduced)
