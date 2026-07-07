"""Learn low-rank object and lexical controls from pooled residual states."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def fit_group_projection(
    vectors: np.ndarray,
    labels: np.ndarray,
    *,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an orthonormal, within-class-scaled between-class subspace."""
    x = np.asarray(vectors, dtype=np.float64)
    mean = x.mean(axis=0)
    centered = x - mean
    by_label: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    for vector, label in zip(centered, labels, strict=True):
        by_label[str(label)].append(vector)
    centroids = np.stack(
        [np.mean(by_label[label], axis=0) for label in sorted(by_label)]
    )
    residuals = np.concatenate(
        [
            np.stack(values) - np.mean(values, axis=0)
            for values in by_label.values()
            if len(values) > 1
        ],
        axis=0,
    )
    within_scale = np.sqrt(np.var(residuals, axis=0) + 1e-5)
    discriminative = centroids / within_scale
    _, singular_values, right = np.linalg.svd(
        discriminative - discriminative.mean(axis=0),
        full_matrices=False,
    )
    numerical_rank = int(np.sum(singular_values > singular_values[0] * 1e-6))
    supervised_dimension = max(1, min(max_dim, numerical_rank, right.shape[0]))
    weighted = right[:supervised_dimension] / within_scale
    basis, _ = np.linalg.qr(weighted.T)
    basis_rows = basis.T
    return mean.astype(np.float32), basis_rows.astype(np.float32)


def fit_pca_projection(
    vectors: np.ndarray,
    *,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an orthonormal PCA baseline."""
    x = np.asarray(vectors, dtype=np.float64)
    mean = x.mean(axis=0)
    _, singular_values, right = np.linalg.svd(x - mean, full_matrices=False)
    rank = int(np.sum(singular_values > singular_values[0] * 1e-6))
    dimension = max(1, min(max_dim, rank, right.shape[0]))
    return mean.astype(np.float32), right[:dimension].astype(np.float32)


def random_projection(
    input_dim: int,
    output_dim: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return a reproducible orthonormal random control basis."""
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.normal(size=(input_dim, output_dim)))
    return basis.T.astype(np.float32)


def project(vectors: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Center and project row vectors."""
    return (np.asarray(vectors, dtype=np.float32) - mean) @ basis.T


def fit_projection_bundle(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
    train_indices: np.ndarray,
    *,
    max_dim: int,
) -> dict[str, np.ndarray]:
    """Fit object, PCA, random, and lexical-family subspaces."""
    train = vectors[train_indices]
    graph_labels = np.asarray(
        [records[index]["canonical_graph_id"] for index in train_indices], dtype=str
    )
    lexical_labels = np.asarray(
        [
            records[index]["surface"]["lexical_family"]
            for index in train_indices
        ],
        dtype=str,
    )
    object_mean, object_basis = fit_group_projection(
        train, graph_labels, max_dim=max_dim
    )
    pca_mean, pca_basis = fit_pca_projection(
        train, max_dim=object_basis.shape[0]
    )
    lexical_mean, lexical_basis = fit_group_projection(
        train, lexical_labels, max_dim=min(max_dim, len(set(lexical_labels)) - 1)
    )
    return {
        "object_mean": object_mean,
        "object_basis": object_basis,
        "pca_mean": pca_mean,
        "pca_basis": pca_basis,
        "random_mean": object_mean,
        "random_basis": random_projection(
            train.shape[1], object_basis.shape[0], seed=42
        ),
        "lexical_mean": lexical_mean,
        "lexical_basis": lexical_basis,
        "object_supervised_rank": np.asarray(
            min(max_dim, max(len(set(graph_labels)) - 1, 1)),
            dtype=np.int32,
        ),
    }
