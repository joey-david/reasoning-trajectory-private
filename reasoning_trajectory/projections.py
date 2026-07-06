"""Low-dimensional projections with diagnostics for visual interpretation."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness


def project_3d(
    x: np.ndarray,
    *,
    random_state: int | None = None,
    tsne_perplexity: int = 30,
) -> dict[str, np.ndarray]:
    """Project a feature matrix into PCA and t-SNE three-dimensional spaces."""
    return {
        "pca": PCA(n_components=3, random_state=random_state).fit_transform(x),
        "tsne": TSNE(
            n_components=3,
            perplexity=min(tsne_perplexity, len(x) - 1),
            init="random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(x),
    }


def projection_diagnostics(
    original: np.ndarray,
    projected: np.ndarray,
    *,
    explained_variance: list[float] | None = None,
) -> dict[str, float | list[float] | str]:
    """Measure neighborhood preservation and flag unreliable projections."""
    if len(original) < 3:
        score = 1.0
    else:
        neighbors = min(5, max(1, (len(original) - 1) // 2))
        score = float(trustworthiness(original, projected, n_neighbors=neighbors))
    return {
        "trustworthiness": score,
        "explained_variance": explained_variance or [],
        "warning": (
            "Low-fidelity projection; inspect metrics in the original space."
            if score < 0.85
            else ""
        ),
    }
