from __future__ import annotations

import numpy as np


def projection_quality(original: np.ndarray, projected: np.ndarray) -> dict[str, float | str]:
    original = np.asarray(original, dtype=float)
    projected = np.asarray(projected, dtype=float)
    if len(original) < 3:
        return {"trustworthiness": 1.0, "warning": "too few points for robust projection diagnostics"}
    try:
        from sklearn.manifold import trustworthiness
        trust = float(trustworthiness(original, projected, n_neighbors=min(5, len(original) - 1)))
    except Exception:
        trust = float("nan")
    warning = "low-fidelity projection; do not use as core scientific evidence" if trust == trust and trust < 0.85 else ""
    return {"trustworthiness": trust, "warning": warning}


def project(points: np.ndarray, method: str = "pca", dims: int = 3, seed: int = 0) -> tuple[np.ndarray, dict]:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError(f"expected [n,d] points, got {points.shape}")
    dims = min(dims, points.shape[0], points.shape[1])
    if method == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=dims, random_state=seed)
        projected = model.fit_transform(points)
        diag = {"explained_variance": model.explained_variance_ratio_.astype(float).tolist()}
    elif method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise ImportError("UMAP projection requires optional dependency umap-learn") from exc
        projected = umap.UMAP(n_components=dims, random_state=seed).fit_transform(points)
        diag = {}
    elif method == "tsne":
        from sklearn.manifold import TSNE
        projected = TSNE(n_components=dims, random_state=seed, perplexity=max(1, min(30, len(points) - 1))).fit_transform(points)
        diag = {}
    else:
        raise ValueError(f"unknown projection method: {method}")
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    diag.update(projection_quality(points, projected[:, :dims]))
    diag["method"] = method
    return projected[:, :3], diag
