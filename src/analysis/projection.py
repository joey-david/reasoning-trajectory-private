from __future__ import annotations

from typing import Any


def pca_project(values: Any, dimensions: int = 3) -> dict[str, Any]:
    """Project token vectors with PCA and always return exactly `dimensions` columns.

    This uses sklearn's randomized PCA path for large matrices. The old full SVD path
    was exact but became painfully slow and memory-heavy for tens of thousands of
    4096D activation vectors.
    """
    import numpy as np
    from sklearn.decomposition import PCA

    x = np.asarray(values, dtype=np.float32)
    dims = max(1, int(dimensions))
    if x.ndim != 2:
        raise ValueError("PCA input must be a 2D matrix")
    if x.shape[0] == 0:
        return _empty_projection(dims)

    use_dims = min(dims, x.shape[0], x.shape[1])
    if x.shape[0] == 1 or use_dims == 0:
        return {
            "coords": np.zeros((x.shape[0], dims), dtype=np.float32).tolist(),
            "explained_variance_ratio": [0.0] * dims,
            "singular_values": [0.0] * dims,
            "dimensions": dims,
        }

    pca = PCA(n_components=use_dims, svd_solver="randomized", random_state=0)
    coords = pca.fit_transform(x)
    if use_dims < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - use_dims)))

    return {
        "coords": coords.astype(np.float32).tolist(),
        "explained_variance_ratio": _pad_float_list(pca.explained_variance_ratio_.tolist(), dims),
        "singular_values": _pad_float_list(pca.singular_values_.tolist(), dims),
        "dimensions": dims,
    }


def tsne_project(values: Any, dimensions: int = 3, max_points: int = 3000) -> dict[str, Any]:
    """Project token vectors with t-SNE, with a hard point cap.

    t-SNE is useful for a qualitative view but scales badly. The caller should
    normally sample with a large interval first; this cap is a second safety rail.
    """
    import numpy as np
    from sklearn.manifold import TSNE

    x = np.asarray(values, dtype=np.float32)
    dims = max(1, int(dimensions))
    if x.ndim != 2:
        raise ValueError("t-SNE input must be a 2D matrix")
    if x.shape[0] < 4:
        result = pca_project(x, dims)
        result["method_actual"] = "pca"
        result["warning"] = "t-SNE needs at least four points; used PCA."
        return result
    if x.shape[0] > int(max_points):
        result = pca_project(x, dims)
        result["method_actual"] = "pca"
        result["warning"] = f"t-SNE skipped for {x.shape[0]} points; use interval>=64 or max_points<={max_points}. Used PCA instead."
        return result

    perplexity = min(30, max(2, (x.shape[0] - 1) // 3))
    coords = TSNE(
        n_components=dims,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=0,
    ).fit_transform(x)
    return {
        "coords": coords.astype(np.float32).tolist(),
        "explained_variance_ratio": [],
        "singular_values": [],
        "method_actual": "tsne",
        "dimensions": dims,
    }


def _empty_projection(dims: int) -> dict[str, Any]:
    return {
        "coords": [],
        "explained_variance_ratio": [0.0] * dims,
        "singular_values": [0.0] * dims,
        "dimensions": dims,
    }


def _pad_float_list(values: list[float], size: int) -> list[float]:
    return [float(value) for value in values[:size]] + [0.0] * max(0, size - len(values))
