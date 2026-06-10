from __future__ import annotations

from typing import Any


def pca_project(values: Any, dimensions: int = 3) -> dict[str, Any]:
    """Project token vectors with PCA and always return exactly `dimensions` columns."""
    import numpy as np

    x = np.asarray(values, dtype=float)
    dims = max(1, int(dimensions))
    if x.ndim != 2:
        raise ValueError("PCA input must be a 2D matrix")
    if x.shape[0] == 0:
        return _empty_projection(dims)

    centered = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        return {
            "coords": np.zeros((1, dims), dtype=float).tolist(),
            "explained_variance_ratio": [0.0] * dims,
            "singular_values": [0.0] * dims,
            "dimensions": dims,
        }

    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    use_dims = min(dims, vh.shape[0])
    coords = centered @ vh[:use_dims].T
    if use_dims < dims:
        coords = np.pad(coords, ((0, 0), (0, dims - use_dims)))

    power = singular**2
    total = float(power.sum())
    explained = (power[:use_dims] / total).tolist() if total else [0.0] * use_dims
    explained = _pad_float_list(explained, dims)
    singular_values = _pad_float_list(singular[:use_dims].tolist(), dims)

    return {
        "coords": coords.tolist(),
        "explained_variance_ratio": explained,
        "singular_values": singular_values,
        "dimensions": dims,
    }


def tsne_project(values: Any, dimensions: int = 3) -> dict[str, Any]:
    """Project token vectors with t-SNE, falling back to PCA when too few points exist."""
    import numpy as np
    from sklearn.manifold import TSNE

    x = np.asarray(values, dtype=float)
    dims = max(1, int(dimensions))
    if x.ndim != 2:
        raise ValueError("t-SNE input must be a 2D matrix")
    if x.shape[0] < 4:
        result = pca_project(x, dims)
        result["method_actual"] = "pca"
        result["warning"] = "t-SNE needs at least four points; used PCA."
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
        "coords": coords.tolist(),
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
