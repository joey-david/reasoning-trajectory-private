from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def projection_payloads(
    records: list[dict[str, Any]],
    means: np.ndarray,
    layer: int,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if len(records) < 3:
        return {}
    step_cfg = cfg.get("step_classification", {})
    random_state = int(step_cfg.get("random_state", 42))
    perplexity = min(int(step_cfg.get("tsne_perplexity", 30)), len(records) - 1)
    projections = {
        "pca": PCA(n_components=3, random_state=random_state).fit_transform(means),
        "tsne": TSNE(
            n_components=3,
            perplexity=perplexity,
            init="random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(means),
    }
    return {
        name: {
            "plot_type": "step_classification",
            "method": name,
            "layer": layer,
            "points": [point_record(record, coords) for record, coords in zip(records, values)],
        }
        for name, values in projections.items()
    }


def point_record(record: dict[str, Any], coords: np.ndarray) -> dict[str, Any]:
    out = dict(record)
    out.update(
        {
            "x": round(float(coords[0]), 6),
            "y": round(float(coords[1]), 6),
            "z": round(float(coords[2]), 6),
        }
    )
    return out

