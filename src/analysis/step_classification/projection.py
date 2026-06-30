"""Project step mean vectors into browser-ready three-dimensional plot payloads."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.analysis.common import evenly_capped, project_3d
from src.analysis.step_classification.features import StepFeature


def projection_payloads(
    records: list[dict[str, Any]],
    features: list[StepFeature],
    layer: int,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build full PCA and bounded t-SNE payloads for step records.

    Args:
        records: Step metadata aligned with ``features``.
        features: Full step features whose means are projected.
        layer: Decoder-layer ID represented by the records.
        cfg: Projection seed, point cap, and t-SNE options.

    Returns:
        Plot payloads keyed by projection method, or an empty mapping with
        fewer than three records.
    """
    if len(records) < 3:
        return {}
    step_cfg = cfg.get("step_classification", {})
    random_state = int(step_cfg.get("random_state", 42))
    max_plot_steps = int(step_cfg.get("max_plot_steps", 4000))
    indices = evenly_capped(list(range(len(records))), max_plot_steps)
    plot_records = [records[i] for i in indices]
    plot_means = np.stack([features[i].mean for i in indices]).astype(np.float32)
    tsne = project_3d(
        plot_means,
        random_state=random_state,
        tsne_perplexity=int(step_cfg.get("tsne_perplexity", 30)),
    )["tsne"]
    pca = PCA(n_components=3, random_state=random_state).fit(plot_means)
    pca_coords = transform_step_means(features, pca)
    return {
        "pca": {
            "plot_type": "step_classification",
            "method": "pca",
            "layer": layer,
            "max_points": max_plot_steps,
            "source_points": len(records),
            "sampled": False,
            "points": [
                point_record(record, coords)
                for record, coords in zip(records, pca_coords)
            ],
        },
        "tsne": {
            "plot_type": "step_classification",
            "method": "tsne",
            "layer": layer,
            "max_points": max_plot_steps,
            "source_points": len(records),
            "sampled": len(plot_records) < len(records),
            "points": [
                point_record(record, coords)
                for record, coords in zip(plot_records, tsne)
            ],
        },
    }


def transform_step_means(
    features: list[StepFeature],
    pca: PCA,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Transform every step mean through a PCA fitted on a bounded sample."""
    chunks = (
        np.stack(
            [feature.mean for feature in features[start : start + chunk_size]]
        ).astype(np.float32)
        for start in range(0, len(features), chunk_size)
    )
    return np.concatenate([pca.transform(chunk) for chunk in chunks])


def point_record(record: dict[str, Any], coords: np.ndarray) -> dict[str, Any]:
    """Add projected coordinates to a copy of one step record.

    Args:
        record: Source step metadata.
        coords: Three-element projected coordinate vector.

    Returns:
        A copied record with rounded ``x``, ``y``, and ``z`` fields.
    """
    out = dict(record)
    out.update(
        {
            "x": round(float(coords[0]), 6),
            "y": round(float(coords[1]), 6),
            "z": round(float(coords[2]), 6),
        }
    )
    return out
