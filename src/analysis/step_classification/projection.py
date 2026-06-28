"""Project step mean vectors into browser-ready three-dimensional plot payloads."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.analysis.common import evenly_capped, project_3d
from src.analysis.step_classification.features import StepMatrices


def projection_payloads(
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    layer: int,
    cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build capped PCA and t-SNE payloads for one layer's step records.

    Args:
        records: Step metadata aligned with the vector matrices.
        vectors: Full step feature matrices.
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
    plot_means = vectors.means[np.asarray(indices)]
    projections = project_3d(
        plot_means,
        random_state=random_state,
        tsne_perplexity=int(step_cfg.get("tsne_perplexity", 30)),
    )
    return {
        name: {
            "plot_type": "step_classification",
            "method": name,
            "layer": layer,
            "points": [
                point_record(record, coords)
                for record, coords in zip(plot_records, values)
            ],
        }
        for name, values in projections.items()
    }


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
