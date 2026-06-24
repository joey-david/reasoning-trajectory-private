from __future__ import annotations

from typing import Any

import numpy as np

from src.analysis.common import evenly_capped, project_3d

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
    max_plot_steps = int(step_cfg.get("max_plot_steps", 4000))
    indices = evenly_capped(list(range(len(records))), max_plot_steps)
    plot_records = [records[i] for i in indices]
    plot_means = means[np.asarray(indices)]
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
            "points": [point_record(record, coords) for record, coords in zip(plot_records, values)],
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
