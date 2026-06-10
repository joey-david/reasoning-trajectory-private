from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.common import analysis_path, load_generations
from src.analysis.projection import pca_project
from src.config import run_path


def write_pca_components(
    config: dict[str, Any],
    layer: str | int | None = None,
    n: int = 24,
    max_vectors: int = 20000,
    skip_first: int = 2,
    skip_last: int = 2,
) -> Path:
    import numpy as np

    base = run_path(config)
    rows = load_generations(config)
    selected_layer = str(layer if layer is not None else _default_layer(config, rows))
    matrices = []
    total_vectors = 0
    for row in rows:
        if not row.get("activation_file"):
            continue
        arrays = np.load(base / row["activation_file"])
        if selected_layer in arrays.files:
            x = arrays[selected_layer]

            # Match trajectory analysis: ignore boundary token states.
            lo = max(0, int(skip_first))
            hi = max(lo, x.shape[0] - max(0, int(skip_last)))
            x = x[lo:hi]

            if x.shape[0]:
                matrices.append(x)
                total_vectors += int(x.shape[0])

    values = np.concatenate(matrices, axis=0) if matrices else np.zeros((0, 1), dtype=np.float32)
    if max_vectors and values.shape[0] > int(max_vectors):
        indices = _even_indices(values.shape[0], int(max_vectors))
        values = values[indices]

    result = pca_project(values, max(1, int(n)))
    singular_values = result["singular_values"]
    explained = result["explained_variance_ratio"]
    components = [
        {
            "component": idx + 1,
            "amplitude": float(singular_values[idx]) if idx < len(singular_values) else 0.0,
            "explained_variance_ratio": float(explained[idx]) if idx < len(explained) else 0.0,
        }
        for idx in range(max(1, int(n)))
    ]
    payload = {
        "tool": "pca_components",
        "layer": selected_layer,
        "token_vectors_total": int(total_vectors),
        "token_vectors_used": int(values.shape[0]),
        "max_vectors": int(max_vectors),
        "skip_first": int(skip_first),
        "skip_last": int(skip_last),
        "components": components,
    }
    target = analysis_path(config, f"pca_components_layer{selected_layer}_n{int(n)}.json")
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return target


def _even_indices(size: int, keep: int) -> list[int]:
    if keep >= size:
        return list(range(size))
    if keep <= 1:
        return [0]
    return sorted({round(i * (size - 1) / (keep - 1)) for i in range(keep)})


def _default_layer(config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    layers = [str(layer) for layer in config.get("layers", [])]
    if layers:
        return layers[-1]
    for row in rows:
        if row.get("activation_file"):
            import numpy as np

            arrays = np.load(run_path(config) / row["activation_file"])
            if arrays.files:
                return arrays.files[-1]
    raise ValueError("No activation layers found for this run")
