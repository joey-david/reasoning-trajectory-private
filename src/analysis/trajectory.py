from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.analysis.common import analysis_path, load_generations, selected_token_indices
from src.analysis.projection import pca_project, tsne_project
from src.config import run_path


def write_trajectory_projection(
    config: dict[str, Any],
    layer: str | int | None = None,
    interval: int = 16,
    method: str = "pca",
    max_points: int = 12000,
    skip_first: int = 2,
    skip_last: int = 2,
) -> Path:
    import numpy as np

    base = run_path(config)
    rows = load_generations(config)
    vectors = []
    points = []
    selected_layer = str(layer if layer is not None else _default_layer(config, rows))
    for row in rows:
        if not row.get("activation_file"):
            continue
        arrays = np.load(base / row["activation_file"])
        if selected_layer not in arrays.files:
            continue
        values = arrays[selected_layer]
        indices = selected_token_indices(row, values.shape[0], interval)

        # Boundary hidden states are often outliers:
        # - first generated token = prompt-to-generation transition
        # - final token = EOS/final-format/truncation-adjacent state
        lo = max(0, int(skip_first))
        hi = max(lo, values.shape[0] - max(0, int(skip_last)))
        indices = [idx for idx in indices if lo <= idx < hi]
        if not indices:
            continue

        for index in indices:
            vectors.append(values[index])
            points.append({
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "token_index": index,
                "success": bool(row.get("success", False)),
                "predicted_answer": row.get("predicted_answer", ""),
                "expected_answer": row.get("expected_answer", ""),
                "is_start": index == indices[0],
                "is_end": index == indices[-1],
            })

    original_points = len(points)
    if max_points and len(points) > int(max_points):
        keep = _even_indices(len(points), int(max_points))
        vectors = [vectors[i] for i in keep]
        points = [points[i] for i in keep]

    warning = ""
    if original_points != len(points):
        warning = f"Displayed {len(points)} evenly sampled points from {original_points}; increase interval or max_points for a denser view."

    projection = {"coords": [], "method_actual": method}
    if vectors:
        projection = _project(vectors, method)
        coords = projection.pop("coords")
        for point, coord in zip(points, coords):
            point["x"], point["y"], point["z"] = [float(value) for value in coord[:3]]
        if projection.get("warning"):
            warning = f"{warning} {projection['warning']}".strip()

    actual_method = projection.get("method_actual", method) if vectors else method
    payload = {
        "tool": "trajectory_projection",
        "layer": selected_layer,
        "interval": int(interval),
        "max_points": int(max_points),
        "skip_first": int(skip_first),
        "skip_last": int(skip_last),
        "original_points": int(original_points),
        "displayed_points": int(len(points)),
        "method_requested": method,
        "method_actual": actual_method,
        "dimensions": 3,
        "coordinate_names": _coordinate_names(actual_method),
        "explained_variance_ratio": projection.get("explained_variance_ratio", []) if vectors else [],
        "warning": warning,
        "points": points,
    }
    target = analysis_path(config, f"trajectory_projection_layer{selected_layer}_i{int(interval)}_{method}.json")
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return target


def _project(vectors: list[Any], method: str) -> dict[str, Any]:
    method = method.lower().strip()
    if method == "tsne":
        try:
            return tsne_project(vectors, 3)
        except Exception as exc:
            result = pca_project(vectors, 3)
            result["method_actual"] = "pca"
            result["warning"] = f"t-SNE unavailable; used PCA. {exc}"
            return result
    return pca_project(vectors, 3) | {"method_actual": "pca"}


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


def _coordinate_names(method: str) -> list[str]:
    prefix = "t-SNE" if method == "tsne" else "PC"
    return [f"{prefix}{idx}" for idx in range(1, 4)]
