from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.analysis.common import analysis_path, load_generations, selected_token_indices
from src.analysis.projection import pca_project, tsne_project
from src.config import run_path


STEP_RE = re.compile(r"^\s*Step\s*(\d{1,2})\s*:", re.IGNORECASE)


def write_trajectory_projection(
    config: dict[str, Any],
    layer: str | int | None = None,
    interval: int = 16,
    method: str = "pca",
    max_points: int = 12000,
    **_: Any,
) -> Path:
    """
    Project reasoning trajectories.

    Preferred mode:
      use activations immediately before generated "Step N:" markers
      and immediately before the final-answer marker "####".

    Fallback mode:
      if a generation has no usable Step/#### markers, sample every `interval`
      tokens as before.

    This matches the paper's geometry much better than every-token PCA.
    """
    import numpy as np

    base = run_path(config)
    rows = load_generations(config)
    vectors = []
    points = []

    selected_layer = str(layer if layer is not None else _default_layer(config, rows))
    used_boundary_mode = 0
    used_fallback_mode = 0

    for row in rows:
        if not row.get("activation_file"):
            continue

        arrays = np.load(base / row["activation_file"])
        if selected_layer not in arrays.files:
            continue

        values = arrays[selected_layer]
        token_count = int(values.shape[0])

        boundary_points = step_boundary_indices(row, token_count)

        if len(boundary_points) >= 2:
            used_boundary_mode += 1
            chosen = boundary_points
        else:
            used_fallback_mode += 1
            chosen = [
                {
                    "index": idx,
                    "label": f"tok {idx}",
                    "step_number": None,
                    "kind": "token",
                }
                for idx in selected_token_indices(row, token_count, interval)
            ]

        for j, item in enumerate(chosen):
            idx = int(item["index"])
            if not (0 <= idx < token_count):
                continue

            vectors.append(values[idx])
            points.append({
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "token_index": idx,
                "step_label": item["label"],
                "step_number": item["step_number"],
                "kind": item["kind"],
                "success": bool(row.get("success", False)),
                "predicted_answer": row.get("predicted_answer", ""),
                "expected_answer": row.get("expected_answer", ""),
                "is_start": j == 0,
                "is_end": j == len(chosen) - 1,
            })

    if vectors:
        projection = _project(vectors, method)
        coords = projection.pop("coords")
        for point, coord in zip(points, coords):
            point["x"], point["y"], point["z"] = [float(value) for value in coord[:3]]
    else:
        projection = {}

    actual_method = projection.get("method_actual", method) if vectors else method
    warning = projection.get("warning", "") if vectors else ""
    if used_fallback_mode:
        extra = f"{used_fallback_mode} generations had no usable Step/#### boundary markers and used token-interval fallback."
        warning = f"{warning} {extra}".strip()

    payload = {
        "tool": "trajectory_projection",
        "sampling": "step_boundaries_preceding_markers",
        "layer": selected_layer,
        "interval": int(interval),
        "method_requested": method,
        "method_actual": actual_method,
        "dimensions": 3,
        "coordinate_names": _coordinate_names(actual_method),
        "explained_variance_ratio": projection.get("explained_variance_ratio", []) if vectors else [],
        "warning": warning,
        "boundary_generations": used_boundary_mode,
        "fallback_generations": used_fallback_mode,
        "points": points,
    }

    target = analysis_path(config, f"trajectory_projection_layer{selected_layer}_i{int(interval)}_{method}.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def step_boundary_indices(row: dict[str, Any], token_count: int) -> list[dict[str, Any]]:
    """
    Return activation indices immediately before generated Step/#### markers.

    If the model emits:
      "... computation ...\\nStep 2:"
    then the activation at the token just before "Step" is treated as the
    completed Step 1 state.

    Likewise, the activation before "####" is the final reasoning state.
    """
    token_texts = row.get("token_texts") or []
    if not token_texts:
        return []

    markers = []
    seen_steps = set()
    seen_answer = False

    for i in range(len(token_texts)):
        clean_current = clean_token(token_texts[i])

        # Detect generated Step N: markers.
        if "Step" in clean_current:
            window = token_window(token_texts, i, width=8)
            m = STEP_RE.match(window)
            if m:
                step_n = int(m.group(1))
                if step_n not in seen_steps:
                    seen_steps.add(step_n)
                    markers.append({
                        "marker_index": i,
                        "marker_kind": "step",
                        "marker_step": step_n,
                    })

        # Detect final-answer marker ####, including tokenized as # # # #.
        if not seen_answer and "#" in clean_current:
            window = token_window(token_texts, i, width=8)
            if "####" in window.replace(" ", ""):
                seen_answer = True
                markers.append({
                    "marker_index": i,
                    "marker_kind": "answer",
                    "marker_step": None,
                })

    markers.sort(key=lambda x: x["marker_index"])
    result = []
    seen_labels = set()

    for marker in markers:
        marker_idx = int(marker["marker_index"])
        boundary_idx = marker_idx - 1
        if boundary_idx < 0:
            # If Step 1 is the first generated token, its preceding activation
            # is the prompt-final state, which we did not save. Skip it.
            continue

        if marker["marker_kind"] == "step":
            step_n = int(marker["marker_step"])

            # "before Step 2" = completed Step 1 state.
            completed_step = step_n - 1
            if completed_step < 1:
                continue

            label = f"Step {completed_step}"
            kind = "step_boundary"
            step_number = completed_step
        else:
            label = "Answer"
            kind = "answer_boundary"
            step_number = 999

        if label in seen_labels:
            continue
        seen_labels.add(label)

        result.append({
            "index": boundary_idx,
            "label": label,
            "step_number": step_number,
            "kind": kind,
        })

    return result


def clean_token(token: Any) -> str:
    text = str(token)
    text = text.replace("Ġ", " ")
    text = text.replace("▁", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("</s>", "")
    text = text.replace("<s>", "")
    return text


def token_window(tokens: list[Any], start: int, width: int = 8) -> str:
    return "".join(clean_token(tok) for tok in tokens[start:start + width])


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
