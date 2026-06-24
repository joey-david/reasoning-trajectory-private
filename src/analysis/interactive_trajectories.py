from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import evenly_capped, project_3d, read_generation_rows
from src.analysis.step_markers import configured_selectors
from src.analysis.token_selectors import build_token_selector
from src.artifact_store import load_hidden_states_npz


PointItem = tuple[np.ndarray, dict[str, Any], int, int, str]


def write_interactive_trajectories(run_path: Path, cfg: dict[str, Any]) -> None:
    rows = read_generation_rows(run_path)
    rows = [row for row in rows if row.get("hidden_states_file")]
    if not rows:
        return

    selectors = configured_selectors(cfg)
    max_points = int(cfg.get("max_interactive_points", cfg.get("max_plot_points", 5000)))
    out_dir = run_path / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    points_by_layer: dict[int, list[PointItem]] = {}
    for traj_id, row in enumerate(rows):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        for selector_name, selector_spec in selectors.items():
            selector = build_token_selector(selector_spec)
            for token_idx in selector(row):
                if 0 <= token_idx < states.shape[0]:
                    for col, layer in enumerate(layers):
                        points_by_layer.setdefault(layer, []).append(
                            (states[token_idx, col], row, int(token_idx), traj_id, selector_name)
                        )

    manifest: list[dict[str, Any]] = []
    for layer, items in points_by_layer.items():
        items = evenly_capped(items, max_points)
        if len(items) < 3:
            continue
        x = np.stack([item[0] for item in items])
        for method, coords in project_3d(x).items():
            path = out_dir / f"{method}_layer{layer}_interactive.json"
            payload = build_payload(method, layer, coords, items, selectors)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            manifest.append(
                {
                    "method": method,
                    "layer": layer,
                    "points": len(items),
                    "path": path.relative_to(run_path).as_posix(),
                }
            )

    (out_dir / "interactive_index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_payload(
    method: str,
    layer: int,
    coords: np.ndarray,
    items: list[PointItem],
    selectors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for point, (_, row, token_idx, traj_id, selector_name) in zip(coords, items):
        points.append(
            {
                "x": round(float(point[0]), 6),
                "y": round(float(point[1]), 6),
                "z": round(float(point[2]), 6),
                "sample_id": row.get("sample_id"),
                "seed": row.get("seed"),
                "trajectory_id": traj_id,
                "selector": selector_name,
                "token_idx": token_idx,
                "token_fraction": token_idx / max(len(row.get("generated_token_ids", [])) - 1, 1),
                "is_correct": row.get("is_correct"),
                "produced_answer": row.get("produced_answer"),
                "reasoning_length": row.get("reasoning_length"),
            }
        )
    return {
        "method": method,
        "layer": layer,
        "selectors": selectors,
        "points": points,
    }
