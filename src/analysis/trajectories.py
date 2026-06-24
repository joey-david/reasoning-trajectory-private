from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.common import evenly_capped, project_3d, read_generation_rows
from src.artifact_store import load_hidden_states_npz
from src.analysis.token_selectors import Selector, build_token_selector


def plot_trajectories(run_path: Path, cfg: dict[str, Any]) -> None:
    rows = read_generation_rows(run_path)
    rows = [r for r in rows if r.get("hidden_states_file")]
    rows = filter_trajectories(rows, cfg.get("trajectory_selector", {}))
    if not rows:
        return
    selector = token_selector(cfg.get("token_selector", {"every_n": 1}))
    max_points = int(cfg.get("max_plot_points", 5000))
    out_dir = run_path / "analysis" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    points: dict[int, list[tuple[np.ndarray, dict[str, Any], int, int]]] = {}
    for traj_id, row in enumerate(rows):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        for t in selector(row):
            if 0 <= t < states.shape[0]:
                for col, layer in enumerate(layers):
                    points.setdefault(layer, []).append((states[t, col], row, t, traj_id))
    manifest = []
    for layer, items in points.items():
        items = evenly_capped(items, max_points)
        if len(items) < 3:
            continue
        x = np.stack([it[0] for it in items])
        for name, coords in project_3d(x).items():
            path = out_dir / f"{name}_layer{layer}.png"
            draw_plot(coords, items, path, f"{name.upper()} layer {layer}")
            manifest.append({"method": name, "layer": layer, "trajectories": len(rows), "path": path.relative_to(run_path).as_posix()})
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def filter_trajectories(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    sample_ids = {str(x) for x in spec.get("sample_ids", [])}
    seeds = {int(x) for x in spec.get("seeds", [])}
    return [
        r for r in rows
        if (not sample_ids or r["sample_id"] in sample_ids)
        and (not seeds or int(r["seed"]) in seeds)
    ]


def token_selector(spec: dict[str, Any]) -> Selector:
    return build_token_selector(spec)


def draw_plot(coords: np.ndarray, items: list[tuple[np.ndarray, dict[str, Any], int, int]], path: Path, title: str) -> None:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection="3d")
    by_traj: dict[int, list[np.ndarray]] = {}
    buckets: dict[tuple[bool, bool, bool], list[np.ndarray]] = {}
    for point, (_, row, t, traj_id) in zip(coords, items):
        by_traj.setdefault(traj_id, []).append(point)
        ok = bool(row.get("is_correct"))
        final = row.get("reasoning_length") is not None and t >= row["reasoning_length"]
        first = t == 0
        buckets.setdefault((ok, first, final), []).append(point)
    for (ok, first, final), bucket in buckets.items():
        arr = np.stack(bucket)
        ax.scatter(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            c=("#8bd18b" if ok else "#e68080"),
            marker=("^" if first else "s" if final else "o"),
            s=(42 if first or final else 8),
            edgecolors=("#f39c12" if first else "none"),
        )
    for traj in by_traj.values():
        if len(traj) > 1:
            line = np.stack(traj)
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color="#9aa4b2", alpha=0.25, linewidth=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
