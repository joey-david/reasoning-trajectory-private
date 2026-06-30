"""Build browser-consumable PCA and t-SNE payloads for selected trajectory points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.analysis.common import evenly_capped, project_3d, read_generation_rows
from src.analysis.step_markers import configured_selectors
from src.analysis.token_alignment import TokenSpan, build_token_spans
from src.analysis.token_selectors import build_token_selector
from src.runtime.artifact_store import load_hidden_states_npz


PointItem = tuple[np.ndarray, dict[str, Any], int, int, str]


def write_interactive_trajectories(run_path: Path, cfg: dict[str, Any]) -> None:
    """Write interactive trajectory point payloads and their manifest.

    Args:
        run_path: Completed run folder with hidden-state artifacts.
        cfg: Analysis selector, projection, and point-limit options.

    Returns:
        None; writes JSON payloads under ``analysis/plots`` when data exists.
    """
    rows = read_generation_rows(run_path)
    rows = [row for row in rows if row.get("hidden_states_file")]
    if not rows:
        return

    selectors = configured_selectors(cfg)
    token_spans = build_token_spans(run_path, rows)
    max_points = int(
        cfg.get("max_interactive_points", cfg.get("max_plot_points", 5000))
    )
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
                            (
                                states[token_idx, col],
                                row,
                                int(token_idx),
                                traj_id,
                                selector_name,
                            )
                        )

    manifest: list[dict[str, Any]] = []
    for layer, items in points_by_layer.items():
        projection_items = evenly_capped(items, max_points)
        if len(projection_items) < 3:
            continue
        projection_x = np.stack([item[0] for item in projection_items])
        projections = {
            "pca": project_all_with_fitted_pca(items, projection_x),
            "tsne": project_3d(projection_x)["tsne"],
        }
        projection_inputs = {
            "pca": items,
            "tsne": projection_items,
        }
        for method, coords in projections.items():
            method_items = projection_inputs[method]
            path = out_dir / f"{method}_layer{layer}_interactive.json"
            payload = build_payload(
                method,
                layer,
                coords,
                method_items,
                selectors,
                max_points=max_points,
                source_points=len(items),
                token_spans=token_spans,
            )
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            manifest.append(
                {
                    "method": method,
                    "layer": layer,
                    "points": len(method_items),
                    "source_points": len(items),
                    "path": path.relative_to(run_path).as_posix(),
                }
            )

    (out_dir / "interactive_index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def project_all_with_fitted_pca(
    items: list[PointItem],
    fit_x: np.ndarray,
    *,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Fit a shared PCA basis on a bounded sample, then transform every point."""
    pca = PCA(n_components=3).fit(fit_x)
    chunks = (
        np.stack([item[0] for item in items[start : start + chunk_size]])
        for start in range(0, len(items), chunk_size)
    )
    return np.concatenate([pca.transform(chunk) for chunk in chunks])


def build_payload(
    method: str,
    layer: int,
    coords: np.ndarray,
    items: list[PointItem],
    selectors: dict[str, dict[str, Any]],
    *,
    max_points: int = 0,
    source_points: int | None = None,
    token_spans: list[list[TokenSpan]] | None = None,
) -> dict[str, Any]:
    """Combine projected coordinates and source metadata for the web UI.

    Args:
        method: Projection name, such as ``pca`` or ``tsne``.
        layer: Decoder-layer ID represented by the points.
        coords: Three-dimensional coordinates aligned with ``items``.
        items: Hidden vectors and associated rollout/token/selector metadata.
        selectors: Named selector specifications used to produce the points.
        max_points: Browser-side cap applied after the active filters.
        source_points: Candidate count before projection-specific sampling.
        token_spans: Character spans indexed by trajectory and generated token.

    Returns:
        A JSON-compatible interactive plot payload.
    """
    points: list[dict[str, Any]] = []
    for point, (_, row, token_idx, traj_id, selector_name) in zip(coords, items):
        span = token_span(token_spans, traj_id, token_idx)
        record = {
            "x": round(float(point[0]), 6),
            "y": round(float(point[1]), 6),
            "z": round(float(point[2]), 6),
            "sample_id": row.get("sample_id"),
            "seed": row.get("seed"),
            "trajectory_id": traj_id,
            "selector": selector_name,
            "token_idx": token_idx,
            "token_fraction": token_idx
            / max(len(row.get("generated_token_ids", [])) - 1, 1),
            "is_correct": row.get("is_correct"),
            "produced_answer": row.get("produced_answer"),
            "reasoning_length": row.get("reasoning_length"),
        }
        if span is not None:
            record["char_start"], record["char_end"] = span
        points.append(record)
    return {
        "method": method,
        "layer": layer,
        "selectors": selectors,
        "max_points": max_points,
        "source_points": source_points if source_points is not None else len(items),
        "sampled": source_points is not None and len(items) < source_points,
        "points": points,
    }


def token_span(
    token_spans: list[list[TokenSpan]] | None,
    trajectory_id: int,
    token_idx: int,
) -> TokenSpan:
    """Return a safely bounded character span for one plotted token."""
    if token_spans is None or not 0 <= trajectory_id < len(token_spans):
        return None
    spans = token_spans[trajectory_id]
    return spans[token_idx] if 0 <= token_idx < len(spans) else None
