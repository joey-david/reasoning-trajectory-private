from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import evenly_capped, read_generation_rows, write_jsonl
from src.analysis.step_classification.clustering import assign_clusters
from src.analysis.step_classification.features import (
    StepMatrices,
    build_step_features,
    stack_features,
)
from src.analysis.step_classification.projection import projection_payloads
from src.analysis.step_classification.segmentation import (
    build_segments,
    configured_segmenters,
)
from src.artifact_store import load_hidden_states_npz


def write_step_classification(run_path: Path, cfg: dict[str, Any]) -> None:
    rows = read_generation_rows(run_path)
    rows = [row for row in rows if row.get("hidden_states_file")]
    if not rows:
        return

    segmenters = configured_segmenters(cfg)
    step_cfg = cfg.get("step_classification", {})
    max_steps = int(step_cfg.get("max_steps", 12000))
    out_dir = run_path / "analysis" / "step_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_layer: dict[int, list[Any]] = {}
    for row in rows:
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        for segmenter_name, segmenter_spec in segmenters.items():
            segments = build_segments(row, segmenter_name, segmenter_spec)
            for layer_col, layer in enumerate(layers):
                by_layer.setdefault(layer, []).extend(
                    build_step_features(
                        states=states,
                        layer=layer,
                        layer_col=layer_col,
                        row=row,
                        segments=segments,
                    )
                )

    manifest: list[dict[str, Any]] = []
    for layer, features in by_layer.items():
        features = evenly_capped(features, max_steps)
        if not features:
            continue
        records = [dict(item.record) for item in features]
        vectors = stack_features(features)
        cluster_info = assign_clusters(records, vectors, cfg)
        save_layer_artifacts(out_dir, layer, records, vectors, cluster_info)
        for method, payload in projection_payloads(
            records, vectors, layer, cfg
        ).items():
            path = out_dir / f"{method}_layer{layer}_steps.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            manifest.append(
                {
                    "plot_type": "step_classification",
                    "method": method,
                    "layer": layer,
                    "points": len(payload["points"]),
                    "path": path.relative_to(run_path).as_posix(),
                }
            )

    (out_dir / "interactive_index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_layer_artifacts(
    out_dir: Path,
    layer: int,
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    cluster_info: dict[str, Any],
) -> None:
    for feature_row, record in enumerate(records):
        record["feature_row"] = feature_row

    write_jsonl(out_dir / f"layer{layer}_steps.jsonl", records)
    (out_dir / f"layer{layer}_clusters.json").write_text(
        json.dumps(cluster_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / f"layer{layer}_vectors.npz",
        mean_vectors=vectors.means.astype(np.float16),
        direction_vectors=vectors.directions.astype(np.float16),
        variance=np.asarray(
            [record["variance"] for record in records], dtype=np.float32
        ),
        cluster_id=np.asarray(
            [record.get("cluster_id", -1) for record in records], dtype=np.int32
        ),
    )
    legacy_probe_path = out_dir / f"layer{layer}_probe_examples.jsonl"
    if legacy_probe_path.exists():
        legacy_probe_path.unlink()
