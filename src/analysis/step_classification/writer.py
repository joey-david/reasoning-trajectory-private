from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.step_classification.clustering import assign_clusters
from src.analysis.step_classification.features import build_step_features, stack_features
from src.analysis.step_classification.projection import projection_payloads
from src.analysis.step_classification.segmentation import build_segments, configured_segmenters
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
        features = capped_features(features, max_steps)
        if not features:
            continue
        records = [dict(item.record) for item in features]
        means, directions, nudges = stack_features(features)
        cluster_info = assign_clusters(records, means, directions, nudges, cfg)
        save_layer_artifacts(out_dir, layer, records, means, directions, nudges, cluster_info)
        for method, payload in projection_payloads(records, means, layer, cfg).items():
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
    means: np.ndarray,
    directions: np.ndarray,
    nudges: np.ndarray,
    cluster_info: dict[str, Any],
) -> None:
    for feature_row, record in enumerate(records):
        record["feature_row"] = feature_row

    (out_dir / f"layer{layer}_steps.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (out_dir / f"layer{layer}_clusters.json").write_text(
        json.dumps(cluster_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / f"layer{layer}_vectors.npz",
        mean_vectors=means.astype(np.float16),
        direction_vectors=directions.astype(np.float16),
        nudge_vectors=nudges.astype(np.float16),
        variance=np.asarray([record["variance"] for record in records], dtype=np.float32),
        cluster_id=np.asarray([record.get("cluster_id", -1) for record in records], dtype=np.int32),
    )
    (out_dir / f"layer{layer}_probe_examples.jsonl").write_text(
        "".join(json.dumps(probe_record(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def probe_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "seed": record["seed"],
        "segmenter": record["segmenter"],
        "step_idx": record["step_idx"],
        "cluster_id": record.get("cluster_id"),
        "text": record["step_text"],
        "features_npz": f"layer{record['layer']}_vectors.npz",
        "feature_row": record["feature_row"],
    }


def capped_features(features: list[Any], max_steps: int) -> list[Any]:
    if max_steps <= 0 or len(features) <= max_steps:
        return features
    keep = np.linspace(0, len(features) - 1, max_steps, dtype=int)
    return [features[int(i)] for i in keep]


def read_generation_rows(run_path: Path) -> list[dict[str, Any]]:
    path = run_path / "generation" / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
