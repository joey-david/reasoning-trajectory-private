"""Orchestrate segmentation, latent featurization, clustering, and artifact writing for reasoning steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_trajectory.artifacts import (
    evenly_capped,
    load_hidden_states_npz,
    read_generation_rows,
    write_jsonl,
)
from reasoning_trajectory.steps.clustering import (
    ClusterModel,
    assign_clusters,
    cluster_metadata,
    fit_cluster_model,
)
from reasoning_trajectory.steps.features import (
    StepFeature,
    StepMatrices,
    build_step_features,
    stack_features,
)
from reasoning_trajectory.steps.projection import projection_payloads
from reasoning_trajectory.steps.segmentation import (
    build_segments,
    configured_segmenters,
)
from reasoning_trajectory.token_alignment import build_token_spans


def write_step_classification(run_path: Path, cfg: dict[str, Any]) -> None:
    """Generate all configured step-classification artifacts for a run.

    Args:
        run_path: Completed run folder with hidden-state artifacts.
        cfg: Analysis step segmenter, clustering, projection, and cap options.

    Returns:
        None; writes per-layer records, vectors, clusters, plots, and a manifest.
    """
    rows = read_generation_rows(run_path)
    rows = [row for row in rows if row.get("hidden_states_file")]
    if not rows:
        return

    segmenters = configured_segmenters(cfg)
    token_spans = build_token_spans(run_path, rows)
    step_cfg = cfg.get("step_classification", {})
    max_steps = int(step_cfg.get("max_steps", 12000))
    out_dir = run_path / "analysis" / "step_classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_layer: dict[int, list[Any]] = {}
    for row_idx, row in enumerate(rows):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        for segmenter_name, segmenter_spec in segmenters.items():
            segments = build_segments(
                row,
                segmenter_name,
                segmenter_spec,
                token_spans=token_spans[row_idx],
            )
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
        if not features:
            continue
        records = [dict(item.record) for item in features]
        fit_indices = evenly_capped(list(range(len(features))), max_steps)
        fit_features = [features[i] for i in fit_indices]
        fit_records = [records[i] for i in fit_indices]
        fit_vectors = stack_features(fit_features)
        cluster_model = fit_cluster_model(fit_records, fit_vectors, cfg)
        assign_all_clusters(records, features, cluster_model)
        cluster_info = cluster_metadata(records, cluster_model)
        save_layer_artifacts(
            out_dir,
            layer,
            [dict(record) for record in fit_records],
            fit_vectors,
            cluster_info,
        )
        for method, payload in projection_payloads(
            records, features, layer, cfg
        ).items():
            path = out_dir / f"{method}_layer{layer}_steps.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            manifest.append(
                {
                    "plot_type": "step_classification",
                    "method": method,
                    "layer": layer,
                    "points": len(payload["points"]),
                    "source_points": payload.get(
                        "source_points", len(payload["points"])
                    ),
                    "sampled": bool(payload.get("sampled", False)),
                    "path": path.relative_to(run_path).as_posix(),
                }
            )

    (out_dir / "interactive_index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def assign_all_clusters(
    records: list[dict[str, Any]],
    features: list[StepFeature],
    model: ClusterModel | None,
    *,
    chunk_size: int = 4096,
) -> None:
    """Assign bounded-fit clusters to every step in memory-safe chunks.

    Args:
        records: Aligned records to analyze or annotate.
        features: Aligned feature arrays or step feature bundle.
        model: Loaded model used for inference or transformation.
        chunk_size: Maximum number of rows transformed at once.

    Returns:
        None.
    """
    for start in range(0, len(features), chunk_size):
        end = start + chunk_size
        assign_clusters(
            records[start:end],
            stack_features(features[start:end]),
            model,
        )


def save_layer_artifacts(
    out_dir: Path,
    layer: int,
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    cluster_info: dict[str, Any],
) -> None:
    """Persist records, cluster metadata, and compact vectors for one layer.

    Args:
        out_dir: Step-classification artifact directory.
        layer: Decoder-layer ID used in output filenames.
        records: Mutable step records aligned with ``vectors``.
        vectors: Full mean, direction, and nudge feature matrices.
        cluster_info: JSON-compatible clustering metadata.

    Returns:
        None; also removes the obsolete probe-example artifact when present.
    """
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
