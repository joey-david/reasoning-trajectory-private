"""Regression tests for the reusable trajectory-analysis package."""

from __future__ import annotations

import json

import numpy as np
import torch

from reasoning_trajectory.bank import TrajectoryPath
from reasoning_trajectory.layer_variations import _model_name, write_layer_plots
from src.models.generation_pipeline import resolve_capture_layer_indices
from reasoning_trajectory.metrics.alignment import alignment_summary
from reasoning_trajectory.metrics.geometry import trajectory_geometry
from reasoning_trajectory.metrics.diagnostics import basin_summary, compression_curve
from reasoning_trajectory.manifest import browser_samples
from reasoning_trajectory.pipeline import analyze_trajectories
from reasoning_trajectory.steps import parse_structured_spans, pool_token_states


def test_geometry_distinguishes_straight_and_meandering_paths() -> None:
    straight = _path("straight", np.column_stack([np.arange(6), np.zeros(6)]))
    wave = _path(
        "wave",
        np.asarray([[0, 0], [1, 2], [2, -2], [3, 2], [4, -2], [5, 0]]),
    )

    straight_metrics = trajectory_geometry(straight)
    wave_metrics = trajectory_geometry(wave)

    assert straight_metrics["net_path_ratio"] == 1.0
    assert wave_metrics["net_path_ratio"] < 0.3
    assert wave_metrics["mean_curvature"] > straight_metrics["mean_curvature"]
    assert wave_metrics["effective_width"] > 0.8


def test_alignment_is_best_for_identical_paths() -> None:
    base = np.column_stack([np.arange(8), np.sin(np.arange(8))])
    shifted = base + np.asarray([0.0, 4.0])
    identical = alignment_summary(base, base)
    different = alignment_summary(base, shifted)

    assert identical["dtw"] == 0.0
    assert identical["cka"] > 0.999
    assert different["frechet"] > identical["frechet"]


def test_pipeline_writes_browser_report_from_run_artifacts(tmp_path) -> None:
    generation = tmp_path / "generation"
    hidden = generation / "hidden_states"
    hidden.mkdir(parents=True)
    rows = []
    for index in range(4):
        sample_id = f"problem-{index // 2}"
        path = hidden / f"trace-{index}.npz"
        states = np.zeros((12, 1, 6), dtype=np.float32)
        states[:, 0, 0] = np.arange(12)
        states[:, 0, 1] = index
        if index % 2:
            states[6:, 0, 2] = np.linspace(0, 8, 6)
        np.savez_compressed(path, hidden_states=states, layer_indices=[7])
        rows.append(
            {
                "sample_id": sample_id,
                "seed": index,
                "is_correct": index % 2 == 0,
                "generated_token_ids": list(range(12)),
                "hidden_states_file": str(path.relative_to(tmp_path)),
            }
        )
    (generation / "generations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = analyze_trajectories(
        tmp_path,
        {
            "max_trajectories": 4,
            "max_tokens_per_trajectory": 8,
            "max_pairs": 4,
            "basin_clusters": 2,
        },
    )

    assert report["layer"] == 7
    assert report["summary"]["correct"] == 2
    assert report["summary"]["incorrect"] == 2
    assert len(report["failures"]) == 2
    assert (tmp_path / "analysis" / "trajectory_metrics.json").exists()


def test_structured_parser_and_pooling_cover_reference_step_tools() -> None:
    text = "Step 1: bind x\nStep 2: compute x + 1\nexact result"
    spans = parse_structured_spans(text)
    states = np.arange(6 * 2 * 3).reshape(6, 2, 3)

    assert [span.labels for span in spans] == [
        ("numbered",),
        ("numbered",),
        ("proof_tactic",),
    ]
    pooled = pool_token_states(states, [(0, 1), (2, 4)], pooling="last")
    assert pooled.shape == (2, 2, 3)
    np.testing.assert_array_equal(pooled[0], states[1])


def test_constant_paths_have_defined_compression_and_basins() -> None:
    paths = [
        _path(f"constant-{index}", np.ones((5, 3)))
        for index in range(3)
    ]

    assert compression_curve(paths, [2]) == [
        {
            "dimensions": 2,
            "explained_variance": 1.0,
            "normalized_reconstruction_error": 0.0,
        }
    ]
    basins = basin_summary(paths, clusters=3)
    assert basins["cluster_count"] == 1
    assert basins["sizes"] == [3]
    assert basins["silhouette"] is None


def test_browser_manifest_drops_generation_only_token_arrays() -> None:
    samples = browser_samples(
        {
            "sample": {
                "sample_id": "sample",
                "prompt": "question",
                "gold_answer": "42",
                "input_ids": [1, 2, 3],
                "dp1_idx": 3,
            }
        }
    )

    assert samples == {
        "sample": {
            "sample_id": "sample",
            "prompt": "question",
            "gold_answer": "42",
        }
    }


def test_capture_layer_slice_resolves_to_all_model_layers() -> None:
    class Model:
        def __init__(self) -> None:
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(
                [torch.nn.Identity() for _ in range(4)]
            )

    assert resolve_capture_layer_indices({"layers": "[:]"}, Model()) == [0, 1, 2, 3]
    assert resolve_capture_layer_indices({"layers": [-1, 2]}, Model()) == [-1, 2]


def test_layer_variations_model_name_accepts_bare_layer_slice(tmp_path) -> None:
    (tmp_path / "config.yaml").write_text(
        """
model:
  name: dummy-model
capture:
  enabled: true
  layers: [:]
""",
        encoding="utf-8",
    )

    assert _model_name(tmp_path) == "dummy-model"


def test_layer_variations_writes_web_payload(tmp_path) -> None:
    generation = tmp_path / "generation"
    hidden = generation / "hidden_states"
    hidden.mkdir(parents=True)
    states = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
    np.savez_compressed(
        hidden / "trace.npz",
        hidden_states=states,
        layer_indices=[0, 1, 2],
    )
    (generation / "generations.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "seed": 7,
                "is_correct": True,
                "produced_answer": "42",
                "hidden_states_file": "generation/hidden_states/trace.npz",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    write_layer_plots(tmp_path, {"layer_variations": {"max_3d_tokens": 2}})

    index = json.loads(
        (tmp_path / "analysis" / "layer_variations" / "web_index.json").read_text(
            encoding="utf-8"
        )
    )
    cosine = next(item for item in index if item["metric"] == "cosine")
    payload = json.loads((tmp_path / cosine["path"]).read_text(encoding="utf-8"))

    assert cosine["traces"] == 1
    assert payload["trace_count"] == 1
    assert payload["traces"][0]["sample_id"] == "sample"
    assert payload["traces"][0]["layer_pairs"][0]["label"] == "0->1"
    assert len(payload["points"]) == 4


def _path(name: str, values: np.ndarray) -> TrajectoryPath:
    """Build a compact synthetic path for metric tests."""
    return TrajectoryPath(
        trajectory_id=name,
        sample_id="sample",
        seed=0,
        is_correct=True,
        layer=0,
        token_indices=list(range(len(values))),
        states=values.astype(np.float32),
    )
