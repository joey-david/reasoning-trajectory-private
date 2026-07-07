"""Completion checks for the increased object-extraction experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .storage import load_experiment_config, output_dir
from .sweeps import read_json


def validate_improvement(run_path: Path) -> dict[str, Any]:
    """Validate feature and report contracts without rerunning any model work."""
    loaded = load_experiment_config(run_path)
    cfg = loaded["experiment"]["improvement"]
    out = output_dir(run_path)
    feature_path = out / "captured_features.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing capture: {feature_path}")
    with np.load(feature_path) as data:
        required_keys = {
            "h_pool",
            "h_last",
            "h_last_two",
            "h_pre_anchor",
            "h_delta",
            "h_text_mean",
            "record_ids",
            "layers",
        }
        missing_keys = sorted(required_keys - set(data.files))
        captured_layers = data["layers"].astype(int).tolist()
    if missing_keys:
        raise ValueError(f"Capture lacks improvement views: {missing_keys}")
    missing_layers = sorted(set(cfg["layers"]) - set(captured_layers))
    if missing_layers:
        raise ValueError(f"Capture lacks configured layers: {missing_layers}")
    improvement = out / "improvement"
    report_names = (
        "retrieval_sweep",
        "causal_sweep",
        "nonlinear_sweep",
        "writer_report",
        "ablation_sweep",
        "trajectory_gate",
    )
    reports = {}
    for name in report_names:
        path = improvement / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing improvement report: {path}")
        reports[name] = read_json(path)
    ablation_grid_path = improvement / "ablation_grid.json"
    if ablation_grid_path.exists():
        reports["ablation_grid"] = read_json(ablation_grid_path)
    scopes = {
        row["scope"] for row in reports["causal_sweep"]["results"]
    }
    missing_scopes = sorted(set(cfg["patch_scopes"]) - scopes)
    if missing_scopes:
        raise ValueError(f"Causal sweep lacks scopes: {missing_scopes}")
    nonlinear_cfg = cfg["nonlinear"]
    hidden = nonlinear_cfg.get(
        "hidden_dimensions", [nonlinear_cfg.get("hidden_dimension", 128)]
    )
    if not isinstance(hidden, list):
        hidden = [hidden]
    epochs = nonlinear_cfg["epochs"]
    if not isinstance(epochs, list):
        epochs = [epochs]
    expected_nonlinear = (
        len(nonlinear_cfg["variants"])
        * len(nonlinear_cfg["latent_dimensions"])
        * len(hidden)
        * len(epochs)
    )
    nonlinear_count = len(reports["nonlinear_sweep"]["results"])
    if nonlinear_count != expected_nonlinear:
        raise ValueError(
            "Nonlinear sweep is incomplete: "
            f"{nonlinear_count}/{expected_nonlinear} cells"
        )
    writer_epochs = cfg["writer"]["epochs"]
    if not isinstance(writer_epochs, list):
        writer_epochs = [writer_epochs]
    if reports["writer_report"]["epoch_sweep"] != sorted(writer_epochs):
        raise ValueError("Writer epoch sweep does not match config")
    if reports["writer_report"]["pair_split"]["overlap"]:
        raise ValueError("Writer train and validation questions overlap")
    if not np.all(np.isfinite(reports["writer_report"]["losses"])):
        raise ValueError("Writer losses contain non-finite values")
    ablation_modes = set(reports["ablation_sweep"]["summary"])
    required_modes = {"object", "random", "lexical", "answer", "compression"}
    if missing_modes := sorted(required_modes - ablation_modes):
        raise ValueError(f"Ablation sweep lacks controls: {missing_modes}")
    ablation_grid_cells = None
    if "ablation_grid" in reports:
        ablation_grid_cells = len(reports["ablation_grid"]["cells"])
        if reports["ablation_grid"]["selected"] is None:
            raise ValueError("Ablation grid did not evaluate any cells")
    budget_hours = float(cfg.get("budget", {}).get("total_hours", 12))
    if budget_hours > 12:
        raise ValueError(f"Configured budget exceeds 12 hours: {budget_hours}")
    result = {
        "run_path": str(run_path),
        "captured_layers": captured_layers,
        "causal_cells": len(reports["causal_sweep"]["results"]),
        "nonlinear_cells": nonlinear_count,
        "writer_cells": len(reports["writer_report"]["writer_results"]),
        "ablation_prompts": reports["ablation_sweep"]["prompts"],
        "trajectory_status": reports["trajectory_gate"]["status"],
        "budget_hours": budget_hours,
        "valid": True,
    }
    if ablation_grid_cells is not None:
        result["ablation_grid_cells"] = ablation_grid_cells
    return result
