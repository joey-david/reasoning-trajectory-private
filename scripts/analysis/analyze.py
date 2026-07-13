#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.answers import update_answers
from src.analysis.hard_questions import write_hard_questions
from reasoning_trajectory.interactive import write_interactive_trajectories
from reasoning_trajectory.layer_variations import (
    write_correctness_group_plots,
    write_layer_plots,
)
from reasoning_trajectory.pipeline import analyze_trajectories
from src.analysis.solution_objects import write_solution_objects
from reasoning_trajectory.steps import write_step_classification
from reasoning_trajectory.markers import write_step_markers
from reasoning_trajectory.manifest import write_manifest
from src.runtime.config import load_config


def main() -> int:
    """Run configured post-processing for one run directory.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Analyze one run folder.")
    parser.add_argument("run_path")
    args = parser.parse_args()
    run_path = Path(args.run_path)
    cfg = load_config(run_path).get("analysis", {})
    update_answers(run_path, cfg)
    write_step_markers(run_path, cfg)
    write_solution_objects(run_path, cfg)
    write_hard_questions(run_path, cfg)
    static_index = run_path / "analysis" / "plots" / "index.json"
    if static_index.exists():
        static_index.unlink()
    write_interactive_trajectories(run_path, cfg)
    if cfg.get("layer_variations", False):
        write_layer_plots(run_path, cfg)
    if cfg.get("correctness_group_plots", False):
        write_correctness_group_plots(run_path, cfg)
    write_step_classification(run_path, cfg)
    if cfg.get("trajectory_metrics", True):
        analyze_trajectories(run_path, cfg.get("trajectory_metrics_config", {}))
    write_manifest(Path("runs"), Path("web/data/runs.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
