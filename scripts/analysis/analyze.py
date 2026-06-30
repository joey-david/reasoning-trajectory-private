#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.answers import update_answers
from src.analysis.hard_questions import write_hard_questions
from src.analysis.interactive_trajectories import write_interactive_trajectories
from src.analysis.solution_objects import write_solution_objects
from src.analysis.step_classification import write_step_classification
from src.analysis.step_markers import write_step_markers
from src.analysis.web_manifest import write_manifest
from src.runtime.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze one run folder.")
    parser.add_argument("run_path")
    args = parser.parse_args()
    run_path = Path(args.run_path)
    cfg = load_config(run_path).get("analysis", {})
    update_answers(run_path, cfg)
    write_step_markers(run_path, cfg)
    write_solution_objects(run_path, cfg)
    write_hard_questions(run_path, cfg)
    if cfg.get("static_plots", False):
        from src.analysis.trajectories import plot_trajectories

        plot_trajectories(run_path, cfg)
    else:
        static_index = run_path / "analysis" / "plots" / "index.json"
        if static_index.exists():
            static_index.unlink()
    write_interactive_trajectories(run_path, cfg)
    write_step_classification(run_path, cfg)
    write_manifest(Path("runs"), Path("web/data/runs.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
