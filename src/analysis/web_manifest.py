"""Discover analyzed runs and write the manifest consumed by the static web interface."""

from __future__ import annotations

import json
from pathlib import Path

from src.analysis.common import read_sample_records


def write_manifest(runs_root: Path, out_path: Path) -> None:
    """Write web-visible artifacts for every generated run below a root.

    Args:
        runs_root: Root containing ``model/run/config.yaml`` folders.
        out_path: Destination JSON manifest used by ``web/index.html``.

    Returns:
        None.
    """
    runs = []
    for config in sorted(runs_root.glob("*/*/config.yaml")):
        run = config.parent
        gen = run / "generation" / "generations.jsonl"
        if not gen.exists():
            continue
        plots = load_json(run / "analysis" / "plots" / "index.json", [])
        interactive_plots = load_json(
            run / "analysis" / "plots" / "interactive_index.json", []
        )
        step_classification_plots = load_json(
            run / "analysis" / "step_classification" / "interactive_index.json", []
        )
        step_markers = run / "analysis" / "step_markers.json"
        solution_objects = run / "analysis" / "solution_objects.jsonl"
        hard_questions = run / "analysis" / "hard_questions.jsonl"
        runs.append(
            {
                "model": run.parent.name,
                "run": run.name,
                "generations": web_path(gen),
                "samples": read_sample_records(run),
                "plots": add_web_paths(run, plots),
                "interactive_plots": add_web_paths(run, interactive_plots),
                "step_classification_plots": add_web_paths(
                    run, step_classification_plots
                ),
                "step_markers": web_path(step_markers)
                if step_markers.exists()
                else None,
                "solution_objects": web_path(solution_objects)
                if solution_objects.exists()
                else None,
                "hard_questions": web_path(hard_questions)
                if hard_questions.exists()
                else None,
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False), encoding="utf-8"
    )


def load_json(path: Path, default):
    """Load a JSON file when present.

    Args:
        path: JSON document to read.
        default: Value returned when the path does not exist.

    Returns:
        The decoded JSON value or ``default``.
    """
    return json.loads(path.read_text()) if path.exists() else default


def web_path(path: Path) -> str:
    """Convert a repository path into a URL relative to the web directory.

    Args:
        path: Repository-relative artifact path.

    Returns:
        A ``../``-prefixed path suitable for the static UI.
    """
    return "../" + path.as_posix()


def add_web_paths(run: Path, plots: list[dict]) -> list[dict]:
    """Rewrite run-relative plot paths for web access.

    Args:
        run: Run folder containing each plot artifact.
        plots: Plot manifest entries with run-relative ``path`` fields.

    Returns:
        Copied plot entries with browser-relative paths.
    """
    return [{**plot, "path": web_path(run / plot["path"])} for plot in plots]
