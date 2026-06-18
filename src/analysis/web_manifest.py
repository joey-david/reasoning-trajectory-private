from __future__ import annotations

import json
from pathlib import Path


def write_manifest(runs_root: Path, out_path: Path) -> None:
    runs = []
    for config in sorted(runs_root.glob("*/*/config.yaml")):
        run = config.parent
        gen = run / "generation" / "generations.jsonl"
        if not gen.exists():
            continue
        samples = {
            p.stem: json.loads(p.read_text())
            for p in (run / "generation" / "samples").glob("*.json")
        }
        plots = []
        index = run / "analysis" / "plots" / "index.json"
        if index.exists():
            plots = json.loads(index.read_text())
        interactive_plots = []
        interactive_index = run / "analysis" / "plots" / "interactive_index.json"
        if interactive_index.exists():
            interactive_plots = json.loads(interactive_index.read_text())
        step_markers = run / "analysis" / "step_markers.json"
        solution_objects = run / "analysis" / "solution_objects.jsonl"
        hard_questions = run / "analysis" / "hard_questions.jsonl"
        runs.append({
            "model": run.parent.name,
            "run": run.name,
            "generations": "../" + gen.as_posix(),
            "samples": samples,
            "plots": [{**p, "path": "../" + (run / p["path"]).as_posix()} for p in plots],
            "interactive_plots": [
                {**p, "path": "../" + (run / p["path"]).as_posix()}
                for p in interactive_plots
            ],
            "step_markers": "../" + step_markers.as_posix() if step_markers.exists() else None,
            "solution_objects": "../" + solution_objects.as_posix() if solution_objects.exists() else None,
            "hard_questions": "../" + hard_questions.as_posix() if hard_questions.exists() else None,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"runs": runs}, ensure_ascii=False), encoding="utf-8")
