"""Single-GPU worker for cross-adapter interface substitution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.runtime.config import load_config
from src.experiments.depth_relief.state_interface_substitution import (
    evaluate_interface_substitution,
)
from src.orchestration.jobs.contract import Task, TaskResult


def _summary_path(run_path: Path) -> Path:
    config = load_config(run_path)["state_interface_substitution"]
    consumer = Path(str(config["consumer_run"]))
    return run_path / "evaluation/substitution" / consumer.name / "summary.json"


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    path = _summary_path(run_path)
    complete = (
        path.exists() and bool(json.loads(path.read_text()).get("complete"))
    )
    return ([] if complete else [{}], 1, int(complete))


def setup_worker(run_path: Path) -> "StateInterfaceSubstitutionWorker":
    return StateInterfaceSubstitutionWorker(run_path)


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "evaluation/substitution/logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateInterfaceSubstitutionWorker:
    run_path: Path

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        progress.set_description("cross-adapter state interface")
        result = evaluate_interface_substitution(
            self.run_path, on_progress=progress.set_description
        )
        return TaskResult(
            units=int(result["case_count"]), unit="substitution case"
        )
