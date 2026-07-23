"""GPU orchestration for cross-adapter state-interface stress tests."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.state_interface_stress import (
    STRESS_ROOT,
    evaluate_stress_condition,
    stress_condition_dir,
    stress_config,
)
from src.orchestration.jobs.contract import Task, TaskResult


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Return each incomplete profile-condition evaluation."""
    config = stress_config(run_path)
    profiles = sorted(
        path.name
        for path in (run_path / STRESS_ROOT).iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    )
    if not profiles:
        raise RuntimeError("No state-interface stress profile is prepared")
    tasks = [
        {"profile": profile, "condition": condition}
        for profile in profiles
        for condition in config["conditions"]
    ]
    pending = []
    for task in tasks:
        path = (
            stress_condition_dir(run_path, task["profile"], task["condition"])
            / "summary.json"
        )
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            pending.append(task)
    return pending, len(tasks), len(tasks) - len(pending)


def setup_worker(run_path: Path) -> "StateInterfaceStressWorker":
    """Create a worker that loads each assigned source adapter on demand."""
    stress_config(run_path)
    return StateInterfaceStressWorker(run_path)


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the stress worker log path."""
    return run_path / STRESS_ROOT / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateInterfaceStressWorker:
    """Evaluate one source adapter and release it when the task ends."""

    run_path: Path

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        profile = str(task["profile"])
        condition = str(task["condition"])
        progress.set_description(f"state stress {profile} {condition}")
        summary = evaluate_stress_condition(
            self.run_path,
            profile,
            condition,
            on_progress=progress.set_description,
        )
        return TaskResult(units=int(summary["case_count"]), unit="stress case")
