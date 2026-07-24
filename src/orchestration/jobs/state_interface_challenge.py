"""Parallel workers for length-extrapolation interface challenges."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.runtime.config import load_config
from src.experiments.depth_relief.state_interface_challenge import (
    challenge_dir,
    evaluate_interface_challenge,
)
from src.orchestration.jobs.contract import Task, TaskResult


def _complete(run_path: Path, profile: str, side: str) -> bool:
    path = challenge_dir(run_path, profile) / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    return int(summary.get(f"{side}_case_count", 0)) == int(
        summary.get("expected_case_count", -1)
    )


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    profiles = load_config(run_path).get("state_interface_challenges", {})
    tasks = [
        {"profile": profile, "side": side}
        for profile in profiles
        for side in ("interface", "outcome")
        if not _complete(run_path, profile, side)
    ]
    total = 2 * len(profiles)
    return tasks, total, total - len(tasks)


def setup_worker(run_path: Path) -> "StateInterfaceChallengeWorker":
    return StateInterfaceChallengeWorker(run_path)


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "evaluation/challenges/logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateInterfaceChallengeWorker:
    run_path: Path

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        profile = str(task["profile"])
        side = str(task["side"])
        progress.set_description(f"long horizon {profile}/{side}")
        result = evaluate_interface_challenge(
            self.run_path,
            profile,
            side,
            on_progress=progress.set_description,
        )
        return TaskResult(
            units=int(result[f"{side}_case_count"]), unit="challenge case"
        )
