"""One-GPU orchestration adapter for matched state-handoff LoRA pilots."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any

from src.runtime.config import load_config
from src.experiments.depth_relief.state_handoff_data import (
    INTERFACE_CONDITIONS,
    configured_training_conditions,
)
from src.experiments.depth_relief.state_handoff_evaluation import (
    condition_evaluation_dir,
    evaluate_state_handoff_condition,
)
from src.experiments.depth_relief.state_handoff_training import (
    condition_training_dir,
    require_phase1_training_gate,
    train_state_handoff_condition,
)
from src.orchestration.jobs.contract import Task, TaskResult


def _condition_complete(run_path: Path, condition: str) -> bool:
    training_path = condition_training_dir(run_path, condition) / "checkpoint_manifest.json"
    if condition in INTERFACE_CONDITIONS:
        from src.experiments.depth_relief.state_interface_evaluation import (
            interface_evaluation_dir,
        )

        evaluation_path = interface_evaluation_dir(run_path, condition) / "summary.json"
    else:
        evaluation_path = condition_evaluation_dir(run_path, condition) / "summary.json"
    if not training_path.exists() or not evaluation_path.exists():
        return False
    training = json.loads(training_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    return training.get("status") == "complete" and bool(evaluation.get("complete"))


def _member_runs(run_path: Path) -> tuple[Path, ...]:
    linked = (
        load_config(run_path)
        .get("state_handoff_training", {})
        .get("linked_runs", ())
    )
    members = [run_path, *(Path(str(value)) for value in linked)]
    return tuple(dict.fromkeys(members))


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Return unfinished training-plus-evaluation conditions."""
    pending = [
        {"run_path": str(member), "condition": condition}
        for member in _member_runs(run_path)
        for condition in configured_training_conditions(member)
        if not _condition_complete(member, condition)
    ]
    total = sum(
        len(configured_training_conditions(member))
        for member in _member_runs(run_path)
    )
    return pending, total, total - len(pending)


def setup_worker(run_path: Path) -> "StateHandoffTrainingWorker":
    """Validate the gate before accepting long-running condition tasks."""
    members = _member_runs(run_path)
    for member in members:
        require_phase1_training_gate(member)
    return StateHandoffTrainingWorker(allowed_runs=members)


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the single-GPU training worker log path."""
    return run_path / "training/orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateHandoffTrainingWorker:
    """Run one complete condition, then release its model before evaluation."""

    allowed_runs: tuple[Path, ...]

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        run_path = Path(str(task["run_path"]))
        if run_path not in self.allowed_runs:
            raise ValueError(f"Training task is outside the linked run set: {run_path}")
        condition = str(task["condition"])
        progress.set_description(
            f"state handoff training {run_path.name}/{condition}"
        )
        train_state_handoff_condition(
            run_path,
            condition,
            on_progress=progress.set_description,
            on_step=progress.set_progress,
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        progress.clear_progress()
        progress.set_description(
            f"state handoff evaluation {run_path.name}/{condition}"
        )
        if condition in INTERFACE_CONDITIONS:
            from src.experiments.depth_relief.state_interface_evaluation import (
                evaluate_state_interface_condition,
            )

            summary = evaluate_state_interface_condition(
                run_path, condition, on_progress=progress.set_description
            )
        else:
            summary = evaluate_state_handoff_condition(
                run_path, condition, on_progress=progress.set_description
            )
        return TaskResult(units=int(summary["case_count"]), unit="evaluation case")
