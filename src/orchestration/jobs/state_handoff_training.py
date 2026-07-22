"""One-GPU orchestration adapter for matched state-handoff LoRA pilots."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from typing import Any

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


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Return unfinished training-plus-evaluation conditions."""
    pending = [
        {"condition": condition}
        for condition in configured_training_conditions(run_path)
        if not _condition_complete(run_path, condition)
    ]
    total = len(configured_training_conditions(run_path))
    return pending, total, total - len(pending)


def setup_worker(run_path: Path) -> "StateHandoffTrainingWorker":
    """Validate the gate before accepting long-running condition tasks."""
    require_phase1_training_gate(run_path)
    return StateHandoffTrainingWorker(run_path=run_path)


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the single-GPU training worker log path."""
    return run_path / "training/orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateHandoffTrainingWorker:
    """Run one complete condition, then release its model before evaluation."""

    run_path: Path

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        condition = str(task["condition"])
        progress.set_description(f"state handoff training {condition}")
        train_state_handoff_condition(
            self.run_path, condition, on_progress=progress.set_description
        )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        progress.set_description(f"state handoff evaluation {condition}")
        if condition in INTERFACE_CONDITIONS:
            from src.experiments.depth_relief.state_interface_evaluation import (
                evaluate_state_interface_condition,
            )

            summary = evaluate_state_interface_condition(
                self.run_path, condition, on_progress=progress.set_description
            )
        else:
            summary = evaluate_state_handoff_condition(
                self.run_path, condition, on_progress=progress.set_description
            )
        return TaskResult(units=int(summary["case_count"]), unit="evaluation case")
