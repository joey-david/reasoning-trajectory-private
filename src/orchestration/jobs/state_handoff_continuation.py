"""One-GPU orchestration adapter for recursive state-interface evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.state_handoff_continuation import (
    CONTINUATION_ROOT,
    continuation_dir,
    evaluate_continuation_profile,
)
from src.experiments.depth_relief.state_handoff_evaluation import (
    _load_evaluation_model,
)
from src.orchestration.jobs.contract import Task, TaskResult


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Return prepared continuation profiles whose summaries are incomplete."""
    root = run_path / CONTINUATION_ROOT
    profiles = sorted(
        path.name
        for path in root.iterdir()
        if path.is_dir() and (path / "manifest.json").exists()
    ) if root.exists() else []
    if not profiles:
        raise RuntimeError("No continuation profile is prepared")
    pending = []
    for profile in profiles:
        path = continuation_dir(run_path, profile) / "summary.json"
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            pending.append({"profile": profile})
    return pending, len(profiles), len(profiles) - len(pending)


def setup_worker(run_path: Path) -> "StateHandoffContinuationWorker":
    """Load the trained explicit adapter once per isolated GPU worker."""
    model, tokenizer = _load_evaluation_model(run_path, "explicit_handoff")
    return StateHandoffContinuationWorker(
        run_path=run_path, model=model, tokenizer=tokenizer
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the recursive-evaluation worker log path."""
    return run_path / CONTINUATION_ROOT / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateHandoffContinuationWorker:
    """Evaluate prepared profiles with one resident adapter."""

    run_path: Path
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        profile = str(task["profile"])
        progress.set_description(f"recursive state handoff {profile}")
        summary = evaluate_continuation_profile(
            self.run_path,
            profile,
            model=self.model,
            tokenizer=self.tokenizer,
            on_progress=progress.set_description,
        )
        return TaskResult(units=int(summary["case_count"]), unit="recursive case")
