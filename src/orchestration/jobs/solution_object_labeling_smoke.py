"""One-record end-to-end smoke job for solution-object labeling."""

from __future__ import annotations

from pathlib import Path

from src.orchestration.jobs.contract import Task
from src.orchestration.jobs.solution_object_labeling import (
    log_path,
    pending_tasks as all_pending_tasks,
    setup_worker,
)


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Return at most one pending labeling task."""
    tasks, _total, _complete = all_pending_tasks(run_path)
    pending = tasks[:1]
    return pending, len(pending), 0


__all__ = ["log_path", "pending_tasks", "setup_worker"]
