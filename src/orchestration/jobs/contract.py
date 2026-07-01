"""Small protocol shared by generic orchestration jobs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


Task = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Report completed work to the coordinator."""

    units: int
    unit: str = "tok"


class JobWorker(Protocol):
    """Long-lived, single-GPU worker state."""

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Execute and persist one coordinator-assigned task."""


class OrchestrationJob(Protocol):
    """Module-level contract implemented by each job adapter."""

    def pending_tasks(self, run_path: Path) -> tuple[list[Task], int, int]:
        """Return pending tasks, total task count, and completed task count."""

    def setup_worker(self, run_path: Path) -> JobWorker:
        """Load long-lived worker state after GPU isolation."""

    def log_path(self, run_path: Path, host: str, gpu: int) -> Path:
        """Return the worker log destination."""
