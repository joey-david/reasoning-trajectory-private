"""Baseline identity-rule competence screen for Yang CMA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.layer_replications.common import read_jsonl
from src.experiments.layer_replications.symbolic import (
    load_pairs,
    screen_pair,
    screen_path,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate unscreened context pairs."""
    pairs = load_pairs(run_path)
    completed = {str(row["id"]) for row in read_jsonl(screen_path(run_path))}
    tasks = [
        {"pair_index": index}
        for index, row in enumerate(pairs)
        if str(row["id"]) not in completed
    ]
    return tasks, len(pairs), len(pairs) - len(tasks)


def setup_worker(run_path: Path) -> "SymbolicScreenWorker":
    """Load one persistent model and attach its tokenizer for paired scoring."""
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return SymbolicScreenWorker(
        run_path=run_path,
        pairs=load_pairs(run_path),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the worker log path."""
    return (
        run_path / "layer_replications/yang_symbolic/logs/screen" / f"{host}_{gpu}.log"
    )


@dataclass(slots=True)
class SymbolicScreenWorker:
    """Long-lived model state for clean task screening."""

    run_path: Path
    pairs: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Score and persist one clean context pair."""
        row = self.pairs[int(task["pair_index"])]
        progress.set_description(f"Yang screen {row['id']}")
        result = screen_pair(self.model, self.tokenizer, row)
        append_jsonl(screen_path(self.run_path), result)
        return TaskResult(units=2, unit="prompt")
