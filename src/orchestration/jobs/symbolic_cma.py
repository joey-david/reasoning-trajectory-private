"""Head-level causal-mediation job for Yang et al."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.layer_replications.common import read_jsonl
from src.experiments.layer_replications.symbolic import (
    causal_mediation,
    cma_path,
    cma_task_key,
    cma_tasks,
    selection_path,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate selected pair/mechanism maps not yet persisted."""
    pairs = read_jsonl(selection_path(run_path))
    tasks = cma_tasks(run_path)
    completed = {str(row["key"]) for row in read_jsonl(cma_path(run_path))}
    pending = [
        task
        for task in tasks
        if cma_task_key(
            str(pairs[int(task["pair_index"])]["id"]), str(task["mechanism"])
        )
        not in completed
    ]
    return pending, len(tasks), len(tasks) - len(pending)


def setup_worker(run_path: Path) -> "SymbolicCMAWorker":
    """Load one persistent hooked model and the fixed clean-pair selection."""
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return SymbolicCMAWorker(
        run_path=run_path,
        config=worker_config,
        pairs=read_jsonl(selection_path(run_path)),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the worker log path."""
    return run_path / "layer_replications/yang_symbolic/logs/cma" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class SymbolicCMAWorker:
    """Long-lived model state for complete layer-by-head patch maps."""

    run_path: Path
    config: RunConfig
    pairs: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Measure and persist one pair/mechanism head map."""
        row = self.pairs[int(task["pair_index"])]
        mechanism = str(task["mechanism"])
        progress.set_description(f"Yang {mechanism} {row['id']}")
        result = causal_mediation(
            self.model,
            self.tokenizer,
            row,
            mechanism=mechanism,
            head_batch_size=int(
                self.config["symbolic_mechanisms"].get("head_batch_size", 7)
            ),
        )
        append_jsonl(cma_path(self.run_path), result)
        return TaskResult(
            units=int(self.config["model"]["layer_count"])
            * int(self.config["model"]["attention_heads"]),
            unit="head-patch",
        )
