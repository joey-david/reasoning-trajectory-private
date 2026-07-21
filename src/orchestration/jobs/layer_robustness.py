"""Resumable Lad layer-intervention job."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from src.experiments.layer_replications.common import read_jsonl
from src.experiments.layer_replications.robustness import (
    KINDS,
    evaluate_chunk,
    load_blocks,
    results_path,
    task_key,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate layer/kind/chunk cells not yet persisted."""
    config = load_config(run_path)
    layer_count = int(config["model"]["layer_count"])
    blocks = load_blocks(run_path)
    per_chunk = int(config["layer_robustness"].get("blocks_per_task", 8))
    chunks = math.ceil(len(blocks) / per_chunk)
    completed = {str(row["key"]) for row in read_jsonl(results_path(run_path))}
    tasks = []
    for kind in KINDS:
        stop = layer_count - 1 if kind == "swap" else layer_count
        for layer in range(stop):
            for chunk in range(chunks):
                if task_key(kind, layer, chunk) not in completed:
                    tasks.append({"kind": kind, "layer": layer, "chunk": chunk})
    total = (2 * layer_count - 1) * chunks
    return tasks, total, total - len(tasks)


def setup_worker(run_path: Path) -> "LayerRobustnessWorker":
    """Load one persistent model and the fixed Pile token blocks."""
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    model, _tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return LayerRobustnessWorker(
        run_path=run_path,
        config=worker_config,
        blocks=load_blocks(run_path),
        model=model,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the worker log path."""
    return run_path / "layer_replications/lad_robustness/logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class LayerRobustnessWorker:
    """Long-lived model state for independent intervention chunks."""

    run_path: Path
    config: RunConfig
    blocks: list[dict[str, Any]]
    model: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Measure and persist one intervention chunk."""
        progress.set_description(
            f"Lad {task['kind']} L{task['layer']} chunk {task['chunk']}"
        )
        row = evaluate_chunk(
            self.model,
            self.blocks,
            kind=str(task["kind"]),
            layer=int(task["layer"]),
            chunk=int(task["chunk"]),
            blocks_per_task=int(
                self.config["layer_robustness"].get("blocks_per_task", 8)
            ),
        )
        append_jsonl(results_path(self.run_path), row)
        return TaskResult(units=int(row["token_count"]), unit="tok")
