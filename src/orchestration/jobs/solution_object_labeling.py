"""Orchestration adapter for local solution-object silver labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.solution_object_labeling import (
    completed_silver_labels,
    generate_hf_output,
    label_messages,
    parse_json_object,
    resolve_window_label,
)
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate token windows without persisted silver labels.

    Args:
        run_path: Labeling run directory.

    Returns:
        Pending tasks, total window count, and completed window count.
    """
    config = load_config(run_path)
    labeling_cfg = config["solution_object_labeling"]
    rows = load_samples(Path(labeling_cfg["input"]).resolve())
    completed = completed_silver_labels(Path(labeling_cfg["output"]).resolve())
    tasks = [
        {"record_index": index}
        for index, row in enumerate(rows)
        if str(row["record_id"]) not in completed
    ]
    return tasks, len(rows), len(rows) - len(tasks)


def setup_worker(run_path: Path) -> SolutionObjectLabelingWorker:
    """Load the sharded labeling model and token-window records.

    Args:
        run_path: Labeling run directory.

    Returns:
        Persistent worker for local silver-label generation.
    """
    config = load_config(run_path)
    labeling_cfg = config["solution_object_labeling"]
    rows = load_samples(Path(labeling_cfg["input"]).resolve())
    backend = str(config["model"].get("backend", "hf"))
    if backend == "vllm":
        from src.models.vllm_backend import load_vllm_model

        model, tokenizer = load_vllm_model(config["model"])
    elif backend == "hf":
        from src.models.hf_loader import load_hf_model_and_tokenizer

        model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    else:
        raise ValueError(f"Unsupported labeling backend: {backend}")
    return SolutionObjectLabelingWorker(
        config=config,
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        backend=backend,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the persistent log path for one labeling worker.

    Args:
        run_path: Labeling run directory.
        host: Remote worker host.
        gpu: GPU index or grouped-device label.

    Returns:
        Per-worker log file path.
    """
    return run_path / "labels" / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class SolutionObjectLabelingWorker:
    """Keep one sharded model resident while labeling token windows."""

    config: RunConfig
    rows: list[dict[str, Any]]
    model: Any
    tokenizer: Any
    backend: str

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Generate, validate, and persist one silver-label proposal.

        Args:
            task: Task containing the bronze record index.
            progress: Coordinator progress callback.

        Returns:
            Generated token count for throughput reporting.
        """
        labeling_cfg = self.config["solution_object_labeling"]
        row = self.rows[int(task["record_index"])]
        outputs: list[str] = []
        token_count = 0
        label = None
        errors: list[str] = []
        retries = int(labeling_cfg.get("alignment_retries", 1))
        for attempt in range(retries + 1):
            messages = label_messages(
                row,
                previous_output=outputs[-1] if outputs else None,
                feedback="; ".join(errors) if outputs else None,
            )
            max_tokens = int(labeling_cfg.get("max_new_tokens", 1200))
            if self.backend == "vllm":
                from src.models.vllm_backend import generate_vllm

                raw_output, generated = generate_vllm(
                    self.model,
                    self.tokenizer,
                    messages,
                    max_tokens=max_tokens,
                )
            else:
                raw_output, generated = generate_hf_output(
                    self.model,
                    self.tokenizer,
                    messages,
                    max_new_tokens=max_tokens,
                )
            outputs.append(raw_output)
            token_count += generated
            try:
                proposal = parse_json_object(raw_output)
                label, errors = resolve_window_label(row, proposal)
            except ValueError as error:
                label = None
                errors = [f"{type(error).__name__}: {error}"]
            if not errors:
                break
        append_jsonl(
            Path(labeling_cfg["output"]),
            {
                "record_id": row["record_id"],
                "model": str(self.config["model"]["name"]),
                "accepted": not errors,
                "validation_errors": errors,
                "silver_label": label,
                "attempts": len(outputs),
                "raw_outputs": outputs,
            },
        )
        progress.set_description(f"label {row['record_id']} {token_count} tok")
        return TaskResult(units=token_count)
