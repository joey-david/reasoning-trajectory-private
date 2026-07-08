"""Orchestration adapter for teacher-forced gold-answer activation capture."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.datasets.loaders import load_run_samples
from src.experiments.gold_answers import (
    capture_gold_answer,
    completed_gold_answers,
    write_gold_answer_metadata,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.config import RunConfig, load_config


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate unique questions whose gold states are not yet persisted.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The computed aligned values described above.
    """
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    completed = completed_gold_answers(
        run_path / "gold_answers" / "manifest.jsonl"
    )
    tasks = [
        {"sample_index": index}
        for index, sample in enumerate(samples)
        if str(sample.get("id") or sample.get("problem_id")) not in completed
    ]
    return tasks, len(samples), len(samples) - len(tasks)


def setup_worker(run_path: Path) -> GoldAnswerWorker:
    """Load one model replica and the run's pinned gold-answer dataset.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        A worker initialized for gold-answer capture.
    """
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = worker_model_config(raw["model"])
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    capture_cfg = worker_config["gold_answer_capture"]
    layers = [int(layer) for layer in capture_cfg.get("layers", [-1])]
    storage_dtype = str(
        capture_cfg.get("activation_storage_dtype", "int8_scaled")
    )
    write_gold_answer_metadata(
        run_path,
        model_name=str(worker_config["model"]["name"]),
        layers=layers,
        storage_dtype=storage_dtype,
    )
    return GoldAnswerWorker(
        run_path=run_path,
        config=worker_config,
        samples=samples,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        storage_dtype=storage_dtype,
    )


def worker_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve worker-local device placement after CUDA_VISIBLE_DEVICES remapping."""
    required_gpus = int(model_cfg.get("required_gpus", 1))
    if required_gpus > 1:
        return {**model_cfg, "device_map": model_cfg.get("device_map", "auto")}
    return {**model_cfg, "device_map": {"": 0}}


def log_path(run_path: Path, host: str, gpu: int) -> Path:
    """Return one persistent worker's gold-answer log path.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        host: Remote worker host name.
        gpu: GPU index on the worker host.

    Returns:
        The path of the written or discovered artifact.
    """
    return (
        run_path
        / "gold_answers"
        / "orchestrator_logs"
        / f"{host}_{gpu}.log"
    )


@dataclass(slots=True)
class GoldAnswerWorker:
    """Keep one model replica resident while capturing independent answers."""

    run_path: Path
    config: RunConfig
    samples: list[dict[str, Any]]
    model: Any
    tokenizer: Any
    layers: list[int]
    storage_dtype: str

    def run_task(self, task: Task, _progress: Any) -> TaskResult:
        """Capture and persist one indexed gold solution.

        Args:
            task: Serialized orchestration task.
            _progress: Unused orchestration progress callback required by the job contract.

        Returns:
            A task result containing its stable key and serialized record.
        """
        record = capture_gold_answer(
            run_path=self.run_path,
            model=self.model,
            tokenizer=self.tokenizer,
            sample=self.samples[int(task["sample_index"])],
            layers=self.layers,
            storage_dtype=self.storage_dtype,
            max_tokens=int(
                self.config["gold_answer_capture"].get("max_tokens", 16384)
            ),
        )
        return TaskResult(units=int(record["token_count"]))
