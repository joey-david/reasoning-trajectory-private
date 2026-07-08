"""Generation adapter for the generic GPU task orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.datasets.loaders import load_run_samples
from src.models.generation_pipeline import (
    generate_task,
    generation_key_for,
    sample_id_from_sample,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.config import RunConfig, load_config
from src.runtime.run_io import load_generation_index


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate incomplete generation rollouts.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The computed aligned values described above.
    """
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    generation_cfg = config["generation"]
    samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    existing = load_generation_index(run_path)
    tasks = []
    complete = 0
    for sample_index, sample in enumerate(samples):
        for sample_iter in range(samples_per_item):
            key = generation_key_for(sample, sample_index, sample_iter, generation_cfg)
            if key in existing:
                complete += 1
            else:
                tasks.append({"sample_index": sample_index, "sample_iter": sample_iter})
    tasks.sort(key=lambda task: (task["sample_iter"], task["sample_index"]))
    return tasks, len(samples) * samples_per_item, complete


def setup_worker(run_path: Path) -> GenerationWorker:
    """Load one generation model replica and its normalized dataset.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        A worker initialized for generation tasks.
    """
    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = worker_model_config(raw["model"])
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return GenerationWorker(
        run_path=run_path,
        config=worker_config,
        samples=samples,
        model=model,
        tokenizer=tokenizer,
    )


def worker_model_config(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve worker-local device placement after CUDA_VISIBLE_DEVICES remapping."""
    required_gpus = int(model_cfg.get("required_gpus", 1))
    if required_gpus > 1:
        return {**model_cfg, "device_map": model_cfg.get("device_map", "auto")}
    return {**model_cfg, "device_map": {"": 0}}


def log_path(run_path: Path, host: str, gpu: int) -> Path:
    """Build the per-worker orchestration log path.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        host: Remote worker host name.
        gpu: GPU index on the worker host.

    Returns:
        The path of the written or discovered artifact.
    """
    return run_path / "generation" / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class GenerationWorker:
    run_path: Path
    config: RunConfig
    samples: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Execute one orchestration task and return its serialized result.

        Args:
            task: Serialized orchestration task.
            progress: Callback used to report worker progress.

        Returns:
            A task result containing its stable key and serialized record.
        """
        sample_index = int(task["sample_index"])
        sample_iter = int(task["sample_iter"])
        sample = self.samples[sample_index]
        samples_per_item = int(self.config["generation"].get("num_samples_per_item", 1))
        label = (
            f"item {sample_index + 1}/{len(self.samples)} "
            f"{sample_id_from_sample(sample)} "
            f"iter {sample_iter + 1}/{samples_per_item}"
        )
        output = generate_task(
            run_path=self.run_path,
            config=self.config,
            model=self.model,
            tokenizer=self.tokenizer,
            sample=sample,
            sample_index=sample_index,
            sample_iter=sample_iter,
            progress=progress,
            progress_label=label,
        )
        return TaskResult(units=len(output.generated_token_ids))
