"""Causal depth-relief adapter for the generic resumable GPU orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.hf import evaluate_case_hf
from src.experiments.depth_relief.pipeline import output_path, read_results
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate unscored benchmark cases by stable case ID."""
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {str(row["id"]) for row in read_results(run_path)}
    tasks = [
        {"case_index": index}
        for index, case in enumerate(cases)
        if str(case["id"]) not in completed
    ]
    return tasks, len(cases), len(cases) - len(tasks)


def setup_worker(run_path: Path) -> "DepthReliefWorker":
    """Load one isolated model replica and the shared deterministic benchmark."""
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return DepthReliefWorker(
        run_path=run_path,
        config=worker_config,
        cases=load_samples(run_path / "dataset.jsonl"),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the per-worker depth-relief log path."""
    return run_path / "depth_relief/orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class DepthReliefWorker:
    """Long-lived model state for independent benchmark-case tasks."""

    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Evaluate and persist one complete controlled case."""
        case = self.cases[int(task["case_index"])]
        experiment = self.config.get("depth_relief", {})
        causal_count = int(experiment.get("causal_examples_per_cell", 0))
        run_causal = int(case["example_index"]) < causal_count
        progress.set_description(
            f"depth relief {case['id']}" + (" causal" if run_causal else " screen")
        )
        row = evaluate_case_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=experiment,
            run_causal=run_causal,
        )
        append_jsonl(output_path(self.run_path), row)
        return TaskResult(units=len(row["conditions"]), unit="condition")
