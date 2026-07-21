"""Behavior-only depth-relief qualification on the generic GPU orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.pipeline import (
    qualification_output_path,
    read_qualification_results,
)
from src.experiments.depth_relief.qualification import (
    evaluate_qualification_case_hf,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate unqualified cases by stable case ID."""
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {str(row["id"]) for row in read_qualification_results(run_path)}
    tasks = [
        {"case_index": index}
        for index, case in enumerate(cases)
        if str(case["id"]) not in completed
    ]
    return tasks, len(cases), len(cases) - len(tasks)


def setup_worker(run_path: Path) -> "DepthReliefQualificationWorker":
    """Load one model replica and the pinned qualification cases."""
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    required_gpus = int(raw["model"].get("required_gpus", 1))
    raw["model"] = {
        **raw["model"],
        "device_map": raw["model"].get("device_map", "auto")
        if required_gpus > 1
        else {"": 0},
    }
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return DepthReliefQualificationWorker(
        run_path=run_path,
        config=worker_config,
        cases=load_samples(run_path / "dataset.jsonl"),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the per-worker qualification log path."""
    return run_path / "depth_relief/qualification_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class DepthReliefQualificationWorker:
    """Long-lived model state for behavior-only qualification cases."""

    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Evaluate and persist one five-condition qualification case."""
        case = self.cases[int(task["case_index"])]
        progress.set_description(f"depth qualification {case['id']}")
        row = evaluate_qualification_case_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=self.config.get("depth_relief_qualification", {}),
        )
        append_jsonl(qualification_output_path(self.run_path), row)
        return TaskResult(units=len(row["conditions"]), unit="condition")
