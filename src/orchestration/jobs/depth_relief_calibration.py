"""Sentinel frontier calibration on the generic GPU orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.calibration import evaluate_calibration_case_hf
from src.experiments.depth_relief.pipeline import (
    calibration_output_path,
    read_calibration_results,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate unfinished frontier cases by stable case ID."""
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {str(row["id"]) for row in read_calibration_results(run_path)}
    tasks = [
        {"case_index": index}
        for index, case in enumerate(cases)
        if str(case["id"]) not in completed
    ]
    return tasks, len(cases), len(cases) - len(tasks)


def setup_worker(run_path: Path) -> "DepthReliefCalibrationWorker":
    """Load one pinned model and the prepared discovery grid."""
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
    return DepthReliefCalibrationWorker(
        run_path=run_path,
        config=worker_config,
        cases=load_samples(run_path / "dataset.jsonl"),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the persistent per-worker calibration log path."""
    return run_path / "depth_relief/calibration_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class DepthReliefCalibrationWorker:
    """Keep one model resident while scoring independent frontier cases."""

    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Evaluate and persist one five-condition frontier case."""
        case = self.cases[int(task["case_index"])]
        progress.set_description(f"depth calibration {case['id']}")
        row = evaluate_calibration_case_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=self.config.get("depth_relief_calibration", {}),
        )
        append_jsonl(calibration_output_path(self.run_path), row)
        return TaskResult(units=len(row["conditions"]), unit="condition")
