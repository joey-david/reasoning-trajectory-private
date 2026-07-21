"""Capture matched-history behavior and semantic residual streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.depth_relief.abstraction import capture_abstraction_case_hf
from src.experiments.depth_relief.abstraction_pipeline import (
    abstraction_activation_dir,
    reconcile_abstraction_artifacts,
)
from src.experiments.depth_relief.pipeline import factorization_output_path
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    cases = load_samples(run_path / "dataset.jsonl")
    completed = reconcile_abstraction_artifacts(run_path, cases)
    tasks = [
        {"case_index": index}
        for index, case in enumerate(cases)
        if str(case["id"]) not in completed
    ]
    return tasks, len(cases), len(cases) - len(tasks)


def _worker_config(run_path: Path) -> RunConfig:
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    required_gpus = int(raw["model"].get("required_gpus", 1))
    raw["model"] = {
        **raw["model"],
        "device_map": raw["model"].get("device_map", "auto")
        if required_gpus > 1
        else {"": 0},
    }
    return RunConfig.from_dict(run_path, raw)


def setup_worker(run_path: Path) -> "StateAbstractionCaptureWorker":
    config = _worker_config(run_path)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    return StateAbstractionCaptureWorker(
        run_path=run_path,
        config=config,
        cases=load_samples(run_path / "dataset.jsonl"),
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "depth_relief/state_abstraction/capture_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateAbstractionCaptureWorker:
    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        case = self.cases[int(task["case_index"])]
        progress.set_description(f"state abstraction {case['id']}")
        row, activations = capture_abstraction_case_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=self.config.get("state_materialization", {}),
        )
        directory = abstraction_activation_dir(self.run_path)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{case['id']}.npz", **activations)
        append_jsonl(factorization_output_path(self.run_path), row)
        return TaskResult(units=len(row["conditions"]), unit="condition")
