"""Resumable two-call and stepwise explicit-state handoff inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.depth_relief.explicit_handoff import (
    evaluate_explicit_handoff_case_hf,
    explicit_handoff_output_path,
    read_explicit_handoff_records,
)
from src.experiments.depth_relief.pipeline import read_factorization_results
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate cases that do not yet have an inference-phase record."""
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {
        str(row["id"])
        for row in read_explicit_handoff_records(run_path)
        if row["phase"] == "inference"
    }
    tasks = [
        {"case_index": index}
        for index, case in enumerate(cases)
        if str(case["id"]) not in completed
    ]
    return tasks, len(cases), len(cases) - len(tasks)


def setup_worker(run_path: Path) -> "StateExplicitHandoffWorker":
    """Load one model worker and the saved factorization source rows."""
    config = load_config(run_path)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    cases = load_samples(run_path / "dataset.jsonl")
    factorization = {
        str(row["id"]): row for row in read_factorization_results(run_path)
    }
    if {str(case["id"]) for case in cases} != set(factorization):
        raise ValueError("Explicit handoff needs complete factorization results")
    experiment = dict(config.get("explicit_handoff", {}))
    if "prompt" not in experiment:
        experiment["prompt"] = config.get("state_materialization", {}).get(
            "prompt", {}
        )
    return StateExplicitHandoffWorker(
        run_path=run_path,
        cases=cases,
        factorization=factorization,
        config=experiment,
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    """Return the per-worker explicit-handoff log path."""
    return run_path / "depth_relief/explicit_handoff/logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateExplicitHandoffWorker:
    """Long-lived inference state for history-free handoff calls."""

    run_path: Path
    cases: list[dict[str, Any]]
    factorization: dict[str, dict[str, Any]]
    config: dict[str, Any]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        """Evaluate one case and append its inference event."""
        case_index = int(task["case_index"])
        case = self.cases[case_index]
        case_id = str(case["id"])
        prefix = f"explicit handoff {case_index + 1}/{len(self.cases)}"
        progress.set_description(f"{prefix} preparing {case_id}")
        row = evaluate_explicit_handoff_case_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            factorization_row=self.factorization[case_id],
            config=self.config,
            on_progress=lambda stage: progress.set_description(f"{prefix} {stage}"),
        )
        append_jsonl(explicit_handoff_output_path(self.run_path), row)
        return TaskResult(
            units=int(case["history_steps"]) + 3,
            unit="model call",
        )
