"""Run several causal reasoning questions while keeping one model loaded."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.causal_reasoning.evaluation import (
    evaluate_case,
    persist_case,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def _suite_runs(run_path: Path) -> list[Path]:
    config = load_config(run_path)
    return [
        Path(str(path))
        for path in config["causal_reasoning_suite"]["runs"]
    ]


def _completed(run_path: Path) -> set[str]:
    path = run_path / "evaluation" / "cases.jsonl"
    if not path.exists():
        return set()
    rows = load_samples(path)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate causal reasoning rows in {path}")
    return set(ids)


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    tasks = []
    complete = 0
    for child in _suite_runs(run_path):
        cases = load_samples(child / "dataset.jsonl")
        done = _completed(child)
        complete += len(done)
        tasks.extend(
            {
                "run_path": child.as_posix(),
                "case_index": index,
            }
            for index, case in enumerate(cases)
            if str(case["id"]) not in done
        )
    total = complete + len(tasks)
    return tasks, total, complete


def _worker_config(run_path: Path) -> RunConfig:
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    return RunConfig.from_dict(run_path, raw)


def setup_worker(run_path: Path) -> "CausalReasoningWorker":
    config = _worker_config(run_path)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    cases = {
        child.as_posix(): load_samples(child / "dataset.jsonl")
        for child in _suite_runs(run_path)
    }
    child_configs = {
        child.as_posix(): load_config(child)["causal_reasoning"]
        for child in _suite_runs(run_path)
    }
    return CausalReasoningWorker(
        cases=cases,
        configs=child_configs,
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "evaluation" / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class CausalReasoningWorker:
    cases: dict[str, list[dict[str, Any]]]
    configs: dict[str, dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        run_path = Path(str(task["run_path"]))
        case = self.cases[run_path.as_posix()][int(task["case_index"])]
        progress.set_description(
            f"{case['experiment']} {case['id']}"
        )
        row = evaluate_case(
            model=self.model,
            tokenizer=self.tokenizer,
            run_path=run_path,
            case=case,
            config=self.configs[run_path.as_posix()],
        )
        persist_case(run_path, row)
        return TaskResult(units=len(row["results"]), unit="cell")
