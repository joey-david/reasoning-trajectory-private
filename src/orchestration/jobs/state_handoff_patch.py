"""Causally transport each recipient's own earlier state to its answer anchor."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.depth_relief.handoff import (
    score_handoff_patches_hf,
    trace_position,
)
from src.experiments.depth_relief.transfer_pipeline import (
    HANDOFF_MANIFEST_PATH,
    PROJECTION_PATH,
    activation_dir,
    handoff_path,
    read_captures,
    read_handoff_patches,
    read_split,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    split = read_split(run_path)
    cases = load_samples(run_path / "dataset.jsonl")
    indexes = {str(case["id"]): index for index, case in enumerate(cases)}
    completed = {str(row["id"]) for row in read_handoff_patches(run_path)}
    tasks = [
        {"case_index": indexes[case_id]}
        for case_id in split["test"]
        if case_id not in completed
    ]
    return tasks, len(split["test"]), len(split["test"]) - len(tasks)


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


def setup_worker(run_path: Path) -> "StateHandoffPatchWorker":
    config = _worker_config(run_path)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    projection = np.load(run_path / PROJECTION_PATH)
    manifest = json.loads((run_path / HANDOFF_MANIFEST_PATH).read_text())
    return StateHandoffPatchWorker(
        run_path=run_path,
        config=config,
        cases=load_samples(run_path / "dataset.jsonl"),
        captures={str(row["id"]): row for row in read_captures(run_path)},
        layer=int(manifest["layer"]),
        state_basis=projection["state_basis"],
        random_basis=projection["random_basis"],
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "depth_relief/state_transfer/handoff_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateHandoffPatchWorker:
    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    captures: dict[str, dict[str, Any]]
    layer: int
    state_basis: np.ndarray
    random_basis: np.ndarray
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        case = self.cases[int(task["case_index"])]
        case_id = str(case["id"])
        progress.set_description(f"self handoff {case_id} L{self.layer}")
        with np.load(activation_dir(self.run_path) / f"{case_id}.npz") as stored:
            arrays = {name: stored[name].astype(np.float32) for name in stored.files}
        history = trace_position(
            arrays,
            self.captures[case_id],
            f"history_step_{int(case['history_steps'])}",
        )[self.layer]
        row = score_handoff_patches_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=self.config.get("state_transfer", {}),
            layer=self.layer,
            answer_state=arrays["compose"][self.layer],
            history_state=history,
            state_basis=self.state_basis[self.layer],
            random_basis=self.random_basis[self.layer],
        )
        append_jsonl(handoff_path(self.run_path), row)
        return TaskResult(units=3, unit="patch")
