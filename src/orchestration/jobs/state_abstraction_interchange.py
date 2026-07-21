"""Apply matched implicit-history interchange interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.depth_relief.abstraction_interchange import (
    score_interchange_patches_hf,
)
from src.experiments.depth_relief.abstraction_pipeline import (
    PROJECTION_PATH,
    abstraction_activation_dir,
    interchange_path,
    read_interchange,
    read_pairs,
)
from src.experiments.depth_relief.handoff import trace_position
from src.experiments.depth_relief.pipeline import read_factorization_results
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    config = load_config(run_path).get("state_abstraction", {})
    pairs = read_pairs(run_path)
    completed = {(str(row["id"]), int(row["layer"])) for row in read_interchange(run_path)}
    tasks = [
        {"pair_index": index, "layer": int(layer)}
        for index, pair in enumerate(pairs)
        if pair["split"] in {"validation", "test"}
        for layer in config["causal_layers"]
        if (str(pair["id"]), int(layer)) not in completed
    ]
    total = sum(pair["split"] in {"validation", "test"} for pair in pairs) * len(
        config["causal_layers"]
    )
    return tasks, total, total - len(tasks)


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


def setup_worker(run_path: Path) -> "StateAbstractionInterchangeWorker":
    config = _worker_config(run_path)
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    with np.load(run_path / PROJECTION_PATH) as projection:
        state_basis = projection["state_basis"].astype(np.float32)
        random_basis = projection["random_basis"].astype(np.float32)
    cases = load_samples(run_path / "dataset.jsonl")
    return StateAbstractionInterchangeWorker(
        run_path=run_path,
        config=config,
        cases={str(case["id"]): case for case in cases},
        captures={
            str(row["id"]): row for row in read_factorization_results(run_path)
        },
        pairs=read_pairs(run_path),
        state_basis=state_basis,
        random_basis=random_basis,
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "depth_relief/state_abstraction/interchange_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateAbstractionInterchangeWorker:
    run_path: Path
    config: RunConfig
    cases: dict[str, dict[str, Any]]
    captures: dict[str, dict[str, Any]]
    pairs: list[dict[str, Any]]
    state_basis: np.ndarray
    random_basis: np.ndarray
    model: Any
    tokenizer: Any
    history_states: dict[str, np.ndarray] = field(default_factory=dict)

    def _compose_trace(self, case_id: str) -> np.ndarray:
        with np.load(
            abstraction_activation_dir(self.run_path) / f"{case_id}.npz"
        ) as arrays:
            return arrays["compose_trace"].astype(np.float32)

    def _history_state(self, case_id: str) -> np.ndarray:
        if case_id in self.history_states:
            return self.history_states[case_id]
        case = self.cases[case_id]
        state = trace_position(
            {"compose_trace": self._compose_trace(case_id)},
            self.captures[case_id],
            f"history_step_{int(case['history_steps'])}",
        )
        self.history_states[case_id] = state
        return state

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        pair = self.pairs[int(task["pair_index"])]
        recipient_id = str(pair["recipient_id"])
        different_id = str(pair["different_state_source_id"])
        same_id = str(pair["same_state_source_id"])
        layer = int(task["layer"])
        progress.set_description(f"state interchange {recipient_id} L{layer}")
        recipient = self.cases[recipient_id]
        position_name = f"history_step_{int(recipient['history_steps'])}"
        token_index = next(
            int(position["token_index"])
            for position in self.captures[recipient_id]["compose_positions"]
            if position["name"] == position_name
        )
        row = score_interchange_patches_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            recipient=recipient,
            different_source=self.cases[different_id],
            config=self.config.get("state_materialization", {}),
            layer=layer,
            token_index=token_index,
            recipient_state=self._history_state(recipient_id)[layer],
            different_state=self._history_state(different_id)[layer],
            same_state=self._history_state(same_id)[layer],
            state_basis=self.state_basis[layer],
            random_basis=self.random_basis[layer],
        )
        append_jsonl(interchange_path(self.run_path), row)
        return TaskResult(units=len(row["conditions"]), unit="patch")
