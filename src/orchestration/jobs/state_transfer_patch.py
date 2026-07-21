"""Apply cross-case state deltas to Compose answer anchors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.depth_relief.transfer import score_transfer_patches_hf
from src.experiments.depth_relief.transfer_pipeline import (
    PROJECTION_PATH,
    activation_dir,
    patch_path,
    read_patches,
    read_split,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    split = read_split(run_path)
    config = load_config(run_path).get("state_transfer", {})
    cases = load_samples(run_path / "dataset.jsonl")
    indexes = {str(case["id"]): index for index, case in enumerate(cases)}
    completed = {(str(row["id"]), int(row["layer"])) for row in read_patches(run_path)}
    tasks = [
        {"case_index": indexes[case_id], "layer": int(layer)}
        for case_id in [*split["validation"], *split["test"]]
        for layer in config["layers"]
        if (case_id, int(layer)) not in completed
    ]
    total = (len(split["validation"]) + len(split["test"])) * len(config["layers"])
    return tasks, total, total - len(tasks)


def setup_worker(run_path: Path) -> "StateTransferPatchWorker":
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    required_gpus = int(raw["model"].get("required_gpus", 1))
    raw["model"] = {
        **raw["model"],
        "device_map": raw["model"].get("device_map", "auto") if required_gpus > 1 else {"": 0},
    }
    worker_config = RunConfig.from_dict(run_path, raw)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    projection = np.load(run_path / PROJECTION_PATH)
    return StateTransferPatchWorker(
        run_path=run_path,
        config=worker_config,
        cases=load_samples(run_path / "dataset.jsonl"),
        split=read_split(run_path),
        state_basis=projection["state_basis"],
        random_basis=projection["random_basis"],
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int | str) -> Path:
    return run_path / "depth_relief/state_transfer/patch_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class StateTransferPatchWorker:
    run_path: Path
    config: RunConfig
    cases: list[dict[str, Any]]
    split: dict[str, Any]
    state_basis: np.ndarray
    random_basis: np.ndarray
    model: Any
    tokenizer: Any

    def _states(self, case_id: str) -> dict[str, np.ndarray]:
        with np.load(activation_dir(self.run_path) / f"{case_id}.npz") as arrays:
            return {name: arrays[name].astype(np.float32) for name in arrays.files}

    def _donor_delta(self, recipient_id: str, branch: str, layer: int) -> np.ndarray:
        donor = self.split["donors"][recipient_id][branch]
        states = self._states(str(donor["case_id"]))
        return states[str(donor["condition"])][layer] - states["compose"][layer]

    def run_task(self, task: Task, progress: Any) -> TaskResult:
        case = self.cases[int(task["case_index"])]
        case_id = str(case["id"])
        layer = int(task["layer"])
        progress.set_description(f"state patch {case_id} L{layer}")
        recipient = self._states(case_id)["compose"][layer]
        row = score_transfer_patches_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            case=case,
            config=self.config.get("state_transfer", {}),
            layer=layer,
            recipient=recipient,
            gold_delta=self._donor_delta(case_id, "gold", layer),
            counterfactual_delta=self._donor_delta(case_id, "counterfactual", layer),
            state_basis=self.state_basis[layer],
            random_basis=self.random_basis[layer],
        )
        row["split"] = "validation" if case_id in self.split["validation"] else "test"
        append_jsonl(patch_path(self.run_path), row)
        return TaskResult(units=5, unit="patch")
