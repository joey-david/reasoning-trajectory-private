"""H3 causal-patching adapter for the generic GPU task orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from reasoning_trajectory.artifacts import read_generation_rows
from src.experiments.process_isomers.causal_patching import (
    ProjectionSubspace,
    generate_patched_continuation,
    load_completed_patches,
    load_projection_subspace,
    resolve_patch_modes,
    resolve_patch_target,
    validate_pair_rows,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate incomplete H3 cells using their persisted four-part key.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The computed aligned values described above.
    """
    config = load_config(run_path)
    patch_cfg = config["patching"]
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())
    pairs = pairs[: int(patch_cfg.get("max_pairs", len(pairs)))]
    modes = resolve_patch_modes(patch_cfg, None)
    conditions = [str(condition) for condition in patch_cfg["conditions"]]
    continuation_count = int(patch_cfg.get("continuations_per_condition", 5))
    completed = load_completed_patches(run_path / "patching" / "continuations.jsonl")
    tasks = []
    complete = 0
    for pair_index, pair in enumerate(pairs):
        for mode in modes:
            for condition in conditions:
                for continuation in range(continuation_count):
                    key = (int(pair["pair_id"]), mode, condition, continuation)
                    if key in completed:
                        complete += 1
                    else:
                        tasks.append(
                            {
                                "pair_index": pair_index,
                                "patch_mode": mode,
                                "condition": condition,
                                "continuation": continuation,
                            }
                        )
    total = len(pairs) * len(modes) * len(conditions) * continuation_count
    return tasks, total, complete


def setup_worker(run_path: Path) -> CausalPatchingWorker:
    """Load one H3 model replica, activation index, and projection.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        A worker initialized for causal patching.
    """
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    patch_cfg = worker_config["patching"]
    activation_run = Path(patch_cfg["activation_run"])
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())
    pairs = pairs[: int(patch_cfg.get("max_pairs", len(pairs)))]
    component, layer = resolve_patch_target(patch_cfg)
    modes = resolve_patch_modes(patch_cfg, None)
    projection = (
        load_projection_subspace(
            Path(patch_cfg["projection_path"]),
            component=component,
            layer=layer,
        )
        if "subspace" in modes
        else None
    )
    rows = {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in read_generation_rows(activation_run)
    }
    validate_pair_rows(pairs, rows)
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return CausalPatchingWorker(
        run_path=run_path,
        config=worker_config,
        activation_run=activation_run,
        pairs=pairs,
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        component=component,
        layer=layer,
        projection=projection,
    )


def log_path(run_path: Path, host: str, gpu: int) -> Path:
    """Build the per-worker orchestration log path.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        host: Remote worker host name.
        gpu: GPU index on the worker host.

    Returns:
        The path of the written or discovered artifact.
    """
    return run_path / "patching" / "orchestrator_logs" / f"{host}_{gpu}.log"


@dataclass(slots=True)
class CausalPatchingWorker:
    run_path: Path
    config: RunConfig
    activation_run: Path
    pairs: list[dict[str, Any]]
    rows: dict[tuple[str, int], dict[str, Any]]
    model: Any
    tokenizer: Any
    component: str
    layer: int
    projection: ProjectionSubspace | None
    vector_cache: dict[tuple[str, int, str, int, int], torch.Tensor] = field(
        default_factory=dict
    )

    def run_task(self, task: Task, _progress: Any) -> TaskResult:
        """Execute one orchestration task and return its serialized result.

        Args:
            task: Serialized orchestration task.
            _progress: Unused orchestration progress callback required by the job contract.

        Returns:
            A task result containing its stable key and serialized record.
        """
        pair_index = int(task["pair_index"])
        continuation = int(task["continuation"])
        patch_cfg = self.config["patching"]
        record = generate_patched_continuation(
            model=self.model,
            tokenizer=self.tokenizer,
            activation_run=self.activation_run,
            rows=self.rows,
            pairs=self.pairs,
            pair=self.pairs[pair_index],
            patch_mode=str(task["patch_mode"]),
            condition=str(task["condition"]),
            continuation=continuation,
            seed=int(patch_cfg.get("base_seed", 0)) + pair_index * 100 + continuation,
            component=self.component,
            layer=self.layer,
            projection=self.projection,
            patch_cfg=patch_cfg,
            analysis_cfg=self.config.get("analysis", {}),
            vector_cache=self.vector_cache,
        )
        append_jsonl(
            self.run_path / "patching" / "continuations.jsonl",
            record,
        )
        return TaskResult(units=len(record["generated_token_ids"]))
