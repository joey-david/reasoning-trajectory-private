"""Orchestration adapter for objective-family boundary interventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.experiments.boundary_interventions import (
    completed_interventions,
    generate_boundary_continuation,
    load_intervention_rows,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.orchestration.jobs.contract import Task, TaskResult
from src.runtime.config import RunConfig, load_config
from src.runtime.data import load_samples


def pending_tasks(run_path: Path) -> tuple[list[Task], int, int]:
    """Enumerate incomplete point-condition-continuation cells.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The computed aligned values described above.
    """
    config = load_config(run_path)
    intervention_cfg = config["boundary_intervention"]
    points = load_samples(Path(intervention_cfg["manifest"]).resolve())
    conditions = [str(value) for value in intervention_cfg["conditions"]]
    continuation_count = int(
        intervention_cfg.get("continuations_per_condition", 1)
    )
    completed = completed_interventions(
        run_path / "interventions" / "continuations.jsonl"
    )
    tasks: list[Task] = []
    complete = 0
    for point_index, point in enumerate(points):
        for condition in conditions:
            for continuation in range(continuation_count):
                key = (int(point["point_id"]), condition, continuation)
                if key in completed:
                    complete += 1
                else:
                    tasks.append(
                        {
                            "point_index": point_index,
                            "condition": condition,
                            "continuation": continuation,
                        }
                    )
    total = len(points) * len(conditions) * continuation_count
    return tasks, total, complete


def setup_worker(run_path: Path) -> BoundaryInterventionWorker:
    """Load one model replica, source index, and prepared boundary manifest.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        A worker initialized for boundary interventions.
    """
    config = load_config(run_path)
    raw = {key: value for key, value in config.raw.items() if key != "_run_path"}
    raw["model"] = {**raw["model"], "device_map": {"": 0}}
    worker_config = RunConfig.from_dict(run_path, raw)
    intervention_cfg = worker_config["boundary_intervention"]
    source_run = Path(intervention_cfg["source_run"])
    points = load_samples(Path(intervention_cfg["manifest"]).resolve())
    rows = load_intervention_rows(source_run)
    missing = {
        (str(point["sample_id"]), int(point["seed"]))
        for point in points
        if (str(point["sample_id"]), int(point["seed"])) not in rows
    }
    if missing:
        raise ValueError(f"Boundary manifest references missing rows: {sorted(missing)[:5]}")
    model, tokenizer = load_hf_model_and_tokenizer(worker_config["model"])
    return BoundaryInterventionWorker(
        run_path=run_path,
        config=worker_config,
        source_run=source_run,
        points=points,
        rows=rows,
        model=model,
        tokenizer=tokenizer,
    )


def log_path(run_path: Path, host: str, gpu: int) -> Path:
    """Return one persistent worker's intervention log path.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        host: Remote worker host name.
        gpu: GPU index on the worker host.

    Returns:
        The path of the written or discovered artifact.
    """
    return (
        run_path
        / "interventions"
        / "orchestrator_logs"
        / f"{host}_{gpu}.log"
    )


@dataclass(slots=True)
class BoundaryInterventionWorker:
    """Keep one model resident while evaluating independent boundary points."""

    run_path: Path
    config: RunConfig
    source_run: Path
    points: list[dict[str, Any]]
    rows: dict[tuple[str, int], dict[str, Any]]
    model: Any
    tokenizer: Any

    def run_task(self, task: Task, _progress: Any) -> TaskResult:
        """Generate and persist one intervention cell.

        Args:
            task: Serialized orchestration task.
            _progress: Unused orchestration progress callback required by the job contract.

        Returns:
            A task result containing its stable key and serialized record.
        """
        intervention_cfg = self.config["boundary_intervention"]
        point = self.points[int(task["point_index"])]
        continuation = int(task["continuation"])
        seed = (
            int(intervention_cfg.get("base_seed", 0))
            + int(point["point_id"]) * 100
            + continuation
        )
        record = generate_boundary_continuation(
            run_path=self.run_path,
            model=self.model,
            tokenizer=self.tokenizer,
            source_run=self.source_run,
            rows=self.rows,
            point=point,
            condition=str(task["condition"]),
            continuation=continuation,
            seed=seed,
            component=str(intervention_cfg["component"]),
            layer=int(intervention_cfg["layer"]),
            intervention_cfg=intervention_cfg,
            analysis_cfg=self.config.get("analysis", {}),
        )
        return TaskResult(units=len(record["generated_token_ids"]))
