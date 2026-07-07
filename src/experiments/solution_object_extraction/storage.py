"""Small, explicit artifact readers and writers for extraction runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


EXPERIMENT_DIR = Path("analysis/experiments/solution_object_extraction")


def output_dir(run_path: Path) -> Path:
    """Return and create the plan-prescribed output directory."""
    path = run_path / EXPERIMENT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_experiment_config(run_path: Path) -> dict[str, Any]:
    """Load and validate the solution-object section of a run config."""
    path = run_path / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    experiment = config.get("solution_object_extraction")
    if not isinstance(experiment, dict):
        raise ValueError(f"{path} lacks solution_object_extraction")
    return {"run": config, "experiment": experiment}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL rows."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write indented JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSON objects one per line atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed NPZ without leaving a partial archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
