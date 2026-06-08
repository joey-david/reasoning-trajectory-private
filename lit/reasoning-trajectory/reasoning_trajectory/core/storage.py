from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from .schema import Trajectory, to_dict, trajectory_from_dict


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def git_commit(cwd: str | Path = ".") -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return "unknown"


def run_dir(base: str | Path, name: str, cfg: dict[str, Any]) -> Path:
    path = Path(base) / f"{name}-{config_hash(cfg)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trajectories(trajectories: Iterable[Trajectory], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            trajectory.require_valid()
            f.write(json.dumps(to_dict(trajectory), ensure_ascii=False) + "\n")
    return path


def save_json(trajectory: Trajectory, path: str | Path) -> Path:
    trajectory.require_valid()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(trajectory), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_trajectories(path: str | Path) -> list[Trajectory]:
    path = Path(path)
    if path.is_dir():
        jsonl = path / "trajectories.jsonl"
        if jsonl.exists():
            return load_jsonl(jsonl)
        return [load_json(p) for p in sorted(path.glob("*.json"))]
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    if path.suffix == ".json":
        return [load_json(path)]
    raise ValueError(f"Unsupported trajectory input: {path}")


def save_table(rows: Iterable[dict], path: str | Path) -> Path:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("parquet export requires pandas and pyarrow") from exc
        pd.DataFrame(rows).to_parquet(path, index=False)
    else:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def save_npz(tensors: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **tensors)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_jsonl(path: str | Path) -> list[Trajectory]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return [trajectory_from_dict(json.loads(line)) for line in f if line.strip()]


def load_json(path: str | Path) -> Trajectory:
    path = Path(path)
    return trajectory_from_dict(json.loads(path.read_text(encoding="utf-8")))


# Backward-compatible names for older notebooks/scripts.
save_jsonl = save_trajectories
