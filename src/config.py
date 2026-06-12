from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(run_path: str | Path) -> dict[str, Any]:
    """Load `config.yaml` from a run folder.

    Implementing notes:
    - Use `Path(run_path)` so strings and Path objects both work.
    - Use `yaml.safe_load(handle)` instead of `yaml.load`.
    - Add `config["_run_path"] = str(run_path)` so later code knows where to
      write `generation/generations.jsonl`.
    """
    run_path = Path(run_path)
    config_path = run_path / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_run_path"] = str(run_path)
    return config


def resolve_repo_path(path_text: str | Path) -> Path:
    """Resolve a path relative to the repository root.

    Useful Python tools:
    - `Path(__file__).resolve().parents[1]` finds this repo root from `src/`.
    - `Path.is_absolute()` tells you whether a path already starts at `/`.
    """
    root = Path(__file__).resolve().parents[1]
    path = Path(path_text)
    return path if path.is_absolute() else root / path
