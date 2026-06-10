from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_run_config(run_path: str | Path) -> dict[str, Any]:
    path = Path(run_path)
    # Accept either the run folder or the config file itself.
    config_path = path / "config.yaml" if path.is_dir() else path
    with config_path.open("r", encoding="utf-8") as handle:
        # YAML maps naturally to Python dictionaries and lists.
        config = yaml.safe_load(handle) or {}

    # Store the run folder so later code can write outputs next to config.yaml.
    config["_run_path"] = str(config_path.parent)
    return config


def run_path(config: dict[str, Any]) -> Path:
    # Convert the stored string back to a Path for clean path joining.
    return Path(config["_run_path"])


def require(config: dict[str, Any], key: str) -> Any:
    # Small helper for values that should fail loudly when absent.
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]
