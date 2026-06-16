from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.config import resolve_repo_path


def load_raw_dataset(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = dataset_cfg.get("source", "jsonl")

    if source == "jsonl":
        path = resolve_repo_path(dataset_cfg["path"])
        with Path(path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if source == "hf":
        ds = load_dataset(
            dataset_cfg["path"],
            dataset_cfg.get("name"),
            split=dataset_cfg.get("split", "train"),
        )
        return [dict(row) for row in ds]

    raise ValueError(f"Unsupported dataset source: {source!r}")
