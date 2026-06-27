from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from src.data import load_samples, select_samples
from src.datasets.adapters import normalize_dataset
from src.paths import resolve_repo_path


def load_raw_dataset(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = dataset_cfg.get("source", "jsonl")

    if source == "jsonl":
        path = resolve_repo_path(dataset_cfg["path"])
        with Path(path).open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    if source == "hf":
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing Hugging Face dependency `datasets`. Install/update the "
                "venv with: uv pip install --python .venv/bin/python -r requirements.txt"
            ) from exc
        ds = load_dataset(
            dataset_cfg["path"],
            dataset_cfg.get("name"),
            split=dataset_cfg.get("split", "train"),
            revision=dataset_cfg.get("revision"),
        )
        return [dict(row) for row in ds]

    raise ValueError(f"Unsupported dataset source: {source!r}")


def select_dataset_rows(
    rows: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    seed = dataset_cfg.get("shuffle_seed")
    if seed is not None:
        random.Random(int(seed)).shuffle(rows)
    return select_samples(rows, dataset_cfg)


def load_normalized_dataset(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = filter_dataset_rows(load_raw_dataset(dataset_cfg), dataset_cfg)
    rows = normalize_dataset(rows, dataset_cfg["adapter"])
    if dataset_cfg.get("require_gold_answer", False):
        rows = [row for row in rows if row.get("gold_answer") is not None]
    return select_dataset_rows(rows, dataset_cfg)


def load_run_samples(
    run_path: Path, dataset_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    dataset_path = run_path / "dataset.jsonl"
    if dataset_path.exists():
        return load_samples(dataset_path)
    return load_normalized_dataset(dataset_cfg)


def filter_dataset_rows(
    rows: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    for key, expected in dataset_cfg.get("filters", {}).items():
        accepted = expected if isinstance(expected, list) else [expected]
        rows = [row for row in rows if row.get(key) in accepted]
    return rows
