"""Load configured dataset sources and produce selected normalized run samples."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from src.runtime.data import load_samples, select_samples
from src.datasets.adapters import normalize_dataset


def load_raw_dataset(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load unnormalized rows from JSONL or Hugging Face Datasets.

    Args:
        dataset_cfg: Source type, path, split, revision, and optional dataset name.

    Returns:
        Raw dataset rows represented as dictionaries.
    """
    source = dataset_cfg.get("source", "jsonl")

    if source == "jsonl":
        return load_samples(dataset_cfg["path"])

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
    """Shuffle and slice dataset rows according to dataset configuration.

    Args:
        rows: Rows to select; seeded shuffling mutates this list in place.
        dataset_cfg: Shuffle seed and selection options.

    Returns:
        Selected rows in their resulting order.
    """
    seed = dataset_cfg.get("shuffle_seed")
    if seed is not None:
        random.Random(int(seed)).shuffle(rows)
    return select_samples(rows, dataset_cfg)


def load_normalized_dataset(dataset_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load, filter, normalize, and select a configured dataset.

    Args:
        dataset_cfg: Complete dataset source and normalization configuration.

    Returns:
        Normalized records satisfying filters and selection constraints.
    """
    rows = filter_dataset_rows(load_raw_dataset(dataset_cfg), dataset_cfg)
    rows = normalize_dataset(rows, dataset_cfg["adapter"])
    if dataset_cfg.get("require_gold_answer", False):
        rows = [row for row in rows if row.get("gold_answer") is not None]
    return select_dataset_rows(rows, dataset_cfg)


def load_run_samples(
    run_path: Path, dataset_cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load the materialized dataset for a run or rebuild it from configuration.

    Args:
        run_path: Run folder that may contain ``dataset.jsonl``.
        dataset_cfg: Fallback dataset loading and normalization configuration.

    Returns:
        Run-ready normalized sample records.
    """
    dataset_path = run_path / "dataset.jsonl"
    if dataset_path.exists():
        return load_samples(dataset_path)
    return load_normalized_dataset(dataset_cfg)


def filter_dataset_rows(
    rows: list[dict[str, Any]],
    dataset_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Retain raw rows matching every configured field filter.

    Args:
        rows: Raw dataset rows.
        dataset_cfg: Dataset configuration with optional accepted values by field.

    Returns:
        Rows that satisfy all configured filters.
    """
    for key, expected in dataset_cfg.get("filters", {}).items():
        accepted = expected if isinstance(expected, list) else [expected]
        rows = [row for row in rows if row.get(key) in accepted]
    return rows
