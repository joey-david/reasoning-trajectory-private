"""Read the stable run artifacts consumed by trajectory analyses."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np


T = TypeVar("T")


def read_generation_rows(run_path: Path) -> list[dict[str, Any]]:
    """Read all rollout rows from a completed run.

    Args:
        run_path: Run folder containing ``generation/generations.jsonl``.

    Returns:
        Parsed non-empty generation rows in file order.
    """
    return read_jsonl(run_path / "generation" / "generations.jsonl")


def read_sample_records(run_path: Path) -> dict[str, dict[str, Any]]:
    """Load persisted per-sample records keyed by their sanitized filename stem.

    Args:
        run_path: Run folder containing ``generation/samples``.

    Returns:
        Mapping from sample artifact stem to parsed sample record.
    """
    sample_dir = run_path / "generation" / "samples"
    return {
        path.stem: json.loads(path.read_text()) for path in sample_dir.glob("*.json")
    }


def evenly_capped(items: list[T], max_items: int) -> list[T]:
    """Downsample an ordered list at evenly spaced positions.

    Args:
        items: Ordered items to retain or sample.
        max_items: Maximum retained count; non-positive values disable capping.

    Returns:
        The original list when under the cap, otherwise evenly spaced items.
    """
    if max_items <= 0 or len(items) <= max_items:
        return items
    keep = np.linspace(0, len(items) - 1, max_items, dtype=int)
    return [items[int(i)] for i in keep]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace a JSONL file with the supplied records."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_hidden_states_npz(path: str | Path) -> tuple[np.ndarray, list[int]]:
    """Load float or symmetrically quantized hidden states from one artifact."""
    with np.load(path) as data:
        layers = data["layer_indices"].astype(int).tolist()
        if "hidden_states" in data:
            return data["hidden_states"].copy(), layers
        if "hidden_states_q" in data and "hidden_states_scale" in data:
            values = data["hidden_states_q"].astype(np.float32)
            scale = data["hidden_states_scale"].astype(np.float32)
            return values * scale[..., None], layers
    raise KeyError(f"No hidden states found in {path}")
