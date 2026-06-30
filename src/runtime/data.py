"""Provide shared JSONL dataset loading, selection, and writing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.runtime.paths import resolve_repo_path


def load_samples(
    dataset_path: str | Path,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Read a JSONL dataset, optionally retaining specific physical row indices.

    Args:
        dataset_path: Absolute path or path relative to the repository root.
        indices: Zero-based line indices to retain, or ``None`` for every row.

    Returns:
        Parsed JSON objects in source order.
    """
    path = resolve_repo_path(dataset_path)
    wanted = set(indices) if indices is not None else None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            if wanted is not None and i not in wanted:
                continue
            rows.append(json.loads(line))
    return rows


def select_samples(
    samples: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply configured ID filtering, offset, and limit to normalized samples.

    Args:
        samples: Normalized sample records in their current order.
        config: Dataset options containing optional IDs, offset, and limit.

    Returns:
        The selected sample records.
    """
    if "sample_ids" in config:
        wanted = {str(item) for item in config["sample_ids"]}
        samples = [
            row
            for row in samples
            if str(row.get("id") or row.get("problem_id")) in wanted
        ]

    offset = int(config.get("sample_offset", 0))
    limit = config.get("sample_limit")
    if limit is None:
        return samples[offset:]
    return samples[offset : offset + int(limit)]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace a JSONL file with the supplied records.

    Args:
        path: Destination file path.
        rows: Dictionary records to serialize, one per line.

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
