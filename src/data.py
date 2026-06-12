from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import resolve_repo_path


def get_num_instances(dataset_path: str | Path) -> int:
    """Returns the total number of instances in a dataset"""
    path = resolve_repo_path(dataset_path)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_samples(
    dataset_path: str | Path,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Read a JSONL dataset into memory.
    If indices is provided, returns only those rows.
    """
    path = resolve_repo_path(dataset_path)
    wanted = set(indices) if indices is not None else None
    rows: list[dict[str, Any]] = []
    # Open the file with `encoding="utf-8"`.
    with path.open("r", encoding="utf-8") as handle:
        # Loop over lines.
        for i, line in enumerate(handle):
            # Skip blank lines with `if not line.strip(): continue`.
            if not line.strip():
                continue
            if wanted is not None and i not in wanted:
                continue
            rows.append(json.loads(line))
    return rows


def select_samples(
    samples: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply simple selection from config.

    Start with these config keys:
    - `sample_ids`: exact ids to keep.
    - `sample_offset`: number of rows to skip.
    - `sample_limit`: maximum number of rows to keep.

    Python patterns worth noticing:
    - `set(...)` makes membership checks fast.
    - list slicing is enough for offset/limit.
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


def prompt_from_sample(sample: dict[str, Any], config: dict[str, Any]) -> str:
    """Build the final prompt text sent to the model.

    Dataset fields to support first:
    - `prompt`
    - `question`
    - `input`

    Config fields to prepend:
    - `system_prompt`
    - `prompt_prefix`
    """
    body = str(
        sample.get("prompt") or sample.get("question") or sample.get("input") or ""
    )
    parts = [config.get("system_prompt", ""), config.get("prompt_prefix", ""), body]
    return "\n\n".join(part for part in parts if part)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as JSONL.

    Useful functions:
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
