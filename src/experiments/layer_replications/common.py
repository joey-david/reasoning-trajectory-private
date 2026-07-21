"""Small shared I/O helpers for layer-paper replications."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact, returning an empty list when it is absent."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Replace a JSONL artifact with deterministic compact records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def replication_dir(run_path: Path) -> Path:
    """Return the compact artifact root shared by the three replications."""
    return run_path / "layer_replications"
