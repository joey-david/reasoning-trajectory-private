from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # Create output folders lazily, right before the first file write.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            # JSONL is easy to append and stream: one complete JSON row per line.
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
