from __future__ import annotations

import json
from pathlib import Path

GenerationKey = tuple[str, int, float]


def load_generation_index(run_path: Path) -> set[GenerationKey]:
    path = run_path / "generation" / "generations.jsonl"
    if not path.exists():
        return set()
    keys = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                keys.add((str(row["sample_id"]), int(row["seed"]), float(row["temperature"])))
    return keys
