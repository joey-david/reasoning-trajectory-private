from __future__ import annotations

from pathlib import Path

from src.data import load_samples

GenerationKey = tuple[str, int, float]


def load_generation_index(run_path: Path) -> set[GenerationKey]:
    path = run_path / "generation" / "generations.jsonl"
    if not path.exists():
        return set()
    return {
        (str(row["sample_id"]), int(row["seed"]), float(row["temperature"]))
        for row in load_samples(path)
    }
