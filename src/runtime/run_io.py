"""Read run-level generation state used to resume completed sample combinations."""

from __future__ import annotations

from pathlib import Path

from src.runtime.data import load_samples

GenerationKey = tuple[str, int, float]


def load_generation_index(run_path: Path) -> set[GenerationKey]:
    """Load the generation identities already present in a run.

    Args:
        run_path: Run folder whose generation index should be read.

    Returns:
        Existing ``(sample_id, seed, temperature)`` tuples, or an empty set
        when the index does not yet exist.
    """
    path = run_path / "generation" / "generations.jsonl"
    if not path.exists():
        return set()
    return {
        (str(row["sample_id"]), int(row["seed"]), float(row["temperature"]))
        for row in load_samples(path)
    }
