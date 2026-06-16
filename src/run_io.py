from __future__ import annotations

from pathlib import Path

from src.artifact_store import artifact_stem


def generation_exists(
    run_path: Path,
    sample_id: str,
    seed: int,
    temperature: float,
) -> bool:
    stem = artifact_stem(sample_id, seed, temperature)
    return (run_path / "generation" / "outputs" / f"{stem}.json").exists()
