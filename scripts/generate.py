#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_samples
from src.datasets.adapters import normalize_dataset
from src.datasets.loaders import load_raw_dataset
from src.models.generation_pipeline import generate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate outputs for one run folder.")
    parser.add_argument("run_path")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    config = load_config(run_path)

    dataset_path = run_path / "dataset.jsonl"
    if dataset_path.exists():
        samples = load_samples(dataset_path)
    else:
        dataset_cfg = config["dataset"]
        samples = normalize_dataset(load_raw_dataset(dataset_cfg), dataset_cfg["adapter"])
        if dataset_cfg.get("shuffle_seed") is not None:
            random.Random(int(dataset_cfg["shuffle_seed"])).shuffle(samples)
        offset = int(dataset_cfg.get("sample_offset", 0))
        limit = dataset_cfg.get("sample_limit")
        samples = samples[offset:] if limit is None else samples[offset : offset + int(limit)]

    generate_run(run_path, config, samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
