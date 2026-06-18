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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate outputs for one or more run folders.")
    parser.add_argument("run_paths", nargs="+", help="Run folder(s), executed sequentially.")
    args = parser.parse_args()

    for i, run_path_arg in enumerate(args.run_paths, start=1):
        run_path = Path(run_path_arg)
        print(f"[{i}/{len(args.run_paths)}] generating {run_path}", flush=True)
        generate_one_run(run_path)
    return 0


def generate_one_run(run_path: Path) -> None:
    from src.models.generation_pipeline import generate_run

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


if __name__ == "__main__":
    raise SystemExit(main())
