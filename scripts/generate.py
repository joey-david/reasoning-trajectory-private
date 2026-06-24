#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    from src.config import load_config
    from src.datasets.loaders import load_run_samples
    from src.models.generation_pipeline import generate_run

    config = load_config(run_path)
    samples = load_run_samples(run_path, config["dataset"])
    generate_run(run_path, config, samples)


if __name__ == "__main__":
    raise SystemExit(main())
