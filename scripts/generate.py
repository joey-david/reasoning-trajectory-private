#!/usr/bin/env python3
"""Run one experiment folder.

Your goal is to make this boring:

    python scripts/generate.py runs/<model>/<experiment>

Start by implementing one small piece at a time. Do not add remote execution,
analysis, dashboards, or multiprocessing here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.data import load_samples, select_samples
from src.generation import generate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate outputs for one run folder.")
    parser.add_argument("run_path", help="Example: runs/Qwen3-14B/gpqa_small")
    args = parser.parse_args()

    run_path = Path(args.run_path)
    config = load_config(run_path)

    # Implement this flow slowly:
    # 1. load_samples(config["dataset_path"])
    # 2. select_samples(..., sample_ids/sample_limit/sample_offset)
    # 3. generate_run(run_path, config, selected_samples)
    samples = load_samples(config["dataset_path"])
    selected = select_samples(samples, config)
    generate_run(run_path, config, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
