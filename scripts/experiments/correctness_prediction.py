#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.correctness_prediction import run_correctness_prediction


def main() -> int:
    parser = argparse.ArgumentParser(description="Run H5 grouped correctness probes.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--per-sample", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = run_correctness_prediction(
        args.run_path,
        per_sample=args.per_sample,
        folds=args.folds,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
