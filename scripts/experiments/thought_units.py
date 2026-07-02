#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.thought_units import run_prompt_transfer, run_thought_units


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run sentence-lattice objective-relative segmentation."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--projection", type=Path)
    parser.add_argument(
        "--gold-answer-run",
        type=Path,
        help="Use teacher-forced gold-solution states for the answer objective.",
    )
    parser.add_argument("--per-sample", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--gram-dim", type=int, default=16)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--max-traces", type=int)
    parser.add_argument(
        "--transfer-run",
        action="append",
        type=Path,
        default=[],
        help="After analysis, test H4 detectors on this prompt run.",
    )
    args = parser.parse_args()
    report = run_thought_units(
        args.run_path,
        projection_path=args.projection,
        gold_answer_run=args.gold_answer_run,
        per_sample=args.per_sample,
        pca_dim=args.pca_dim,
        gram_dim=args.gram_dim,
        rebuild_features=args.rebuild_features,
        max_traces=args.max_traces,
    )
    print(report)
    if args.transfer_run:
        print(run_prompt_transfer(args.run_path, args.transfer_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
