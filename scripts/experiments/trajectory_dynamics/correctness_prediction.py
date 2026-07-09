#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.experiments.trajectory_dynamics.correctness_prediction import run_correctness_prediction


CANONICAL_RUN = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)


def main() -> int:
    """Run grouped H5 correctness-prediction probes."""
    parser = argparse.ArgumentParser(
        description="Run H5 on the canonical activation corpus or an explicit run."
    )
    parser.add_argument("run_path", nargs="?", type=Path, default=CANONICAL_RUN)
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
