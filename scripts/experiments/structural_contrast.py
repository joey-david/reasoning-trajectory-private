#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.structural_contrast import run_structural_contrast


def main() -> int:
    """Train and evaluate the H4 structural projection.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Run H4 contrastive projection.")
    parser.add_argument("h2_dir", type=Path)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--max-updates", type=int, default=12000)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    report = run_structural_contrast(
        args.h2_dir,
        layer=args.layer,
        max_updates=args.max_updates,
        max_pairs=args.max_pairs,
        epochs=args.epochs,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
