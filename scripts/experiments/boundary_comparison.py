#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.boundary_comparison import run_boundary_comparison


def main() -> int:
    """Run the H1 text-versus-latent boundary comparison.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Run H1 boundary comparison.")
    parser.add_argument("run_paths", nargs="+", type=Path)
    parser.add_argument("--per-sample", type=int, default=5)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    report = run_boundary_comparison(
        args.run_paths,
        per_sample=args.per_sample,
        window=args.window,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
