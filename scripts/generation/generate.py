#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestration.generation_runner import generate_runs


def main() -> int:
    """Generate outputs for the requested run directories.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(
        description="Generate outputs for one or more run folders."
    )
    parser.add_argument(
        "run_paths", nargs="+", help="Run folder(s), executed sequentially."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Generate at most this many dataset rows per run.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    generate_runs(
        [Path(run_path_arg) for run_path_arg in args.run_paths],
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
