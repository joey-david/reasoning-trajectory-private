#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.orchestration.generation_runner import generate_runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate outputs for one or more run folders."
    )
    parser.add_argument(
        "run_paths", nargs="+", help="Run folder(s), executed sequentially."
    )
    args = parser.parse_args()

    generate_runs([Path(run_path_arg) for run_path_arg in args.run_paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
