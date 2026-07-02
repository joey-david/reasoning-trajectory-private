#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.boundary_interventions import (
    analyze_boundary_interventions,
    prepare_boundary_manifest,
)


def main() -> int:
    """Prepare sentence-boundary intervention tasks.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(
        description="Prepare or analyze objective-family boundary interventions."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze completed continuations instead of rebuilding the manifest.",
    )
    args = parser.parse_args()
    operation = (
        analyze_boundary_interventions if args.analyze else prepare_boundary_manifest
    )
    print(operation(args.run_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
