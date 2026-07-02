#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.patching_analysis import (
    analyze_causal_patching,
    validate_h3_smoke,
)


def main() -> int:
    """Analyze H3 patching outputs or apply its smoke gate.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Analyze H3 causal patching.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--smoke-gate", action="store_true")
    parser.add_argument("--smoke-pairs", type=int, default=2)
    parser.add_argument("--smoke-continuations", type=int, default=1)
    args = parser.parse_args()
    if args.smoke_gate:
        print(
            validate_h3_smoke(
                args.run_path,
                pair_count=args.smoke_pairs,
                continuation_count=args.smoke_continuations,
            )
        )
    else:
        print(analyze_causal_patching(args.run_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
