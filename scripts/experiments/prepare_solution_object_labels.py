#!/usr/bin/env python3
"""Prepare token-window tasks for remote solution-object labeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.solution_object_labeling import build_label_windows


def main() -> int:
    """Build token windows from an existing activation run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--window-tokens", type=int, default=256)
    parser.add_argument("--overlap-tokens", type=int, default=48)
    args = parser.parse_args()
    count = build_label_windows(
        args.source_run,
        args.updates,
        args.output,
        window_tokens=args.window_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    print(f"{args.output}: {count} token windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
