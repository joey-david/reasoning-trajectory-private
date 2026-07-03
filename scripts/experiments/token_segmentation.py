#!/usr/bin/env python3
"""Evaluate objective-relative boundaries at generated-token resolution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.token_segmentation import run_token_segmentation


def main() -> int:
    """Parse paths and run token-level segmentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--gold-run", type=Path, required=True)
    parser.add_argument("--updates", type=Path)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--min-segment-tokens", type=int, default=4)
    args = parser.parse_args()
    print(
        run_token_segmentation(
            args.run,
            gold_run=args.gold_run,
            updates_path=args.updates,
            layer=args.layer,
            min_segment_tokens=args.min_segment_tokens,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
