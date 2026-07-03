#!/usr/bin/env python3
"""Evaluate Qwen-labeled semantic units against token-level objectives."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.token_segmentation.semantic_evaluation import (
    run_semantic_token_segmentation,
)


def main() -> int:
    """Parse artifact paths and run semantic token evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--labels-run", type=Path, required=True)
    parser.add_argument("--gold-run", type=Path, required=True)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--min-segment-tokens", type=int, default=4)
    args = parser.parse_args()
    report = run_semantic_token_segmentation(
        args.run,
        labels_run=args.labels_run,
        gold_run=args.gold_run,
        updates_path=args.updates,
        layer=args.layer,
        min_segment_tokens=args.min_segment_tokens,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
