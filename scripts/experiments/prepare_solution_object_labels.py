#!/usr/bin/env python3
"""Prepare token-window tasks for remote solution-object labeling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.solution_object_labeling import build_label_windows


CANONICAL_SOURCE = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)
CANONICAL_OUTPUT = Path(
    "runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver/token_windows.jsonl"
)
CANONICAL_UPDATES = (
    CANONICAL_SOURCE / "analysis/experiments/h2_localized_updates/updates.jsonl"
)


def main() -> int:
    """Build token windows from an existing activation run."""
    parser = argparse.ArgumentParser(
        description="Prepare canonical semantic-label windows or explicit artifacts."
    )
    parser.add_argument(
        "source_run", nargs="?", type=Path, default=CANONICAL_SOURCE
    )
    parser.add_argument("output", nargs="?", type=Path, default=CANONICAL_OUTPUT)
    parser.add_argument("--updates", type=Path, default=CANONICAL_UPDATES)
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
