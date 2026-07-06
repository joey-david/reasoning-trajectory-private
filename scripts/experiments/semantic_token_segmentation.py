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


CANONICAL_RUN = Path(
    "runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)
CANONICAL_LABELS_RUN = Path(
    "runs/Qwen3.5-122B-A10B-FP8/solution_object_silver"
)
CANONICAL_GOLD_RUN = Path("runs/SmolLM3-3B/thought_units_gold_answers")
CANONICAL_UPDATES = (
    CANONICAL_RUN / "analysis/experiments/h2_localized_updates/updates.jsonl"
)
CANONICAL_GAPS = (1, 4, 8)


def main() -> int:
    """Evaluate judged semantic intervals against token-level objectives."""
    parser = argparse.ArgumentParser(
        description=(
            "Run semantic token segmentation. With no arguments, reproduce the "
            "canonical 1/4/8-token granularity sweep."
        )
    )
    parser.add_argument("run", nargs="?", type=Path, default=CANONICAL_RUN)
    parser.add_argument("--labels-run", type=Path, default=CANONICAL_LABELS_RUN)
    parser.add_argument("--gold-run", type=Path, default=CANONICAL_GOLD_RUN)
    parser.add_argument("--updates", type=Path, default=CANONICAL_UPDATES)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument(
        "--min-segment-tokens",
        type=int,
        help="Run one granularity instead of the canonical 1/4/8-token sweep.",
    )
    args = parser.parse_args()
    gaps = (
        (args.min_segment_tokens,)
        if args.min_segment_tokens is not None
        else CANONICAL_GAPS
    )
    for gap in gaps:
        print(f"minimum segment: {gap} tokens", flush=True)
        print(
            run_semantic_token_segmentation(
                args.run,
                labels_run=args.labels_run,
                gold_run=args.gold_run,
                updates_path=args.updates,
                layer=args.layer,
                min_segment_tokens=gap,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
