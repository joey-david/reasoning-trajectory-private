#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.process_isomers import write_process_isomer_pairs


CANONICAL_RUN = Path(
    "runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)
CANONICAL_H2_DIR = CANONICAL_RUN / "analysis/experiments/h2_localized_updates"
CANONICAL_OUTPUT = Path("experiments/h3_process_isomer_pairs.jsonl")
CANONICAL_AUDIT = Path("experiments/h3_process_isomer_pair_audit.json")


def main() -> int:
    """Mine symbolically equivalent trace pairs for H3."""
    parser = argparse.ArgumentParser(
        description=(
            "Mine the canonical H3 process-isomer pairs, or use explicit paths "
            "and selection parameters."
        )
    )
    parser.add_argument("h2_dir", nargs="?", type=Path, default=CANONICAL_H2_DIR)
    parser.add_argument("output", nargs="?", type=Path, default=CANONICAL_OUTPUT)
    parser.add_argument("--activation-run", type=Path)
    parser.add_argument("--generation-run", type=Path, default=CANONICAL_RUN)
    parser.add_argument("--audit-path", type=Path, default=CANONICAL_AUDIT)
    parser.add_argument("--per-sample", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-path-edits", type=int, default=2)
    parser.add_argument("--min-path-distance", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-question", type=int, default=3)
    parser.add_argument("--max-trajectory-reuse", type=int, default=3)
    parser.add_argument("--max-target-remaining-tokens", type=int, default=896)
    correctness = parser.add_mutually_exclusive_group()
    correctness.add_argument(
        "--require-target-correct",
        dest="require_target_correct",
        action="store_true",
        default=True,
        help="Require correct target traces (the canonical setting).",
    )
    correctness.add_argument(
        "--allow-incorrect-target",
        dest="require_target_correct",
        action="store_false",
        help="Disable the canonical requirement that target traces be correct.",
    )
    args = parser.parse_args()
    print(
        write_process_isomer_pairs(
            args.h2_dir,
            args.output,
            activation_run=args.activation_run,
            generation_run=args.generation_run,
            audit_path=args.audit_path,
            per_sample=args.per_sample,
            max_pairs=args.max_pairs,
            min_pairs=args.min_pairs,
            min_path_edits=args.min_path_edits,
            min_normalized_path_distance=args.min_path_distance,
            max_pairs_per_question=args.max_pairs_per_question,
            max_trajectory_reuse=args.max_trajectory_reuse,
            max_target_remaining_tokens=args.max_target_remaining_tokens,
            require_target_correct=args.require_target_correct,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
