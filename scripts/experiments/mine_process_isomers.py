#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.process_isomers import write_process_isomer_pairs


def main() -> int:
    """Mine symbolically equivalent trace pairs for H3.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Mine H3 symbolic-state pairs.")
    parser.add_argument("h2_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--activation-run", type=Path)
    parser.add_argument("--generation-run", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--per-sample", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=30)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--min-path-edits", type=int, default=2)
    parser.add_argument("--min-path-distance", type=float, default=0.2)
    parser.add_argument("--max-pairs-per-question", type=int, default=2)
    parser.add_argument("--max-trajectory-reuse", type=int, default=2)
    parser.add_argument("--max-target-remaining-tokens", type=int)
    parser.add_argument("--require-target-correct", action="store_true")
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
