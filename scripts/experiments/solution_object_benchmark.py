#!/usr/bin/env python3
"""Build or silver-label the compact solution-object benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.solution_object_benchmark import (
    build_solution_object_benchmark,
)
from src.experiments.solution_object_labeling import (
    label_benchmark_with_deepseek,
)


def main() -> int:
    """Run benchmark construction or resumable DeepSeek silver labeling.

    Args:
        None.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--updates", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--questions", type=int, default=50)
    parser.add_argument("--audit-size", type=int, default=120)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--label-silver", action="store_true")
    parser.add_argument("--label-limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out_dir = args.out or (
        args.run_path / "analysis" / "experiments" / "solution_object_benchmark"
    )
    report = build_solution_object_benchmark(
        args.run_path,
        args.updates,
        out_dir,
        question_limit=args.questions,
        audit_size=args.audit_size,
        partitions_path=args.partitions,
    )
    print(report)
    if args.label_silver:
        print(
            label_benchmark_with_deepseek(
                out_dir / "bronze_sentences.jsonl",
                out_dir / "silver_labels.jsonl",
                limit=args.label_limit,
                workers=args.workers,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
