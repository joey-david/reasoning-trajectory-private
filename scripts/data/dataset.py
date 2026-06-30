#!/usr/bin/env python3
"""Make tiny JSONL datasets for experiments.

Example target:

    python scripts/data/dataset.py subset data/gpqa/gpqa_diamond.jsonl \
      data/gpqa/gpqa_diamond_repeat.jsonl --limit 3 --offset 2 --repeat 50
    # Takes the 3rd, 4th and 5th instances of the dataset and repeat them 50x each,
    # outputting the result in .../gpqa_diamond_repeat.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runtime.data import load_samples, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Small dataset helper.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subset = subcommands.add_parser("subset", help="Write a small selected JSONL file.")
    subset.add_argument("source")
    subset.add_argument("dest")
    subset.add_argument("--limit", type=int, default=None)
    subset.add_argument("--offset", type=int, default=0)
    subset.add_argument("--repeat", type=int, default=1)

    args = parser.parse_args()

    if args.command == "subset":
        rows = load_samples(args.source)
        stop = None if args.limit is None else args.offset + args.limit
        selected = rows[args.offset : stop]
        repeated = [row for row in selected for _ in range(args.repeat)]

        write_jsonl(Path(args.dest), repeated)
        print(f"wrote {len(repeated)} rows to {args.dest}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
