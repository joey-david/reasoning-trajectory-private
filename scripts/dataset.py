#!/usr/bin/env python3
# NOTE:the line above is a *shebang*. Tells unix what to run the file with
# when one attempts to run it as an executable, e.g. with "./"
"""Make tiny JSONL datasets for experiments.

Example target:

    python scripts/dataset.py subset data/gpqa/gpqa_diamond.jsonl \
      data/gpqa/gpqa_diamond_repeat.jsonl --limit 3 --offset 2 --repeat 50
    # Takes the 3rd, 4th and 5th instances of the dataset and repeat them 50x each,
    # outputting the result in .../gpqa_diamond_repeat.jsonl
The first argument determines the kind of sampling to do.

"""

# annotations are imported from future.
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# insert the repo's root path before the file name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import get_num_instances, load_samples, write_jsonl


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
        # Use list slicing first; TODO: add randomization
        num_inst = get_num_instances(args.source)
        start = args.offset
        # be protective about oob
        stop = num_inst if args.limit is None else min(num_inst, start + args.limit)
        indices = list(range(start, stop))
        selected = load_samples(args.source, indices)

        repeated = []
        for row in selected:
            repeated.extend([row] * args.repeat)

        write_jsonl(Path(args.dest), repeated)
        print(f"wrote {len(repeated)} rows to {args.dest}")
        return 0

    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
