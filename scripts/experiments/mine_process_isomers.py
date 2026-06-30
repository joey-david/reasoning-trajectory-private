#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.process_isomers import write_process_isomer_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine H3 symbolic-state pairs.")
    parser.add_argument("h2_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-sample", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=30)
    args = parser.parse_args()
    print(
        write_process_isomer_pairs(
            args.h2_dir,
            args.output,
            per_sample=args.per_sample,
            max_pairs=args.max_pairs,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
