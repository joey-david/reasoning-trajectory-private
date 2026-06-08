#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.generation_summary import write_generation_summary
from src.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a CSV summary of generated text.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    args = parser.parse_args()
    print(write_generation_summary(load_run_config(args.run_path)))


if __name__ == "__main__":
    main()

