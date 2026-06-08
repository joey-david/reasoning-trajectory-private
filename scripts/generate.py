#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_run_config
from src.generation import run_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model generation for a run folder.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    args = parser.parse_args()
    output = run_generation(load_run_config(args.run_path))
    print(output)


if __name__ == "__main__":
    main()

