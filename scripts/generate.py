#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Let this script import the local `src` package when run from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_run_config
from src.generation import run_generation


def main() -> None:
    # argparse turns command-line text into a small object we can read below.
    parser = argparse.ArgumentParser(description="Run model generation for a run folder.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    args = parser.parse_args()

    # The run folder owns its config; the core generator owns all model work.
    output = run_generation(load_run_config(args.run_path))
    print(output)


# Only run the CLI when this file is executed directly, not when imported.
if __name__ == "__main__":
    main()
