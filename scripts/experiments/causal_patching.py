#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.causal_patching import run_causal_patching


def main() -> int:
    parser = argparse.ArgumentParser(description="Run H3 causal patching.")
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()
    run_causal_patching(args.run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
