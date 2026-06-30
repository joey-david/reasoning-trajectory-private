#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.replay_capture import replay_capture_run


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-force existing generations and capture activations."
    )
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()
    replay_capture_run(args.run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
