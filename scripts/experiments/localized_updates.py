#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.localized_updates import run_localized_update_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run H2 spike localization.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--per-sample", type=int, default=10)
    parser.add_argument("--spike-z", type=float, default=3.0)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    report = run_localized_update_analysis(
        args.run_path,
        per_sample=args.per_sample,
        spike_z=args.spike_z,
        window=args.window,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
