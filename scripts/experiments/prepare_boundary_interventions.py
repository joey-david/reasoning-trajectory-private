#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.boundary_interventions import prepare_boundary_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare objective-family sentence-boundary interventions."
    )
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()
    print(prepare_boundary_manifest(args.run_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
