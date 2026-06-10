#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.components import write_pca_components
from src.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Write PCA component amplitudes for a run.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    parser.add_argument("--layer")
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()
    print(write_pca_components(load_run_config(args.run_path), args.layer, args.n))


if __name__ == "__main__":
    main()
