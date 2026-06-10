#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.trajectory import write_trajectory_projection
from src.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Project token activations to a 3D trajectory.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    parser.add_argument("--layer")
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    args = parser.parse_args()
    print(write_trajectory_projection(load_run_config(args.run_path), args.layer, args.interval, args.method))


if __name__ == "__main__":
    main()
