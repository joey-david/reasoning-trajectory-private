#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.tools import run_all_tools, run_tool
from src.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one or all analysis tools.")
    parser.add_argument("run_path", help="Path to runs/<model>/<run>")
    parser.add_argument("--tool", default="all", help="Tool name, or all")
    parser.add_argument("--layer", help="Activation layer for projection tools")
    parser.add_argument("--interval", type=int, default=16, help="Token interval for trajectory projection")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--n", type=int, default=24, help="Number of PCA components to report")
    parser.add_argument("--max-points", type=int, default=12000, help="Maximum displayed trajectory points")
    parser.add_argument("--max-vectors", type=int, default=20000, help="Maximum vectors used for PCA component analysis")
    parser.add_argument("--skip-first", type=int, default=2, help="Ignore this many initial generated-token states")
    parser.add_argument("--skip-last", type=int, default=2, help="Ignore this many final generated-token states")
    args = parser.parse_args()

    config = load_run_config(args.run_path)
    params = {
        "trajectory_projection": {
            "layer": args.layer,
            "interval": args.interval,
            "method": args.method,
            "max_points": args.max_points,
            "skip_first": args.skip_first,
            "skip_last": args.skip_last,
        },
        "pca_components": {
            "layer": args.layer,
            "n": args.n,
            "max_vectors": args.max_vectors,
            "skip_first": args.skip_first,
            "skip_last": args.skip_last,
        },
    }
    if args.tool == "all":
        outputs = run_all_tools(config, params)
    else:
        outputs = [run_tool(config, args.tool, params.get(args.tool, {}))]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
