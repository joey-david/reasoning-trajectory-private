"""Command-line entry point for bounded trajectory diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from reasoning_trajectory.pipeline import analyze_trajectories


def main() -> int:
    """Analyze one completed run and print the output artifact path."""
    parser = argparse.ArgumentParser(
        description="Analyze token-level hidden-state trajectories."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max-trajectories", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=200)
    args = parser.parse_args()
    analyze_trajectories(
        args.run_path,
        {
            "layer": args.layer,
            "max_trajectories": args.max_trajectories,
            "max_tokens_per_trajectory": args.max_tokens,
            "max_pairs": args.max_pairs,
        },
    )
    print(args.run_path / "analysis" / "trajectory_metrics.json")
    return 0
