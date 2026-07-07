#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.boundaries.comparison import run_boundary_comparison


CANONICAL_RUNS = (
    Path("runs/SmolLM3-3B/pilots/h1_freeform_replay"),
    Path("runs/SmolLM3-3B/pilots/h1_numbered_steps_pilot"),
    Path("runs/SmolLM3-3B/pilots/h1_sentence_separated_pilot"),
    Path("runs/SmolLM3-3B/pilots/h1_paragraph_separated_pilot"),
)


def main() -> int:
    """Run the H1 text-versus-latent boundary comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Run H1 over the canonical four prompting conditions, or over "
            "explicit run folders."
        )
    )
    parser.add_argument(
        "run_paths",
        nargs="*",
        type=Path,
        help="Generation runs; defaults to the canonical H1 conditions.",
    )
    parser.add_argument("--per-sample", type=int, default=5)
    parser.add_argument("--window", type=int, default=2)
    args = parser.parse_args()
    report = run_boundary_comparison(
        args.run_paths or CANONICAL_RUNS,
        per_sample=args.per_sample,
        window=args.window,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
