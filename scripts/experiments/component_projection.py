#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.component_projection import run_component_projection


def main() -> int:
    """Train an H4 projection in a selected component space.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(
        description="Train an H4 projection in an H3 component space."
    )
    parser.add_argument("replay_run", type=Path)
    parser.add_argument("h2_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--component",
        choices=("attention_output", "mlp_output"),
        required=True,
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-updates", type=int, default=12000)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--projection-dim", type=int, default=128)
    args = parser.parse_args()
    print(
        run_component_projection(
            args.replay_run,
            args.h2_dir,
            args.out_dir,
            component=args.component,
            layer=args.layer,
            max_updates=args.max_updates,
            max_pairs=args.max_pairs,
            epochs=args.epochs,
            projection_dim=args.projection_dim,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
