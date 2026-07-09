#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.experiments.process_isomers.component_projection import run_component_projection


CANONICAL_REPLAY = Path("runs/SmolLM3-3B/replay/h2_component_replay")
CANONICAL_H2_DIR = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/"
    "gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates"
)
CANONICAL_OUTPUT = Path("experiments/h3_projections")


def main() -> int:
    """Train H4 projections in the component spaces used by H3."""
    parser = argparse.ArgumentParser(
        description=(
            "Train both canonical H3 component projections, or select one "
            "component and explicit artifact paths."
        )
    )
    parser.add_argument(
        "replay_run", nargs="?", type=Path, default=CANONICAL_REPLAY
    )
    parser.add_argument(
        "h2_dir", nargs="?", type=Path, default=CANONICAL_H2_DIR
    )
    parser.add_argument(
        "out_dir", nargs="?", type=Path, default=CANONICAL_OUTPUT
    )
    parser.add_argument(
        "--component",
        choices=("attention_output", "mlp_output", "both"),
        default="both",
    )
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--max-updates", type=int, default=12000)
    parser.add_argument("--max-pairs", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--projection-dim", type=int, default=128)
    args = parser.parse_args()
    components = (
        ("attention_output", "mlp_output")
        if args.component == "both"
        else (args.component,)
    )
    for component in components:
        print(f"component: {component}", flush=True)
        print(
            run_component_projection(
                args.replay_run,
                args.h2_dir,
                args.out_dir,
                component=component,
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
