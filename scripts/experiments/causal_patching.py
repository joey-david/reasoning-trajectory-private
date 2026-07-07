#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.process_isomers.causal_patching import (
    run_causal_patching,
    validate_causal_patching_setup,
)


def main() -> int:
    """Run or validate H3 causal patching.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Run H3 causal patching.")
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--patch-mode",
        choices=("full", "subspace", "both"),
        help="Override the configured patch modes.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate pairs, activations, projection, and reconstruction without inference.",
    )
    parser.add_argument(
        "--allow-missing-activations",
        action="store_true",
        help="For local config checks before the targeted replay is available.",
    )
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--continuations-per-condition", type=int)
    args = parser.parse_args()
    if args.validate_only:
        print(
            validate_causal_patching_setup(
                args.run_path,
                patch_mode=args.patch_mode,
                require_activations=not args.allow_missing_activations,
            )
        )
    else:
        run_causal_patching(
            args.run_path,
            patch_mode=args.patch_mode,
            max_pairs=args.max_pairs,
            continuations_per_condition=args.continuations_per_condition,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
