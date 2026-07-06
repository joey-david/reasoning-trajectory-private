#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.localized_updates import run_localized_update_analysis


CANONICAL_RUN = Path(
    "runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)


def main() -> int:
    """Analyze H2 interval activation dynamics."""
    parser = argparse.ArgumentParser(
        description="Run H2 on the canonical activation corpus or an explicit run."
    )
    parser.add_argument("run_path", nargs="?", type=Path, default=CANONICAL_RUN)
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
