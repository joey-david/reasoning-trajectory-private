#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.process_isomers.component_localization import run_component_localization


CANONICAL_REPLAY = Path("runs/SmolLM3-3B/replay/h2_component_replay")
CANONICAL_H2_DIR = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/"
    "gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates"
)


def main() -> int:
    """Analyze H2 localization across captured components."""
    parser = argparse.ArgumentParser(
        description="Analyze the canonical H2 component replay or explicit artifacts."
    )
    parser.add_argument(
        "replay_run", nargs="?", type=Path, default=CANONICAL_REPLAY
    )
    parser.add_argument("h2_dir", nargs="?", type=Path, default=CANONICAL_H2_DIR)
    args = parser.parse_args()
    print(run_component_localization(args.replay_run, args.h2_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
