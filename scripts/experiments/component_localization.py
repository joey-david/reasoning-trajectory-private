#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.component_localization import run_component_localization


def main() -> int:
    """Analyze H2 localization across captured components.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(description="Analyze H2 component replay.")
    parser.add_argument("replay_run", type=Path)
    parser.add_argument("h2_dir", type=Path)
    args = parser.parse_args()
    print(run_component_localization(args.replay_run, args.h2_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
