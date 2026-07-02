#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.replay_capture import rebuild_replay_index, replay_capture_run


def main() -> int:
    """Capture replay activations or rebuild their metadata index.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(
        description="Teacher-force existing generations and capture activations."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild JSON metadata from completed NPZ artifacts without inference.",
    )
    args = parser.parse_args()
    if args.rebuild_index:
        print(f"rebuilt {rebuild_replay_index(args.run_path)} rows")
    else:
        replay_capture_run(args.run_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
