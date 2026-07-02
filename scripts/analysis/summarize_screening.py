#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.dataset_screening import (
    summarize_run,
    update_screening_csv,
    write_mixed_samples,
)


def main() -> int:
    """Update the screening CSV from completed run summaries.

    Args:
        None.

    Returns:
        The computed index, count, or status code.
    """
    parser = argparse.ArgumentParser(
        description="Update the model/dataset screening CSV from completed runs."
    )
    parser.add_argument("run_paths", nargs="+")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("experiments/dataset_saturation.csv"),
    )
    args = parser.parse_args()

    summaries = [summarize_run(Path(run_path)) for run_path in args.run_paths]
    update_screening_csv(args.csv, summaries)
    for summary in summaries:
        mixed_path = write_mixed_samples(Path(summary["run_path"]))
        print(
            f"{summary['run_path']}: {summary['classification']} "
            f"accuracy={summary['accuracy']} "
            f"frontier_items={summary['frontier_instances']}/{summary['instances']} "
            f"mixed_samples={mixed_path}"
        )
    print(f"updated {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
