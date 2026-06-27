#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.dataset_screening import write_mixed_samples
from src.config import load_config
from src.data import write_jsonl
from src.datasets.loaders import load_run_samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export mixed-success samples below a pass-rate threshold."
    )
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pass-rate", type=float, default=0.8)
    args = parser.parse_args()

    if not 0.0 < args.max_pass_rate <= 1.0:
        parser.error("--max-pass-rate must be in (0, 1]")

    mixed_path = write_mixed_samples(args.source_run)
    with mixed_path.open(newline="", encoding="utf-8") as handle:
        selected_ids = {
            row["sample_id"]
            for row in csv.DictReader(handle)
            if row["mixed"].lower() == "true"
            and float(row["pass_rate"]) < args.max_pass_rate
        }

    config = load_config(args.source_run)
    samples = load_run_samples(args.source_run, config["dataset"])
    selected = [sample for sample in samples if str(sample["id"]) in selected_ids]
    found_ids = {str(sample["id"]) for sample in selected}
    missing = selected_ids - found_ids
    if missing:
        raise ValueError(f"Selected sample IDs not found in source dataset: {missing}")

    write_jsonl(args.out, selected)
    print(
        f"wrote {len(selected)} mixed samples with pass rate "
        f"< {args.max_pass_rate:.2f} to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
