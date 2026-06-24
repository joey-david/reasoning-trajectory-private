#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.datasets.loaders import load_normalized_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    cfg = load_config(run_path).to_dict()
    dataset_cfg = cfg["dataset"]

    rows = load_normalized_dataset(dataset_cfg)

    out = Path(args.out) if args.out else run_path / "dataset.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
