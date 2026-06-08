from __future__ import annotations

from pathlib import Path

import csv
import json

from src.config import run_path


def write_activation_norms(config: dict) -> Path:
    import numpy as np

    base = run_path(config)
    source = base / "generation" / "generations.jsonl"
    target = base / "analysis" / "activation_norms.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as handle, target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["sample_id", "seed", "temperature", "layer", "l2_norm", "mean", "std"])
        writer.writeheader()
        for line in handle:
            row = json.loads(line)
            if not row.get("activation_file"):
                continue
            arrays = np.load(base / row["activation_file"])
            for layer in arrays.files:
                values = arrays[layer]
                token_norms = np.linalg.norm(values, axis=-1)
                writer.writerow({
                    "sample_id": row["sample_id"],
                    "seed": row["seed"],
                    "temperature": row["temperature"],
                    "layer": layer,
                    "l2_norm": float(token_norms.mean()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                })
    return target
