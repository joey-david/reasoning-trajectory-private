from __future__ import annotations

from pathlib import Path

import csv
import json

from src.config import run_path


def write_generation_summary(config: dict) -> Path:
    source = run_path(config) / "generation" / "generations.jsonl"
    target = run_path(config) / "analysis" / "generation_summary.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8") as handle, target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["sample_id", "seed", "temperature", "tokens", "chars", "avg_logprob"])
        writer.writeheader()
        for line in handle:
            row = json.loads(line)
            logprobs = row.get("logprobs") or []
            writer.writerow({
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "tokens": len(row.get("token_ids") or []),
                "chars": len(row.get("text") or ""),
                "avg_logprob": sum(logprobs) / len(logprobs) if logprobs else "",
            })
    return target

