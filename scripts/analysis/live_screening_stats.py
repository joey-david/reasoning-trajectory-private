#!/usr/bin/env python3
"""Write live screening success stats from a growing generations.jsonl."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


def read_max_new_tokens(config_path: Path) -> int:
    text = config_path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*max_new_tokens:\s*(\d+)\s*$", text)
    return int(match.group(1)) if match else 0


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def per_sample_stats(rows: list[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("sample_id")), []).append(row)

    stats = []
    for sample_id, sample_rows in sorted(grouped.items()):
        scored = [row for row in sample_rows if row.get("is_correct") is not None]
        correct = sum(row.get("is_correct") is True for row in scored)
        incorrect = len(scored) - correct
        capped = sum(
            max_new_tokens > 0
            and len(row.get("generated_token_ids") or []) >= max_new_tokens
            for row in sample_rows
        )
        pass_rate = correct / len(scored) if scored else None
        stats.append(
            {
                "sample_id": sample_id,
                "rollouts": len(sample_rows),
                "scored": len(scored),
                "correct": correct,
                "incorrect": incorrect,
                "unscored": len(sample_rows) - len(scored),
                "pass_rate": pass_rate,
                "capped": capped,
                "mixed": pass_rate is not None and 0.0 < pass_rate < 1.0,
                "frontier": pass_rate is not None and 0.2 <= pass_rate <= 0.8,
            }
        )
    return stats


def decimal(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def write_outputs(run_path: Path, stats: list[dict[str, Any]], max_new_tokens: int) -> None:
    output_dir = run_path / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = sum(row["rollouts"] for row in stats)
    scored = sum(row["scored"] for row in stats)
    correct = sum(row["correct"] for row in stats)
    capped = sum(row["capped"] for row in stats)
    summary = {
        "run_path": run_path.as_posix(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "max_new_tokens": max_new_tokens,
        "instances": len(stats),
        "rollouts": rollouts,
        "scored_rollouts": scored,
        "correct_rollouts": correct,
        "accuracy": correct / scored if scored else None,
        "capped_rollouts": capped,
        "capped_rollout_rate": capped / rollouts if rollouts else None,
        "mixed_instances": sum(row["mixed"] for row in stats),
        "frontier_instances": sum(row["frontier"] for row in stats),
    }
    (output_dir / "live_screening_stats.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = [
        "sample_id",
        "rollouts",
        "scored",
        "correct",
        "incorrect",
        "unscored",
        "pass_rate",
        "capped",
        "mixed",
        "frontier",
    ]
    with (output_dir / "live_screening_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in stats:
            writer.writerow({**row, "pass_rate": decimal(row["pass_rate"])})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", type=Path)
    args = parser.parse_args()

    run_path = args.run_path
    max_new_tokens = read_max_new_tokens(run_path / "config.yaml")
    rows = read_rows(run_path / "generation" / "generations.jsonl")
    write_outputs(run_path, per_sample_stats(rows, max_new_tokens), max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
