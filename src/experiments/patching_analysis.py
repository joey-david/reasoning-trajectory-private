"""Summarize H3 continuation validity and correctness by patch condition."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.data import load_samples


def analyze_causal_patching(run_path: Path) -> Path:
    rows = load_samples((run_path / "patching" / "continuations.jsonl").resolve())
    by_condition: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: defaultdict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = str(row["condition"])
        by_condition[condition].append(row)
        by_pair[(int(row["pair_id"]), condition)].append(row)

    conditions = {
        condition: {
            "continuations": len(condition_rows),
            "valid_answer_rate": mean_bool(condition_rows, "has_valid_answer"),
            "correct_rate": mean_bool(condition_rows, "is_correct"),
        }
        for condition, condition_rows in sorted(by_condition.items())
    }
    pair_ids = sorted({pair_id for pair_id, _ in by_pair})
    effects = []
    for comparison in ("equivalent", "position_random", "mismatched"):
        differences = []
        for pair_id in pair_ids:
            baseline = by_pair.get((pair_id, "baseline"), [])
            treatment = by_pair.get((pair_id, comparison), [])
            if baseline and treatment:
                differences.append(
                    mean_bool(treatment, "is_correct")
                    - mean_bool(baseline, "is_correct")
                )
        if differences:
            effects.append(
                {
                    "condition": comparison,
                    "baseline": "baseline",
                    "paired_correctness_difference": float(np.mean(differences)),
                    "pair_count": len(differences),
                }
            )
    report = {
        "hypothesis": "H3_causal_process_isomer_verification",
        "conditions": conditions,
        "paired_effects": effects,
    }
    out_dir = run_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def mean_bool(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(bool(row[field])) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else float("nan")
