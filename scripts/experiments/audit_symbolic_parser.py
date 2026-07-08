#!/usr/bin/env python3
"""Audit symbolic-update extraction against hand-labeled trace windows."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.symbolic import extract_symbolic_updates
from src.runtime.data import load_samples, write_jsonl


CANONICAL_RUN = Path(
    "runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k"
)
CANONICAL_LABELS = Path("experiments/symbolic_parser_audit_labels.jsonl")
CANONICAL_OUT = Path("experiments/symbolic_parser_audit_report.json")
CANONICAL_MATCHES = Path("experiments/symbolic_parser_audit_matches.jsonl")


def main() -> int:
    """Run a deterministic audit from checked-in hand labels."""
    parser = argparse.ArgumentParser(
        description="Compare symbolic parser outputs with hand-labeled windows."
    )
    parser.add_argument("--run", type=Path, default=CANONICAL_RUN)
    parser.add_argument("--labels", type=Path, default=CANONICAL_LABELS)
    parser.add_argument("--out", type=Path, default=CANONICAL_OUT)
    parser.add_argument("--matches", type=Path, default=CANONICAL_MATCHES)
    args = parser.parse_args()
    report, matches = run_audit(args.run, args.labels)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_jsonl(args.matches, matches)
    print(args.out)
    return 0


def run_audit(
    run_path: Path,
    labels_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate parser precision and recall on labeled text windows."""
    labels = load_samples(labels_path)
    rows = {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in load_samples(run_path / "generation" / "generations.jsonl")
    }
    window_results: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    totals = Counter()
    by_gold_operator: dict[str, Counter] = {}
    by_predicted_operator: dict[str, Counter] = {}

    for label in labels:
        key = (str(label["sample_id"]), int(label["seed"]))
        row = rows[key]
        text = str(row["produced_text"])
        start = int(label["window_start"])
        end = int(label["window_end"])
        predictions = [
            prediction_record(update)
            for update in extract_symbolic_updates(
                text,
                token_count=len(row.get("generated_token_ids", [])),
            )
            if update.char_start < end and update.char_end > start
        ]
        gold = [gold_record(event) for event in label["gold_events"]]
        matched_prediction_indices: set[int] = set()
        matched_gold_indices: set[int] = set()
        for gold_idx, event in enumerate(gold):
            for pred_idx, prediction in enumerate(predictions):
                if pred_idx in matched_prediction_indices:
                    continue
                if event_matches_prediction(event, prediction):
                    matched_gold_indices.add(gold_idx)
                    matched_prediction_indices.add(pred_idx)
                    break

        true_positives = len(matched_gold_indices)
        false_negatives = len(gold) - true_positives
        false_positives = len(predictions) - len(matched_prediction_indices)
        totals.update(
            {
                "gold": len(gold),
                "predicted": len(predictions),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            }
        )
        for idx, event in enumerate(gold):
            counter = by_gold_operator.setdefault(event["operator"], Counter())
            counter.update({"gold": 1})
            if idx in matched_gold_indices:
                counter.update({"true_positives": 1})
            else:
                counter.update({"false_negatives": 1})
        for idx, prediction in enumerate(predictions):
            counter = by_predicted_operator.setdefault(prediction["operator"], Counter())
            counter.update({"predicted": 1})
            if idx in matched_prediction_indices:
                counter.update({"true_positives": 1})
            else:
                counter.update({"false_positives": 1})

        window_id = label["window_id"]
        false_positive_records = [
            prediction
            for idx, prediction in enumerate(predictions)
            if idx not in matched_prediction_indices
        ]
        false_negative_records = [
            event for idx, event in enumerate(gold) if idx not in matched_gold_indices
        ]
        window_result = {
            "window_id": window_id,
            "sample_id": key[0],
            "seed": key[1],
            "is_correct": bool(row.get("is_correct")),
            "gold": len(gold),
            "predicted": len(predictions),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": safe_div(true_positives, true_positives + false_positives),
            "recall": safe_div(true_positives, true_positives + false_negatives),
            "false_positive_expressions": [
                record["expression"] for record in false_positive_records
            ],
            "false_negative_descriptions": [
                record["description"] for record in false_negative_records
            ],
            "notes": label.get("notes", ""),
        }
        window_results.append(window_result)
        match_rows.append(
            {
                **window_result,
                "predictions": predictions,
                "gold_events": gold,
                "snippet": text[start:end],
            }
        )

    report = {
        "experiment": "symbolic_parser_fidelity_audit",
        "source_run": run_path.as_posix(),
        "labels": labels_path.as_posix(),
        "definition": {
            "gold_event": (
                "hand-labeled textual arithmetic, numeric binding, verification, "
                "or final-answer event in a selected trace window"
            ),
            "match_rule": (
                "parser prediction matches a gold event when value is equal and the "
                "normalized parser expression matches one of the event aliases"
            ),
            "scope": (
                "small stratified audit of arithmetic-heavy windows, not a corpus-wide "
                "estimate"
            ),
        },
        "window_count": len(labels),
        "totals": dict(totals),
        "metrics": {
            "micro_precision": safe_div(
                totals["true_positives"],
                totals["true_positives"] + totals["false_positives"],
            ),
            "micro_recall": safe_div(
                totals["true_positives"],
                totals["true_positives"] + totals["false_negatives"],
            ),
            "micro_f1": f1(
                safe_div(
                    totals["true_positives"],
                    totals["true_positives"] + totals["false_positives"],
                ),
                safe_div(
                    totals["true_positives"],
                    totals["true_positives"] + totals["false_negatives"],
                ),
            ),
        },
        "by_gold_operator": {
            operator: {
                **dict(counts),
                "recall": safe_div(
                    counts["true_positives"],
                    counts["true_positives"] + counts["false_negatives"],
                ),
            }
            for operator, counts in sorted(by_gold_operator.items())
        },
        "by_predicted_operator": {
            operator: {
                **dict(counts),
                "precision": safe_div(
                    counts["true_positives"],
                    counts["true_positives"] + counts["false_positives"],
                ),
            }
            for operator, counts in sorted(by_predicted_operator.items())
        },
        "windows": window_results,
        "interpretation": {
            "adequate_for": (
                "high-confidence explicit arithmetic equations and some final-answer "
                "markers in GSM-Symbolic-style traces"
            ),
            "not_adequate_for": (
                "complete reasoning-state segmentation: prose arithmetic, partial "
                "variable equations, units, and ambiguous bindings need manual or "
                "stronger structured labeling"
            ),
        },
    }
    return report, match_rows


def prediction_record(update: Any) -> dict[str, Any]:
    """Serialize the fields used by the audit matcher."""
    return {
        "operator": str(update.operator),
        "expression": str(update.expression),
        "normalized": normalize(str(update.expression)),
        "value": round(float(update.value), 8),
        "char_start": int(update.char_start),
        "char_end": int(update.char_end),
    }


def gold_record(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize one hand-labeled event."""
    aliases = event.get("aliases") or [event["expression"]]
    return {
        "operator": str(event["operator"]),
        "expression": str(event["expression"]),
        "aliases": [normalize(str(alias)) for alias in aliases],
        "value": round(float(event["value"]), 8),
        "description": str(event.get("description", event["expression"])),
    }


def event_matches_prediction(event: dict[str, Any], prediction: dict[str, Any]) -> bool:
    """Return whether one prediction satisfies one gold event."""
    if not math.isclose(event["value"], prediction["value"], rel_tol=1e-7, abs_tol=1e-7):
        return False
    return prediction["normalized"] in event["aliases"]


def normalize(expression: str) -> str:
    """Normalize arithmetic text for alias matching."""
    text = expression.lower()
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("€", "").replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    text = text.strip(".;:")
    return text


def safe_div(numerator: int, denominator: int) -> float | None:
    """Return a float division or None for empty denominators."""
    return float(numerator / denominator) if denominator else None


def f1(precision: float | None, recall: float | None) -> float | None:
    """Return F1 for optional precision and recall."""
    if precision is None or recall is None or precision + recall == 0:
        return None
    return float(2 * precision * recall / (precision + recall))


if __name__ == "__main__":
    raise SystemExit(main())
