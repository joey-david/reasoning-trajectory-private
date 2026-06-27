from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analysis.common import read_generation_rows
from src.config import load_config


FIELDNAMES = [
    "model",
    "backend",
    "precision",
    "model_revision",
    "dataset",
    "run_path",
    "status",
    "instances",
    "rollouts",
    "expected_rollouts",
    "min_rollouts_per_instance",
    "max_rollouts_per_instance",
    "scored_rollouts",
    "scored_rollout_rate",
    "capped_rollouts",
    "capped_rollout_rate",
    "accuracy",
    "mixed_instances",
    "mixed_instance_rate",
    "frontier_instances",
    "frontier_instance_rate",
    "classification",
    "capture_enabled",
    "temperature",
    "top_p",
    "evaluated_at",
    "notes",
]


def summarize_run(run_path: Path) -> dict[str, Any]:
    config = load_config(run_path)
    rows = read_generation_rows(run_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    scored = [row for row in rows if row.get("is_correct") is not None]
    accuracy = (
        sum(row["is_correct"] is True for row in scored) / len(scored)
        if scored
        else None
    )
    rates = [
        sum(row.get("is_correct") is True for row in item_rows)
        / len([row for row in item_rows if row.get("is_correct") is not None])
        for item_rows in grouped.values()
        if any(row.get("is_correct") is not None for row in item_rows)
    ]
    mixed = sum(0.0 < rate < 1.0 for rate in rates)
    frontier = sum(0.2 <= rate <= 0.8 for rate in rates)
    counts = [len(item_rows) for item_rows in grouped.values()]
    dataset_cfg = config["dataset"]
    generation_cfg = config["generation"]
    model_cfg = config["model"]
    expected_instances = int(dataset_cfg.get("sample_limit") or len(grouped))
    samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    expected_rollouts = expected_instances * samples_per_item
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 0))
    capped = sum(
        max_new_tokens > 0 and len(row.get("generated_token_ids", [])) >= max_new_tokens
        for row in rows
    )
    status = "completed" if len(rows) >= expected_rollouts else "partial"

    return {
        "model": model_cfg.get("source_name", model_cfg["name"]),
        "backend": model_cfg.get("backend", "hf"),
        "precision": model_cfg.get("quantization", model_cfg.get("dtype", "")),
        "model_revision": model_cfg.get("revision", ""),
        "dataset": dataset_cfg["path"],
        "run_path": run_path.as_posix(),
        "status": status,
        "instances": len(grouped),
        "rollouts": len(rows),
        "expected_rollouts": expected_rollouts,
        "min_rollouts_per_instance": min(counts, default=0),
        "max_rollouts_per_instance": max(counts, default=0),
        "scored_rollouts": len(scored),
        "scored_rollout_rate": decimal(len(scored) / len(rows) if rows else None),
        "capped_rollouts": capped,
        "capped_rollout_rate": decimal(capped / len(rows) if rows else None),
        "accuracy": decimal(accuracy),
        "mixed_instances": mixed,
        "mixed_instance_rate": decimal(mixed / len(rates) if rates else None),
        "frontier_instances": frontier,
        "frontier_instance_rate": decimal(frontier / len(rates) if rates else None),
        "classification": classify_screening(
            accuracy,
            rates,
            len(scored) / len(rows) if rows else 0.0,
            capped / len(rows) if rows else 0.0,
            complete=status == "completed",
        ),
        "capture_enabled": bool(config.get("capture", {}).get("enabled", True)),
        "temperature": generation_cfg.get("temperature"),
        "top_p": generation_cfg.get("top_p"),
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "notes": "",
    }


def write_mixed_samples(run_path: Path) -> Path:
    config = load_config(run_path)
    rows = read_generation_rows(run_path)
    max_new_tokens = int(config["generation"].get("max_new_tokens", 0))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    output = run_path / "analysis" / "mixed_samples.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "rollouts",
        "correct",
        "incorrect",
        "unscored",
        "pass_rate",
        "capped",
        "mixed",
        "frontier",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id, sample_rows in sorted(grouped.items()):
            scored = [row for row in sample_rows if row.get("is_correct") is not None]
            correct = sum(row.get("is_correct") is True for row in scored)
            pass_rate = correct / len(scored) if scored else None
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "rollouts": len(sample_rows),
                    "correct": correct,
                    "incorrect": len(scored) - correct,
                    "unscored": len(sample_rows) - len(scored),
                    "pass_rate": decimal(pass_rate),
                    "capped": sum(
                        max_new_tokens > 0
                        and len(row.get("generated_token_ids", [])) >= max_new_tokens
                        for row in sample_rows
                    ),
                    "mixed": pass_rate is not None and 0.0 < pass_rate < 1.0,
                    "frontier": pass_rate is not None and 0.2 <= pass_rate <= 0.8,
                }
            )
    return output


def classify_screening(
    accuracy: float | None,
    item_rates: list[float],
    scored_rate: float = 1.0,
    capped_rate: float = 0.0,
    *,
    complete: bool = True,
) -> str:
    if not complete:
        return "partial"
    if capped_rate >= 0.5:
        return "length_capped"
    if accuracy is None or not item_rates or scored_rate < 0.95:
        return "unscored"
    mixed_rate = sum(0.0 < rate < 1.0 for rate in item_rates) / len(item_rates)
    frontier_count = sum(0.2 <= rate <= 0.8 for rate in item_rates)
    required_frontier = max(3, round(len(item_rates) * 0.1))
    if accuracy >= 0.95 and mixed_rate < 0.1:
        return "saturated"
    if accuracy <= 0.05 and mixed_rate < 0.1:
        return "too_hard"
    if 0.15 <= accuracy <= 0.85 and frontier_count >= required_frontier:
        return "frontier"
    return "middling"


def update_screening_csv(csv_path: Path, summaries: list[dict[str, Any]]) -> None:
    existing: dict[tuple[str, str], dict[str, str]] = {}
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing[(row["model"], row["run_path"])] = row

    for summary in summaries:
        key = (str(summary["model"]), str(summary["run_path"]))
        previous = existing.get(key, {})
        summary["notes"] = previous.get("notes", summary.get("notes", ""))
        existing[key] = {name: str(summary.get(name, "")) for name in FIELDNAMES}

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(
            sorted(existing.values(), key=lambda row: (row["model"], row["dataset"]))
        )


def decimal(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"
