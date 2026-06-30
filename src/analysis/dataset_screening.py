"""Summarize generation runs for dataset-difficulty screening and maintain screening CSV artifacts."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.analysis.common import read_generation_rows
from src.runtime.config import load_config


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


def _sample_screening_stats(
    rows: list[dict[str, Any]], max_new_tokens: int
) -> list[dict[str, Any]]:
    """Aggregate scoring and length-cap outcomes for each sample.

    Args:
        rows: Generation rows to group by sample ID.
        max_new_tokens: Configured generation cap, or zero when disabled.

    Returns:
        Per-sample rollout counts, pass rates, cap counts, and difficulty flags.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["sample_id"]), []).append(row)

    stats = []
    for sample_id, sample_rows in grouped.items():
        scored = [row for row in sample_rows if row.get("is_correct") is not None]
        correct = sum(row.get("is_correct") is True for row in scored)
        pass_rate = correct / len(scored) if scored else None
        stats.append(
            {
                "sample_id": sample_id,
                "rollouts": len(sample_rows),
                "correct": correct,
                "incorrect": len(scored) - correct,
                "unscored": len(sample_rows) - len(scored),
                "pass_rate": pass_rate,
                "capped": sum(
                    max_new_tokens > 0
                    and len(row.get("generated_token_ids", [])) >= max_new_tokens
                    for row in sample_rows
                ),
                "mixed": pass_rate is not None and 0.0 < pass_rate < 1.0,
                "frontier": pass_rate is not None and 0.2 <= pass_rate <= 0.8,
            }
        )
    return stats


def summarize_run(run_path: Path) -> dict[str, Any]:
    """Aggregate one run's completion, scoring, capping, and frontier statistics.

    Args:
        run_path: Generated run folder with configuration and rollout rows.

    Returns:
        A screening-table row containing model, dataset, rollout, and
        classification fields.
    """
    config = load_config(run_path)
    rows = read_generation_rows(run_path)
    dataset_cfg = config["dataset"]
    generation_cfg = config["generation"]
    model_cfg = config["model"]
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 0))
    item_stats = _sample_screening_stats(rows, max_new_tokens)
    scored_rollouts = sum(item["correct"] + item["incorrect"] for item in item_stats)
    correct_rollouts = sum(item["correct"] for item in item_stats)
    accuracy = correct_rollouts / scored_rollouts if scored_rollouts else None
    rates = [
        item["pass_rate"] for item in item_stats if item["pass_rate"] is not None
    ]
    mixed = sum(item["mixed"] for item in item_stats)
    frontier = sum(item["frontier"] for item in item_stats)
    counts = [item["rollouts"] for item in item_stats]
    capped = sum(item["capped"] for item in item_stats)
    expected_instances = int(dataset_cfg.get("sample_limit") or len(item_stats))
    samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    expected_rollouts = expected_instances * samples_per_item
    status = "completed" if len(rows) >= expected_rollouts else "partial"

    return {
        "model": model_cfg.get("source_name", model_cfg["name"]),
        "backend": model_cfg.get("backend", "hf"),
        "precision": model_cfg.get("quantization", model_cfg.get("dtype", "")),
        "model_revision": model_cfg.get("revision", ""),
        "dataset": dataset_cfg["path"],
        "run_path": run_path.as_posix(),
        "status": status,
        "instances": len(item_stats),
        "rollouts": len(rows),
        "expected_rollouts": expected_rollouts,
        "min_rollouts_per_instance": min(counts, default=0),
        "max_rollouts_per_instance": max(counts, default=0),
        "scored_rollouts": scored_rollouts,
        "scored_rollout_rate": decimal(
            scored_rollouts / len(rows) if rows else None
        ),
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
            scored_rollouts / len(rows) if rows else 0.0,
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
    """Write per-sample rollout outcomes for mixed and frontier cases.

    Args:
        run_path: Generated run folder to summarize.

    Returns:
        Path to the written ``analysis/mixed_samples.csv`` file.
    """
    config = load_config(run_path)
    rows = read_generation_rows(run_path)
    max_new_tokens = int(config["generation"].get("max_new_tokens", 0))
    item_stats = _sample_screening_stats(rows, max_new_tokens)

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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in sorted(item_stats, key=lambda row: row["sample_id"]):
            writer.writerow(
                {
                    **item,
                    "pass_rate": decimal(item["pass_rate"]),
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
    """Assign a coarse screening outcome from run-level and per-item results.

    Args:
        accuracy: Accuracy across scored rollouts, or ``None`` when unscored.
        item_rates: Per-sample pass rates for samples with scored rollouts.
        scored_rate: Fraction of all rollouts with correctness labels.
        capped_rate: Fraction reaching the configured generation token cap.
        complete: Whether the expected rollout count was reached.

    Returns:
        One of the screening classification labels.
    """
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
    """Upsert run summaries into the persistent screening table.

    Args:
        csv_path: Screening CSV to create or update.
        summaries: Fresh summary dictionaries keyed logically by model and run path.

    Returns:
        None; rewrites the CSV while preserving existing notes.
    """
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
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            sorted(existing.values(), key=lambda row: (row["model"], row["dataset"]))
        )


def decimal(value: float | None) -> str:
    """Format an optional ratio for stable CSV output.

    Args:
        value: Ratio to format, or ``None`` for a blank cell.

    Returns:
        Four-decimal text or an empty string.
    """
    return "" if value is None else f"{value:.4f}"
