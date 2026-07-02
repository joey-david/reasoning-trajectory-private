"""Analyze the eight-cell H3 full-vector/subspace intervention design."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.config import load_config
from src.runtime.data import load_samples


RATE_FIELDS = (
    "degenerate_output",
    "has_valid_answer",
    "matches_target_answer",
    "matches_donor_answer",
    "matches_neither_answer",
    "hit_token_limit",
)


def analyze_causal_patching(run_path: Path) -> Path:
    """Write cell summaries and paired question-grouped H3 contrasts.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The path of the written or discovered artifact.
    """
    config = load_config(run_path)
    patch_cfg = config["patching"]
    rows = load_samples((run_path / "patching" / "continuations.jsonl").resolve())
    if any("patch_mode" not in row for row in rows):
        raise ValueError("H3 analysis requires two-variant rows with patch_mode")

    by_cell: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_pair_cell: defaultdict[
        tuple[int, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for row in rows:
        mode = str(row["patch_mode"])
        condition = str(row["condition"])
        by_cell[(mode, condition)].append(row)
        by_pair_cell[(int(row["pair_id"]), mode, condition)].append(row)

    configured_modes = [
        str(mode) for mode in patch_cfg.get("patch_modes", ("full", "subspace"))
    ]
    configured_conditions = [str(condition) for condition in patch_cfg["conditions"]]
    configured_pair_limit = int(patch_cfg.get("max_pairs", 0))
    manifest_pair_count = len(load_samples(Path(patch_cfg["pairs"]).resolve()))
    expected_pairs = min(configured_pair_limit, manifest_pair_count)
    continuations_per_cell = int(patch_cfg.get("continuations_per_condition", 5))
    cells = []
    for mode in configured_modes:
        for condition in configured_conditions:
            cell_rows = by_cell[(mode, condition)]
            expected = expected_pairs * continuations_per_cell
            cells.append(
                {
                    "patch_mode": mode,
                    "condition": condition,
                    "continuations": len(cell_rows),
                    "expected_continuations": expected,
                    "completion_fraction": len(cell_rows) / expected
                    if expected
                    else None,
                    "questions": len(
                        {str(row["target_question"]) for row in cell_rows}
                    ),
                    "rates": {
                        field: grouped_rate_summary(cell_rows, field)
                        for field in RATE_FIELDS
                    },
                    "reconstruction": reconstruction_summary(cell_rows),
                }
            )

    comparisons = []
    for mode in configured_modes:
        for reference in ("baseline", "position_random", "mismatched"):
            comparisons.append(
                paired_comparison(
                    by_pair_cell,
                    treatment=(mode, "equivalent"),
                    reference=(mode, reference),
                )
            )
    comparisons.append(
        paired_comparison(
            by_pair_cell,
            treatment=("subspace", "equivalent"),
            reference=("full", "equivalent"),
        )
    )

    expected_total = (
        expected_pairs
        * len(configured_modes)
        * len(configured_conditions)
        * continuations_per_cell
    )
    report = {
        "hypothesis": "H3_causal_process_isomer_full_vs_subspace",
        "run": run_path.as_posix(),
        "component": patch_cfg["component"],
        "layer": patch_cfg["layer"],
        "projection_path": patch_cfg["projection_path"],
        "design": {
            "patch_modes": configured_modes,
            "conditions": configured_conditions,
            "expected_pairs": expected_pairs,
            "continuations_per_cell": continuations_per_cell,
            "expected_total_continuations": expected_total,
            "observed_total_continuations": len(rows),
            "completion_fraction": len(rows) / expected_total
            if expected_total
            else None,
            "collapse_definition": (
                "empty output, >=32 identical-token run, very low token diversity, "
                "or repeated four-gram collapse"
            ),
        },
        "cells": cells,
        "paired_effects": [comparison for comparison in comparisons if comparison],
        "fallback_gate_inputs": fallback_gate_inputs(cells),
    }
    out_dir = run_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def validate_h3_smoke(
    run_path: Path,
    *,
    pair_count: int = 2,
    continuation_count: int = 1,
    residual_tolerance: float = 1e-4,
) -> Path:
    """Gate the full H3 run on a complete, numerically valid eight-cell smoke.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        pair_count: Number of smoke-test pairs required.
        continuation_count: Number of continuations required per smoke-test cell.
        residual_tolerance: Maximum normalized residual reconstruction error.

    Returns:
        The path of the written or discovered artifact.
    """
    config = load_config(run_path)
    patch_cfg = config["patching"]
    modes = [str(mode) for mode in patch_cfg.get("patch_modes", ("full", "subspace"))]
    conditions = [str(condition) for condition in patch_cfg["conditions"]]
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())[:pair_count]
    pair_ids = {int(pair["pair_id"]) for pair in pairs}
    rows = [
        row
        for row in load_samples(
            (run_path / "patching" / "continuations.jsonl").resolve()
        )
        if int(row["pair_id"]) in pair_ids
        and int(row["continuation"]) < continuation_count
    ]
    observed = {
        (
            int(row["pair_id"]),
            str(row["patch_mode"]),
            str(row["condition"]),
            int(row["continuation"]),
        )
        for row in rows
    }
    expected = {
        (pair_id, mode, condition, continuation)
        for pair_id in pair_ids
        for mode in modes
        for condition in conditions
        for continuation in range(continuation_count)
    }
    errors = []
    missing = sorted(expected - observed)
    if missing:
        errors.append(f"missing {len(missing)} smoke cells")

    indexed = {
        (
            int(row["pair_id"]),
            str(row["patch_mode"]),
            str(row["condition"]),
            int(row["continuation"]),
        ): row
        for row in rows
    }
    for pair_id in pair_ids:
        for continuation in range(continuation_count):
            full = indexed.get((pair_id, "full", "baseline", continuation))
            subspace = indexed.get((pair_id, "subspace", "baseline", continuation))
            if full and subspace:
                if full["generated_token_ids"] != subspace["generated_token_ids"]:
                    errors.append(
                        f"pair {pair_id} duplicated baselines differ at continuation "
                        f"{continuation}"
                    )
                if not full.get("has_valid_answer"):
                    errors.append(f"pair {pair_id} baseline has no valid answer")
                if full.get("degenerate_output"):
                    errors.append(f"pair {pair_id} baseline is degenerate")
                if full.get("hit_token_limit"):
                    errors.append(f"pair {pair_id} baseline hit the token limit")

    residuals = [
        float(row["reconstruction"]["coordinate_reconstruction_relative_residual"])
        for row in rows
        if row.get("reconstruction")
    ]
    leakages = [
        float(row["reconstruction"]["orthogonal_leakage_relative_residual"])
        for row in rows
        if row.get("reconstruction")
    ]
    if not residuals:
        errors.append("smoke run contains no subspace reconstruction diagnostics")
    elif max(residuals) > residual_tolerance:
        errors.append(
            f"coordinate residual {max(residuals):.3g} exceeds {residual_tolerance:.3g}"
        )
    if leakages and max(leakages) > residual_tolerance:
        errors.append(
            f"orthogonal leakage {max(leakages):.3g} exceeds {residual_tolerance:.3g}"
        )

    report = {
        "run": run_path.as_posix(),
        "passed": not errors,
        "errors": errors,
        "pairs": len(pair_ids),
        "expected_cells": len(expected),
        "observed_cells": len(observed),
        "maximum_coordinate_residual": max(residuals) if residuals else None,
        "maximum_orthogonal_leakage": max(leakages) if leakages else None,
        "residual_tolerance": residual_tolerance,
    }
    report_path = run_path / "preflight" / "smoke_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError(f"H3 smoke gate failed; see {report_path}")
    return report_path


def grouped_rate_summary(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any] | None:
    """Summarize a boolean outcome with question-grouped uncertainty.

    Args:
        rows: Generation or analysis records to process.
        field: Record field to read or summarize.

    Returns:
        The resulting keyed records or metrics.
    """
    by_question: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            by_question[str(row["target_question"])].append(float(bool(value)))
    values = np.asarray(
        [np.mean(question_rows) for question_rows in by_question.values()],
        dtype=np.float64,
    )
    if not len(values):
        return None
    return {
        "question_mean": float(values.mean()),
        "question_bootstrap_95ci": bootstrap_mean_interval(values),
        "questions": len(values),
    }


def reconstruction_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize available subspace reconstruction diagnostics.

    Args:
        rows: Generation or analysis records to process.

    Returns:
        The resulting keyed records or metrics.
    """
    records = [
        row["reconstruction"]
        for row in rows
        if isinstance(row.get("reconstruction"), dict)
    ]
    if not records:
        return None
    return {
        "continuations": len(records),
        **{
            field: {
                "mean": float(np.mean([record[field] for record in records])),
                "maximum": float(np.max([record[field] for record in records])),
            }
            for field in records[0]
        },
    }


def paired_comparison(
    by_pair_cell: dict[tuple[int, str, str], list[dict[str, Any]]],
    *,
    treatment: tuple[str, str],
    reference: tuple[str, str],
) -> dict[str, Any] | None:
    """Compare two cells after averaging paired effects by question.

    Args:
        by_pair_cell: Intervention rows indexed by pair, mode, and condition.
        treatment: Treatment cell name.
        reference: Reference cell name.

    Returns:
        The resulting keyed records or metrics.
    """
    pair_ids = sorted({pair_id for pair_id, _, _ in by_pair_cell})
    metrics = {}
    pair_count = 0
    question_count = 0
    for field in RATE_FIELDS:
        by_question: defaultdict[str, list[float]] = defaultdict(list)
        for pair_id in pair_ids:
            treatment_rows = by_pair_cell.get((pair_id, *treatment), [])
            reference_rows = by_pair_cell.get((pair_id, *reference), [])
            treatment_values = present_bool_values(treatment_rows, field)
            reference_values = present_bool_values(reference_rows, field)
            if not treatment_values or not reference_values:
                continue
            question = str(treatment_rows[0]["target_question"])
            by_question[question].append(
                float(np.mean(treatment_values) - np.mean(reference_values))
            )
        values = np.asarray(
            [np.mean(pair_differences) for pair_differences in by_question.values()],
            dtype=np.float64,
        )
        if len(values):
            pair_count = sum(
                len(pair_differences) for pair_differences in by_question.values()
            )
            question_count = len(values)
            metrics[field] = {
                "question_mean_difference": float(values.mean()),
                "question_bootstrap_95ci": bootstrap_mean_interval(values),
            }
    if not metrics:
        return None
    return {
        "treatment": {"patch_mode": treatment[0], "condition": treatment[1]},
        "reference": {"patch_mode": reference[0], "condition": reference[1]},
        "pair_count": pair_count,
        "question_count": question_count,
        "metrics": metrics,
    }


def present_bool_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    """Convert present boolean field values to numeric rates.

    Args:
        rows: Generation or analysis records to process.
        field: Record field to read or summarize.

    Returns:
        The resulting ordered records or values.
    """
    return [float(bool(row[field])) for row in rows if row.get(field) is not None]


def fallback_gate_inputs(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Extract prespecified attention-to-MLP fallback contrasts.

    Args:
        cells: Available patching cells keyed by mode and condition.

    Returns:
        The resulting keyed records or metrics.
    """
    indexed = {
        (cell["patch_mode"], cell["condition"]): cell
        for cell in cells
        if cell["completion_fraction"] == 1.0
    }
    required = (
        ("full", "equivalent"),
        ("full", "position_random"),
        ("subspace", "equivalent"),
        ("subspace", "position_random"),
    )
    if not all(key in indexed for key in required):
        return None

    def rate(key: tuple[str, str], field: str) -> float | None:
        """Read one question-mean cell rate when available.

        Args:
            key: Sample and seed key identifying a trace.
            field: Record field to read or summarize.

        Returns:
            The computed scalar metric, or ``None`` when unavailable.
        """
        summary = indexed[key]["rates"][field]
        return summary["question_mean"] if summary else None

    return {
        "full_equivalent_minus_random_target_accuracy": subtract_optional(
            rate(("full", "equivalent"), "matches_target_answer"),
            rate(("full", "position_random"), "matches_target_answer"),
        ),
        "subspace_equivalent_minus_random_target_accuracy": subtract_optional(
            rate(("subspace", "equivalent"), "matches_target_answer"),
            rate(("subspace", "position_random"), "matches_target_answer"),
        ),
        "full_equivalent_minus_random_collapse": subtract_optional(
            rate(("full", "equivalent"), "degenerate_output"),
            rate(("full", "position_random"), "degenerate_output"),
        ),
        "subspace_equivalent_minus_random_collapse": subtract_optional(
            rate(("subspace", "equivalent"), "degenerate_output"),
            rate(("subspace", "position_random"), "degenerate_output"),
        ),
        "interpretation": (
            "Use these prespecified inputs to decide whether the attention-18 "
            "result triggers the MLP-18 fallback; no automatic causal verdict."
        ),
    }


def subtract_optional(left: float | None, right: float | None) -> float | None:
    """Subtract two values when both are present.

    Args:
        left: Left operand or comparison input.
        right: Right operand or comparison input.

    Returns:
        The computed scalar metric, or ``None`` when unavailable.
    """
    return left - right if left is not None and right is not None else None


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    draws: int = 1000,
) -> list[float]:
    """Bootstrap a deterministic 95% interval for a mean.

    Args:
        values: Values to summarize or transform.
        draws: Number of bootstrap resamples.

    Returns:
        The resulting ordered records or values.
    """
    rng = np.random.default_rng(42)
    means = [
        float(np.mean(rng.choice(values, size=len(values), replace=True)))
        for _ in range(draws)
    ]
    return np.quantile(means, [0.025, 0.975]).tolist()
