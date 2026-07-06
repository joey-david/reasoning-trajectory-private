"""Extract comparable answers from generated text and update rollout correctness metadata."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from reasoning_trajectory.artifacts import read_generation_rows, read_sample_records
from src.runtime.data import write_jsonl


NUMBER_RE = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


def update_answers(run_path: Path, cfg: dict[str, Any]) -> None:
    """Recompute answer, correctness, and reasoning-boundary fields for a run.

    Args:
        run_path: Completed run folder containing generation and sample records.
        cfg: Analysis options for answer regexes and the think-end token ID.

    Returns:
        None; the generation JSONL file is replaced in place.
    """
    gen_path = run_path / "generation" / "generations.jsonl"
    rows = read_generation_rows(run_path)
    samples = read_sample_records(run_path)
    produced_re = cfg.get("produced_answer_regex")
    gold_re = cfg.get("gold_answer_regex")
    think_end_id = cfg.get("think_end_token_id", 151668)

    for row in rows:
        sample = samples[row["sample_id"]]
        row["produced_answer"] = extract_answer(
            row.get("produced_text", ""), produced_re
        )
        gold = extract_answer(sample.get("gold_answer", ""), gold_re)
        row["is_correct"] = answers_match(
            row["produced_answer"],
            gold,
        )
        if think_end_id in row.get("generated_token_ids", []):
            row["reasoning_length"] = row["generated_token_ids"].index(think_end_id) + 1
            row["dp2_idx"] = sample.get("dp1_idx", 0) + row["reasoning_length"]

    write_jsonl(gen_path, rows)


def extract_answer(text: str, pattern: str | None) -> str | None:
    """Extract an answer using a configured regex or the final numeric value.

    Args:
        text: Generated or gold-answer text to inspect.
        pattern: Optional regular expression whose final participating group wins.

    Returns:
        The stripped matched answer, final number, or ``None`` when absent.
    """
    if pattern:
        match = re.search(pattern, text, re.S)
        if match:
            groups = [g for g in match.groups() if g is not None]
            return (groups[-1] if groups else match.group(0)).strip()
    nums = re.findall(NUMBER_RE, text.replace(",", ""))
    return nums[-1] if nums else None


def answers_match(
    a: str | None,
    b: str | None,
) -> bool | None:
    """Compare two extracted answers numerically when possible, otherwise as text.

    Args:
        a: First normalized answer.
        b: Second normalized answer.

    Returns:
        Whether the answers match, or ``None`` when either answer is missing.
    """
    if a is None or b is None:
        return None
    try:
        return Decimal(a.replace(",", "")) == Decimal(b.replace(",", ""))
    except InvalidOperation:
        return a.strip().lower() == b.strip().lower()
