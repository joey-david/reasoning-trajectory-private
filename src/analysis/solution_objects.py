from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.analysis.answers import extract_answer
from src.analysis.common import read_generation_rows, read_sample_records, write_jsonl


NUMBER_RE = r"-?\d+(?:\.\d+)?"


def write_solution_objects(run_path: Path, cfg: dict[str, Any]) -> None:
    rows = read_generation_rows(run_path)
    samples = read_sample_records(run_path)
    out_dir = run_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = [
        build_solution_object(row, samples.get(row["sample_id"], {}), cfg)
        for row in rows
    ]
    write_jsonl(out_dir / "solution_objects.jsonl", objects)


def build_solution_object(
    row: dict[str, Any],
    sample: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    produced_text = row.get("produced_text", "")
    reasoning_text, final_text = split_reasoning_and_final(produced_text)
    produced_answer = row.get("produced_answer") or extract_answer(
        produced_text,
        cfg.get("produced_answer_regex"),
    )
    gold_answer = extract_answer(
        str(sample.get("gold_answer", "")),
        cfg.get("gold_answer_regex"),
    )

    return {
        "sample_id": row.get("sample_id"),
        "seed": row.get("seed"),
        "dataset_source": sample.get("source")
        or sample.get("metadata", {}).get("source"),
        "question": sample.get("question") or sample.get("prompt"),
        "reasoning_text": reasoning_text,
        "final_text": final_text,
        "produced_answer": produced_answer,
        "gold_answer": gold_answer,
        "numeric_values": re.findall(NUMBER_RE, produced_text.replace(",", "")),
        "latent_anchor": {
            "hidden_states_file": row.get("hidden_states_file"),
            "dp1_idx": sample.get("dp1_idx"),
            "dp2_idx": row.get("dp2_idx"),
            "reasoning_length": row.get("reasoning_length"),
        },
        "is_correct": row.get("is_correct"),
    }


def split_reasoning_and_final(text: str) -> tuple[str | None, str]:
    match = re.search(r"<think>\s*(.*?)\s*</think>\s*(.*)", text, re.S)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, text.strip()
