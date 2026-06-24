from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.common import read_generation_rows, read_sample_records, write_jsonl


def write_hard_questions(run_path: Path, cfg: dict[str, Any]) -> None:
    rows = read_generation_rows(run_path)
    samples = read_sample_records(run_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["sample_id"], []).append(row)

    candidates = [
        score_sample(sample_id, sample_rows, samples.get(sample_id, {}))
        for sample_id, sample_rows in grouped.items()
    ]
    candidates.sort(key=lambda x: (-x["hardness_score"], x["sample_id"]))

    limit = int(cfg.get("hard_question_limit", 50))
    out_dir = run_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "hard_questions.jsonl", candidates[:limit])


def score_sample(
    sample_id: str,
    rows: list[dict[str, Any]],
    sample: dict[str, Any],
) -> dict[str, Any]:
    correctness = [row.get("is_correct") for row in rows]
    known = [x for x in correctness if x is not None]
    wrong_rate = 0.0 if not known else sum(x is False for x in known) / len(known)
    unknown_rate = sum(x is None for x in correctness) / max(len(correctness), 1)
    lengths = [len(row.get("generated_token_ids", [])) for row in rows]
    avg_tokens = sum(lengths) / max(len(lengths), 1)
    disagreement = len({row.get("produced_answer") for row in rows}) / max(len(rows), 1)
    hardness_score = (
        (2.0 * wrong_rate) + unknown_rate + min(avg_tokens / 4096.0, 1.0) + disagreement
    )

    return {
        "sample_id": sample_id,
        "hardness_score": round(hardness_score, 4),
        "wrong_rate": round(wrong_rate, 4),
        "unknown_rate": round(unknown_rate, 4),
        "answer_disagreement": round(disagreement, 4),
        "avg_generated_tokens": round(avg_tokens, 1),
        "question": sample.get("question") or sample.get("prompt"),
        "gold_answer": sample.get("gold_answer"),
    }
