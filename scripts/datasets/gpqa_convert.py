#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


LETTERS = ["A", "B", "C", "D"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GPQA CSV files into repo-ready JSONL datasets.")
    parser.add_argument("source", nargs="?", default="datasets/gpqa/gpqa_diamond.csv")
    parser.add_argument("--out", default="datasets/gpqa/gpqa_diamond.jsonl")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    rows = convert(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")


def convert(source: Path) -> list[dict]:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [convert_row(row, idx, source) for idx, row in enumerate(reader)]


def convert_row(row: dict[str, str], idx: int, source: Path) -> dict:
    record_id = clean(row.get("Record ID")) or f"row_{idx:05d}"
    question = clean(row["Question"])
    correct = clean(row["Correct Answer"])
    incorrect = [
        clean(row["Incorrect Answer 1"]),
        clean(row["Incorrect Answer 2"]),
        clean(row["Incorrect Answer 3"]),
    ]
    choices = _shuffled_choices(record_id, correct, incorrect)
    correct_letter = next(item["letter"] for item in choices if item["is_correct"])

    prompt = (
        "Answer the following GPQA Diamond multiple-choice question. "
        "Think step by step, then end with only the final choice letter in \\\\boxed{}.\n\n"
        f"Question:\n{question}\n\n"
        "Choices:\n"
        + "\n".join(f"{item['letter']}. {item['text']}" for item in choices)
    )

    return {
        "id": f"gpqa_diamond_{record_id}",
        "prompt": prompt,
        "expected_answer": correct_letter,
        "question": question,
        "choices": {item["letter"]: item["text"] for item in choices},
        "correct_answer": correct,
        "correct_letter": correct_letter,
        "explanation": clean(row.get("Explanation")),
        "source": "gpqa_diamond",
        "record_id": record_id,
        "high_level_domain": clean(row.get("High-level domain")),
        "subdomain": clean(row.get("Subdomain")),
        "writer_difficulty_estimate": clean(row.get("Writer's Difficulty Estimate")),
        "metadata": {
            "source_file": str(source),
            "non_expert_validator_accuracy": clean(row.get("Non-Expert Validator Accuracy")),
            "expert_validator_accuracy": clean(row.get("Expert Validator Accuracy")),
            "majority_non_expert_vals_incorrect": clean(row.get("Majority Non-Expert Vals Incorrect")),
        },
    }


def _shuffled_choices(record_id: str, correct: str, incorrect: list[str]) -> list[dict]:
    items = [{"text": correct, "is_correct": True}]
    items.extend({"text": value, "is_correct": False} for value in incorrect)
    digest = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    rng.shuffle(items)
    for letter, item in zip(LETTERS, items):
        item["letter"] = letter
    return items


def clean(value: str | None) -> str:
    return " ".join((value or "").strip().split())


if __name__ == "__main__":
    main()
