from __future__ import annotations

import hashlib
import random
from typing import Any


def adapt_row(row: dict[str, Any], adapter: str, idx: int) -> dict[str, Any]:
    if adapter == "math_qa":
        return {
            "id": str(row.get("id") or row.get("problem_id") or f"item_{idx:06d}"),
            "question": str(
                row.get("problem") or row.get("question") or row.get("input")
            ),
            "gold_answer": row.get("answer") or row.get("final_answer"),
            "source": adapter,
            "metadata": row,
        }

    if adapter == "plain_question":
        return {
            "id": str(row.get("id") or f"item_{idx:06d}"),
            "question": str(
                row.get("question") or row.get("prompt") or row.get("input")
            ),
            "gold_answer": row.get("answer") or row.get("gold_answer"),
            "source": adapter,
            "metadata": row,
        }

    if adapter == "gsm_symbolic":
        template_id = row.get("id")
        instance_id = row.get("instance")

        return {
            "id": f"gsm_symbolic_{template_id}_{instance_id}",
            "question": str(row["question"]),
            "gold_answer": row.get("answer"),
            "source": "apple/GSM-Symbolic",
            "metadata": row,
        }

    if adapter == "gsm8k":
        return {
            "id": str(row.get("id") or f"gsm8k_{idx:06d}"),
            "question": str(row["question"]),
            "gold_answer": row.get("answer"),
            "source": "openai/gsm8k",
            "metadata": row,
        }

    if adapter == "hendrycks_math":
        return {
            "id": str(row.get("id") or f"math_{idx:06d}"),
            "question": str(row.get("problem") or row.get("question")),
            "gold_answer": row.get("solution") or row.get("answer"),
            "source": "EleutherAI/hendrycks_math",
            "metadata": row,
        }

    if adapter == "gpqa":
        question = row.get("Question") or row.get("question") or row.get("prompt")
        correct = row.get("Correct Answer") or row.get("correct_answer") or row.get("answer")
        incorrect = [
            row.get("Incorrect Answer 1") or row.get("incorrect_answer_1"),
            row.get("Incorrect Answer 2") or row.get("incorrect_answer_2"),
            row.get("Incorrect Answer 3") or row.get("incorrect_answer_3"),
        ]
        choices = [(str(correct), True), *[(str(x), False) for x in incorrect if x]]
        stable_shuffle(choices, str(row.get("Record ID") or row.get("id") or idx))
        gold = next(chr(65 + i) for i, (_, is_correct) in enumerate(choices) if is_correct)
        prompt = "\n".join(
            [str(question), "", *[f"{chr(65 + i)}. {choice}" for i, (choice, _) in enumerate(choices)]]
        )
        return {
            "id": str(row.get("Record ID") or row.get("id") or f"gpqa_{idx:06d}"),
            "question": prompt,
            "gold_answer": gold,
            "source": "Idavidrein/gpqa",
            "metadata": row,
        }

    raise ValueError(f"Unknown dataset adapter: {adapter!r}")


def stable_shuffle(items: list[Any], key: str) -> None:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(items)


def normalize_dataset(
    rows: list[dict[str, Any]],
    adapter: str,
) -> list[dict[str, Any]]:
    return [adapt_row(row, adapter, i) for i, row in enumerate(rows)]
