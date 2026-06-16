from __future__ import annotations

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

    raise ValueError(f"Unknown dataset adapter: {adapter!r}")


def normalize_dataset(
    rows: list[dict[str, Any]],
    adapter: str,
) -> list[dict[str, Any]]:
    return [adapt_row(row, adapter, i) for i, row in enumerate(rows)]
