"""Adapt supported benchmark schemas into the repository's normalized sample records."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def adapt_row(row: dict[str, Any], adapter: str, idx: int) -> dict[str, Any]:
    """Normalize one source row using a named dataset adapter.

    Args:
        row: Raw source-dataset record.
        adapter: Supported adapter name selecting the source schema.
        idx: Source row index used to synthesize stable fallback IDs.

    Returns:
        A record with ``id``, ``question``, ``gold_answer``, ``source``, and
        the original row under ``metadata``.
    """
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
        correct = (
            row.get("Correct Answer") or row.get("correct_answer") or row.get("answer")
        )
        incorrect = [
            row.get("Incorrect Answer 1") or row.get("incorrect_answer_1"),
            row.get("Incorrect Answer 2") or row.get("incorrect_answer_2"),
            row.get("Incorrect Answer 3") or row.get("incorrect_answer_3"),
        ]
        choices = [(str(correct), True), *[(str(x), False) for x in incorrect if x]]
        stable_shuffle(choices, str(row.get("Record ID") or row.get("id") or idx))
        gold = next(
            chr(65 + i) for i, (_, is_correct) in enumerate(choices) if is_correct
        )
        prompt = "\n".join(
            [
                str(question),
                "",
                *[f"{chr(65 + i)}. {choice}" for i, (choice, _) in enumerate(choices)],
            ]
        )
        return {
            "id": str(row.get("Record ID") or row.get("id") or f"gpqa_{idx:06d}"),
            "question": prompt,
            "gold_answer": gold,
            "source": "Idavidrein/gpqa",
            "metadata": row,
        }

    if adapter == "bigcodebench":
        prompt = (
            row.get("instruct_prompt")
            or row.get("complete_prompt")
            or row.get("question")
        )
        code_prompt = row.get("code_prompt")
        if code_prompt:
            prompt = f"{prompt}\n\nStarter code:\n```python\n{code_prompt}\n```"
        return {
            "id": str(
                row.get("task_id") or row.get("_id") or f"bigcodebench_{idx:06d}"
            ),
            "question": str(prompt),
            "gold_answer": None,
            "source": "bigcode/bigcodebench-hard",
            "metadata": row,
        }

    if adapter == "mbppplus":
        public_tests = "\n".join(str(test) for test in row.get("test_list", []))
        return {
            "id": f"mbppplus_{row['task_id']}",
            "question": (
                f"{row['prompt']}\n\n"
                "The implementation must satisfy these examples:\n"
                f"```python\n{public_tests}\n```"
            ),
            "gold_answer": None,
            "source": "evalplus/mbppplus",
            "metadata": row,
        }

    if adapter == "aime":
        return {
            "id": f"aime_{row.get('year', 2025)}_{row.get('problem_idx', row.get('id', idx))}",
            "question": str(row["problem"]),
            "gold_answer": str(row["answer"]),
            "source": "AIME",
            "metadata": row,
        }

    if adapter == "olympiadbench_numeric":
        answer = simple_numeric_answer(row.get("final_answer"))
        return {
            "id": f"olympiadbench_{row.get('id', idx)}",
            "question": str(row["question"]),
            "gold_answer": answer,
            "source": "Hothan/OlympiadBench",
            "metadata": row,
        }

    if adapter == "polymath_numeric":
        return {
            "id": str(row.get("id") or f"polymath_{idx:06d}"),
            "question": str(row["question"]),
            "gold_answer": simple_numeric_answer([row.get("answer")]),
            "source": "Qwen/PolyMath",
            "metadata": row,
        }

    raise ValueError(f"Unknown dataset adapter: {adapter!r}")


def stable_shuffle(items: list[Any], key: str) -> None:
    """Shuffle a list deterministically using a string key.

    Args:
        items: Mutable list to shuffle in place.
        key: Stable value hashed into the pseudo-random seed.

    Returns:
        None.
    """
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(seed).shuffle(items)


def simple_numeric_answer(value: Any) -> str | None:
    """Extract a single scalar numeric answer from a one-element list.

    Args:
        value: Candidate source answer.

    Returns:
        A normalized numeric string, or ``None`` for unsupported values.
    """
    if not isinstance(value, list) or len(value) != 1:
        return None
    answer = str(value[0]).strip().strip("$").replace(",", "").strip()
    try:
        float(answer)
    except ValueError:
        return None
    return answer


def normalize_dataset(
    rows: list[dict[str, Any]],
    adapter: str,
) -> list[dict[str, Any]]:
    """Normalize all rows with one adapter.

    Args:
        rows: Raw source records.
        adapter: Adapter name passed to :func:`adapt_row`.

    Returns:
        Normalized records in source order.
    """
    return [adapt_row(row, adapter, i) for i, row in enumerate(rows)]
