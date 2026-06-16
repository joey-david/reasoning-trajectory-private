from __future__ import annotations

from typing import Any


def generation_record(
    *,
    sample: dict[str, Any],
    prompt: str,
    text: str,
    token_ids: list[int],
    seed: int,
    temperature: float,
    activation_file: str | None,
) -> dict[str, Any]:
    """Create one JSON-serializable generation row."""
    return {
        "sample_id": sample.get("id") or sample.get("problem_id"),
        "seed": seed,
        "temperature": temperature,
        "prompt": prompt,
        "text": text,
        "token_ids": token_ids,
        "activation_file": activation_file,
        "expected_answer": sample.get("expected_answer") or sample.get("answer"),
    }
