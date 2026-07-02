"""Resumable DeepSeek proposals for silver solution-object labels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.experiments.solution_object_edits import validate_silver_label
from src.runtime.artifact_store import append_jsonl
from src.runtime.data import load_samples


MODEL = "deepseek-v4-flash"
ENDPOINT = "https://api.deepseek.com/chat/completions"
SYSTEM_PROMPT = """You label one sentence as an edit to a partial math solution.
Return only a JSON object with:
edit_type: one of none, add_entity, add_quantity, bind_variable, add_relation,
derive_value, verify, extract_answer, mixed;
entities: string list;
quantities: list of {name, value, unit};
relations: string list;
operation: concise string or null;
confidence: number from 0 to 1;
rationale: one short sentence.
Do not invent numbers or relations absent from the current sentence."""


def label_benchmark_with_deepseek(
    bronze_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    workers: int = 8,
) -> Path:
    """Propose and validate resumable silver labels with DeepSeek V4 Flash.

    Args:
        bronze_path: Bronze benchmark sentence JSONL.
        output_path: Destination for append-only silver proposal records.
        limit: Optional maximum number of pending records to label.
        workers: Number of concurrent API requests.

    Returns:
        Path to the silver proposal JSONL.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for silver labeling")
    rows = load_samples(bronze_path.resolve())
    completed = (
        {str(row["record_id"]) for row in load_samples(output_path.resolve())}
        if output_path.exists()
        else set()
    )
    pending = [
        contextualize_sentence(rows, index)
        for index, row in enumerate(rows)
        if str(row["record_id"]) not in completed
    ]
    if limit is not None:
        pending = pending[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(propose_silver_label, row, api_key): row for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                label = future.result()
                errors = validate_silver_label(
                    str(row["text"]),
                    label,
                    row["bronze_edits"],
                )
                record = {
                    "record_id": row["record_id"],
                    "model": MODEL,
                    "accepted": not errors,
                    "validation_errors": errors,
                    "silver_label": label,
                }
            except Exception as error:
                record = {
                    "record_id": row["record_id"],
                    "model": MODEL,
                    "accepted": False,
                    "validation_errors": [f"{type(error).__name__}: {error}"],
                    "silver_label": None,
                }
            append_jsonl(output_path, record)
    return output_path


def propose_silver_label(
    row: dict[str, Any],
    api_key: str,
    *,
    retries: int = 3,
) -> dict[str, Any]:
    """Request one structured sentence label with bounded retry backoff.

    Args:
        row: Bronze sentence record.
        api_key: DeepSeek API bearer token.
        retries: Maximum request attempts.

    Returns:
        Parsed JSON object from the model response.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": row["question"],
                        "previous_sentence": row.get("previous_sentence"),
                        "sentence": row["text"],
                        "next_sentence": row.get("next_sentence"),
                        "bronze_edits": row["bronze_edits"],
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 500,
        "stream": False,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=90) as response:
                body = json.loads(response.read())
            content = body["choices"][0]["message"]["content"]
            label = json.loads(content)
            if not isinstance(label, dict):
                raise ValueError("model response must be a JSON object")
            return label
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, KeyError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable retry state")


def contextualize_sentence(
    rows: list[dict[str, Any]],
    index: int,
) -> dict[str, Any]:
    """Attach adjacent same-trace sentences without crossing trace boundaries.

    Args:
        rows: Ordered bronze benchmark records.
        index: Index of the sentence being labeled.

    Returns:
        A copy of the record with optional previous and next sentence text.
    """
    row = dict(rows[index])
    trace = (row["sample_id"], row["seed"])
    previous = rows[index - 1] if index > 0 else None
    following = rows[index + 1] if index + 1 < len(rows) else None
    row["previous_sentence"] = (
        previous["text"]
        if previous and (previous["sample_id"], previous["seed"]) == trace
        else None
    )
    row["next_sentence"] = (
        following["text"]
        if following and (following["sample_id"], following["seed"]) == trace
        else None
    )
    return row
