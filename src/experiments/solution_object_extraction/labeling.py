"""Token-window preparation and local solution-object labeling."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from reasoning_trajectory.artifacts import read_generation_rows
from reasoning_trajectory.token_alignment import build_token_spans, token_range_for_chars
from src.runtime.data import load_samples


SYSTEM_PROMPT = """Segment a reasoning excerpt into coherent solution-object edits.
Do not use sentence punctuation as a required boundary: a unit may be shorter or
longer than a sentence. Return only:
{"spans":[{"text":str,"label":str,"confidence":number}]}
For each unit, copy the whole concerned contiguous interval verbatim from the
excerpt. Make the quote long enough to occur exactly once; include surrounding
words when a shorter expression is repeated. Return only complete units fully
contained in the excerpt. Allowed labels: orient, bind_variable, add_relation,
derive_value, verify, correct_error, plan, extract_answer, mixed."""

ALLOWED_LABELS = {
    "orient",
    "bind_variable",
    "add_relation",
    "derive_value",
    "verify",
    "correct_error",
    "plan",
    "extract_answer",
    "mixed",
}

EXAMPLE_INPUT = {
    "question": "There are 12 boxes with 3 pens each. How many pens are there?",
    "excerpt": (
        "Let n be the number of boxes. Set n = 12, so there are 12 boxes. "
        "Each box holds 3 pens, giving 12 * 3 = 36 pens. "
        "Therefore the answer is 36."
    ),
    "bronze_edits": [
        {
            "edit_type": "OPERATE",
            "operation": "MULTIPLY",
            "text": "12 * 3 = 36",
            "start_char": 95,
            "end_char": 106,
        }
    ],
}
EXAMPLE_OUTPUT = {
    "spans": [
        {
            "text": (
                "Let n be the number of boxes. Set n = 12, so there are 12 boxes."
            ),
            "label": "bind_variable",
            "confidence": 0.98,
        },
        {
            "text": "Each box holds 3 pens, giving 12 * 3 = 36 pens.",
            "label": "derive_value",
            "confidence": 0.99,
        },
        {
            "text": "Therefore the answer is 36.",
            "label": "extract_answer",
            "confidence": 0.99,
        },
    ]
}


def build_label_windows(
    source_run: Path,
    updates_path: Path,
    output_path: Path,
    *,
    window_tokens: int = 256,
    overlap_tokens: int = 48,
) -> int:
    """Write overlapping token windows for one shortest correct trace per question."""
    rows = shortest_correct_rows(read_generation_rows(source_run))
    spans_by_row = build_token_spans(source_run, rows)
    updates = load_updates(updates_path)
    questions = {
        str(row["id"]): str(row.get("question", ""))
        for row in load_samples((source_run / "dataset.jsonl").resolve())
    }
    step = window_tokens - overlap_tokens
    if step <= 0:
        raise ValueError("overlap_tokens must be smaller than window_tokens")
    records: list[dict[str, Any]] = []
    for row, token_spans in zip(rows, spans_by_row, strict=True):
        key = str(row["sample_id"]), int(row["seed"])
        for token_start in range(0, len(token_spans), step):
            token_end = min(token_start + window_tokens, len(token_spans))
            window = window_chars(token_spans, token_start, token_end)
            if window is None:
                continue
            char_start, char_end = window
            text = str(row["produced_text"])[char_start:char_end]
            bronze = []
            for update in updates.get(key, []):
                if update["char_start"] < char_start or update["char_end"] > char_end:
                    continue
                bronze.append(
                    {
                        "edit_type": update["operator"],
                        "operation": update["operation_signature"],
                        "text": update["expression"],
                        "start_char": update["char_start"] - char_start,
                        "end_char": update["char_end"] - char_start,
                    }
                )
            records.append(
                {
                    "record_id": f"{key[0]}::{key[1]}::{token_start}-{token_end}",
                    "sample_id": key[0],
                    "seed": key[1],
                    "question": questions.get(key[0], ""),
                    "token_start": token_start,
                    "token_end": token_end,
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_char_spans": [
                        (
                            [span[0] - char_start, span[1] - char_start]
                            if span is not None
                            else None
                        )
                        for span in token_spans[token_start:token_end]
                    ],
                    "text": text,
                    "bronze_edits": bronze,
                }
            )
            if token_end == len(token_spans):
                break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def shortest_correct_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one concise correct trace for each question."""
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row.get("is_correct") or not row.get("generated_token_ids"):
            continue
        sample_id = str(row["sample_id"])
        current = selected.get(sample_id)
        if current is None or len(row["generated_token_ids"]) < len(
            current["generated_token_ids"]
        ):
            selected[sample_id] = row
    return [selected[key] for key in sorted(selected)]


def load_updates(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Group deterministic symbolic edits by trace."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            grouped.setdefault(
                (str(row["sample_id"]), int(row["seed"])), []
            ).append(row)
    return grouped


def window_chars(
    spans: list[tuple[int, int] | None],
    token_start: int,
    token_end: int,
) -> tuple[int, int] | None:
    """Map a token window to its exact decoded-text character interval."""
    valid = [span for span in spans[token_start:token_end] if span is not None]
    return (valid[0][0], valid[-1][1]) if valid else None


def completed_silver_labels(path: Path) -> set[str]:
    """Load record IDs already persisted by the labeling job."""
    if not path.exists():
        return set()
    return {str(row["record_id"]) for row in load_samples(path.resolve())}


def label_messages(
    row: dict[str, Any],
    *,
    previous_output: str | None = None,
    feedback: str | None = None,
) -> list[dict[str, str]]:
    """Build the demonstrated labeling conversation, including retry feedback."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(EXAMPLE_INPUT, ensure_ascii=False),
        },
        {
            "role": "assistant",
            "content": json.dumps(EXAMPLE_OUTPUT, ensure_ascii=False),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": row["question"],
                    "excerpt": row["text"],
                    "bronze_edits": row["bronze_edits"],
                },
                ensure_ascii=False,
            ),
        },
    ]
    if previous_output is not None and feedback is not None:
        messages.extend(
            [
                {"role": "assistant", "content": previous_output},
                {
                    "role": "user",
                    "content": (
                        "The previous response could not be aligned to the excerpt:\n"
                        f"{feedback}\nReturn corrected JSON only. Copy longer, "
                        "verbatim text intervals so every quote has one match."
                    ),
                },
            ]
        )
    return messages


def generate_hf_output(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = 1200,
) -> tuple[str, int]:
    """Generate one deterministic label with the Hugging Face backend."""
    import torch

    from src.models.introspection import get_input_device

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(get_input_device(model))
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, input_ids.shape[1] :].detach().cpu()
    return (
        tokenizer.decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        int(generated.numel()),
    )


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from model output."""
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model output does not contain a valid JSON object")


def resolve_window_label(
    row: dict[str, Any],
    label: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve quoted intervals to unique offsets and validate their labels."""
    spans = label.get("spans")
    if not isinstance(spans, list):
        return {"spans": []}, ["spans must be a list"]
    errors: list[str] = []
    excerpt = str(row["text"])
    resolved: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            errors.append(f"span {index} must be an object")
            continue
        quote = span.get("text")
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"span {index} text must be a nonempty string")
            continue
        match, match_error = locate_unique_quote(excerpt, quote)
        if match_error:
            errors.append(f"span {index}: {match_error}")
            continue
        label_name = span.get("label")
        if label_name not in ALLOWED_LABELS:
            errors.append(f"span {index}: unsupported label {label_name!r}")
        confidence = span.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            errors.append(f"span {index}: confidence must be in [0, 1]")
        assert match is not None
        record = {
            "text": excerpt[match[0] : match[1]],
            "label": label_name,
            "confidence": confidence,
            "start_char": match[0],
            "end_char": match[1],
        }
        local_tokens = token_range_for_chars(
            row.get("token_char_spans", []), match[0], match[1]
        )
        if local_tokens is not None:
            record["token_start"] = int(row.get("token_start", 0)) + local_tokens[0]
            record["token_end"] = int(row.get("token_start", 0)) + local_tokens[1]
        resolved.append(record)
    for left, right in zip(
        sorted(resolved, key=lambda item: item["start_char"]),
        sorted(resolved, key=lambda item: item["start_char"])[1:],
        strict=False,
    ):
        if left["end_char"] > right["start_char"]:
            errors.append("resolved spans overlap")
            break
    return {"spans": resolved}, errors


def locate_unique_quote(
    excerpt: str,
    quote: str,
) -> tuple[tuple[int, int] | None, str | None]:
    """Find one verbatim quote, allowing only whitespace normalization as fallback."""
    exact = list(re.finditer(re.escape(quote), excerpt))
    if len(exact) == 1:
        return exact[0].span(), None
    if len(exact) > 1:
        return None, f"quoted text occurs {len(exact)} times; include more context"
    parts = re.split(r"\s+", quote.strip())
    flexible = re.compile(r"\s+".join(re.escape(part) for part in parts))
    matches = list(flexible.finditer(excerpt))
    if len(matches) == 1:
        return matches[0].span(), None
    if len(matches) > 1:
        return None, (
            f"whitespace-normalized quote occurs {len(matches)} times; "
            "include more context"
        )
    return None, "quoted text does not occur in the excerpt; copy it verbatim"
