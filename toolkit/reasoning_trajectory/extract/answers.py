from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def normalize_number(text: str | int | float | None) -> str | None:
    if text is None:
        return None
    match = NUMBER_RE.search(str(text).replace("$", ""))
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return raw
    return str(value.normalize()) if value == value.to_integral() else str(value.normalize())


def extract_final_answer(text: str) -> str | None:
    marker = re.search(r"(?is)(?:final\s+answer|answer)\s*:?\s*(.+)$", text)
    tail = marker.group(1) if marker else text
    matches = NUMBER_RE.findall(tail.replace("$", ""))
    if not matches:
        return None
    return normalize_number(matches[-1])


def answer_correct(generated_text: str, expected_answer: str | int | float | None) -> tuple[str | None, bool | None]:
    predicted = extract_final_answer(generated_text)
    expected = normalize_number(expected_answer)
    if expected is None:
        return predicted, None
    if predicted is None:
        return None, False
    return predicted, predicted == expected
