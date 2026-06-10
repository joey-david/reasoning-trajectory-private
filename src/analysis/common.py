from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import run_path


BOXED_RE = re.compile(
    r"\\boxed\s*\{([^{}]+)\}|boxed\s*[:=]?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def generation_path(config: dict[str, Any]) -> Path:
    return run_path(config) / "generation" / "generations.jsonl"


def analysis_path(config: dict[str, Any], name: str) -> Path:
    path = run_path(config) / "analysis" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_generations(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = generation_path(config)
    if not source.exists():
        return []

    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["predicted_answer"] = extract_answer(row.get("text") or "")
            row["success"] = is_success(row)
            rows.append(row)

    return rows


def activation_layers(config: dict[str, Any]) -> list[str]:
    layers = []
    base = run_path(config)

    for row in load_generations(config):
        if not row.get("activation_file"):
            continue

        import numpy as np

        arrays = np.load(base / row["activation_file"])
        for layer in arrays.files:
            if layer not in layers:
                layers.append(layer)

    return sorted(layers, key=lambda value: int(value) if value.isdigit() else value)


def extract_answer(text: str) -> str:
    """Extract a GSM8K-style numeric answer.

    Priority:
    1. Last \\boxed{...} or "boxed: N" answer.
    2. Last number anywhere in the generated text.
    3. Empty string.
    """
    if not text:
        return ""

    boxed_matches = list(BOXED_RE.finditer(text))
    if boxed_matches:
        match = boxed_matches[-1]
        return normalize_answer(match.group(1) or match.group(2) or "")

    numbers = NUMBER_RE.findall(text)
    return normalize_answer(numbers[-1]) if numbers else ""


def is_success(row: dict[str, Any]) -> bool:
    expected = row.get("expected_answer")
    if expected is None or expected == "":
        return False

    return extract_answer(row.get("text") or "") == normalize_answer(str(expected))


def normalize_answer(value: str | int | float | None) -> str:
    """Normalize lightweight numeric answers.

    Examples:
    - "$1,200" -> "1200"
    - "24.0" -> "24"
    - "\\boxed{3 football players and 12 cheerleaders}" -> "12"
    """
    if value is None:
        return ""

    text = str(value).strip().lower()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = re.sub(r"^answer\s*(is|:)?\s*", "", text).strip()

    numbers = NUMBER_RE.findall(text)
    if numbers:
        text = numbers[-1].replace(",", "")

    if re.fullmatch(r"[-+]?\d+\.0+", text):
        text = text.split(".", 1)[0]

    return text


def selected_token_indices(row: dict[str, Any], token_count: int, interval: int) -> list[int]:
    step = max(1, int(interval))
    indices = list(range(0, token_count, step))

    boxed_idx = last_before_boxed_index(row, token_count)
    if boxed_idx is not None and boxed_idx not in indices:
        indices.append(boxed_idx)

    if token_count and token_count - 1 not in indices:
        indices.append(token_count - 1)

    return sorted(i for i in indices if 0 <= i < token_count)


def last_before_boxed_index(row: dict[str, Any], token_count: int) -> int | None:
    text = row.get("text") or ""

    boxed_at = text.lower().rfind("\\boxed")
    if boxed_at < 0:
        boxed_at = text.lower().rfind("boxed")

    token_texts = row.get("token_texts") or []

    if boxed_at < 0 or not token_texts:
        return token_count - 1 if token_count else None

    cursor = 0
    for idx, token in enumerate(token_texts):
        cursor += len(str(token).replace("Ġ", " ").replace("▁", " "))
        if cursor >= boxed_at:
            return max(0, idx - 1)

    return token_count - 1 if token_count else None
