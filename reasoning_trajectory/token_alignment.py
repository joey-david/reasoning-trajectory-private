"""Align generated token IDs with character spans in decoded transcripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reasoning_trajectory.config import load_run_config
from tokenizers.decoders import DecodeStream


TokenSpan = tuple[int, int] | None


def build_token_spans(
    run_path: Path,
    rows: list[dict[str, Any]],
) -> list[list[TokenSpan]]:
    """Map generated token indices to exact decoded-text character spans.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        rows: Generation or analysis records to process.

    Returns:
        The resulting ordered records or values.
    """
    model_cfg = load_run_config(run_path).get("model", {})
    try:
        from transformers import AutoTokenizer

        model_name = model_cfg.get("path") if model_cfg.get("backend") == "mlx" else model_cfg["name"]
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            revision=model_cfg.get("revision"),
        )
        return [token_spans_for_row(tokenizer, row) for row in rows]
    except (ImportError, OSError, TypeError, ValueError, NotImplementedError) as error:
        print(f"token alignment unavailable: {error}")
        return [[] for _ in rows]


def token_spans_for_row(tokenizer: Any, row: dict[str, Any]) -> list[TokenSpan]:
    """Align one decoded generation with its stored token IDs.

    Args:
        tokenizer: Tokenizer aligned with the loaded model.
        row: Generation or analysis record to process.

    Returns:
        The resulting ordered records or values.
    """
    generated_ids = [int(token_id) for token_id in row.get("generated_token_ids", [])]
    text = row.get("produced_text", "")
    if not generated_ids or not text:
        return [None] * len(generated_ids)

    stream = DecodeStream(skip_special_tokens=True)
    pieces = [
        stream.step(tokenizer.backend_tokenizer, token_id) or ""
        for token_id in generated_ids
    ]
    if "".join(pieces) != text:
        return [None] * len(generated_ids)

    spans: list[TokenSpan] = [None] * len(pieces)
    char_start = 0
    pending: list[int] = []
    for token_idx, piece in enumerate(pieces):
        pending.append(token_idx)
        if not piece:
            continue
        char_end = char_start + len(piece)
        for pending_idx in pending:
            spans[pending_idx] = (char_start, char_end)
        pending.clear()
        char_start = char_end
    for pending_idx in pending:
        spans[pending_idx] = (char_start, char_start)
    return spans


def token_range_for_chars(
    spans: list[TokenSpan],
    char_start: int,
    char_end: int,
) -> tuple[int, int] | None:
    """Return the inclusive token range overlapping a character span.

    Args:
        spans: Token spans used for character alignment.
        char_start: Inclusive character offset.
        char_end: Exclusive character offset.

    Returns:
        The computed aligned values described above.
    """
    indices = [
        token_idx
        for token_idx, span in enumerate(spans)
        if span is not None and span[1] > char_start and span[0] < char_end
    ]
    return (indices[0], indices[-1]) if indices else None
