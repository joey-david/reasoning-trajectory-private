"""Data contracts shared by generation and activation capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GenerationRequest:
    """Hold all resolved inputs needed to generate and analyze one rollout."""

    prompt: str
    sample_id: str
    seed: int
    temperature: float
    max_new_tokens: int
    forced_prefix: str
    stop_regex: str | None
    cap_fallback_prefix: str
    cap_fallback_min_new_tokens: int
    cap_fallback_max_new_tokens: int
    layer_indices: list[int]
    capture_components: list[str]
    model_name: str
    gold_answer: str | None
    gold_token_id: int | None
    capture_diagnostics: bool
    top_p: float | None
    top_k: int | None
    progress: Any | None
    progress_label: str


@dataclass(slots=True)
class GeneratedSequence:
    """Hold the full sequence, generated suffix, and decoded generated text."""

    full_ids: list[int]
    token_ids: list[int]
    text: str
