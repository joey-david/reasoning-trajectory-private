from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.schema import Step, VerifierState


STEP_PATTERNS = [
    re.compile(r"(?m)^\s*(?:Step\s+\d+|[0-9]+[.)]|[-*])\s*:?\s+"),
    re.compile(r"(?m)^\s*(?:intro|apply|rw|rewrite|simp|exact|cases|induction|constructor|have)\b.*$"),
    re.compile(r"(?m)^\s*(?:def |class |return |if |for |while |assert |# edit:).*$"),
]


@dataclass
class ParsedSpan:
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    labels: list[str]


def _token_index(text: str, char: int) -> int:
    return len(text[:char].split())


def _label(line: str) -> list[str]:
    low = line.strip().lower()
    labels = []
    if low.startswith("step") or re.match(r"^\d+[.)]", low):
        labels.append("numbered")
    if re.match(r"^(intro|apply|rw|rewrite|simp|exact|cases|induction|constructor|have)\b", low):
        labels.append("proof_tactic")
    if re.match(r"^(def |class |return |if |for |while |assert |# edit:)", low):
        labels.append("code_edit")
    return labels or ["newline"]


@tool(
    "step-parser",
    "extract",
    "Segment generated reasoning into newline, numbered, proof tactic, or code-edit steps.",
    "rt parse-steps --input text.txt --out steps.jsonl",
    "reasoning_trajectory.extract.token_steps.parse_steps",
    "toolkit/docs/tools/step-parser.md",
)
def parse_steps(text: str) -> list[ParsedSpan]:
    """Parse meaningful reasoning steps from generated text."""
    markers: list[tuple[int, int]] = []
    for pattern in STEP_PATTERNS:
        markers.extend((m.start(), m.end()) for m in pattern.finditer(text))
    markers = sorted(set(markers))
    if not markers:
        lines = [(m.start(), m.end()) for m in re.finditer(r"(?m)^.+$", text) if m.group(0).strip()]
        markers = [(start, start) for start, _ in lines]
    spans: list[ParsedSpan] = []
    for i, (start, marker_end) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        spans.append(
            ParsedSpan(
                text=chunk,
                char_start=start,
                char_end=end,
                token_start=_token_index(text, start),
                token_end=max(_token_index(text, end), _token_index(text, start) + 1),
                labels=_label(text[start:marker_end] or chunk),
            )
        )
    return spans


def aggregate_step_hidden_states(
    token_hidden_states: np.ndarray,
    spans: Sequence[ParsedSpan],
    layers: Sequence[int] | None = None,
    pooling: str = "mean",
    attention_weights: np.ndarray | None = None,
) -> list[dict[str, list[float]]]:
    """Pool token hidden states into step-level vectors.

    token_hidden_states shape is [tokens, layers, hidden].
    """
    arr = np.asarray(token_hidden_states, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"expected [tokens,layers,hidden], got {arr.shape}")
    layers = list(range(arr.shape[1])) if layers is None else list(layers)
    out: list[dict[str, list[float]]] = []
    for span in spans:
        token_slice = arr[span.token_start : min(span.token_end, arr.shape[0]), layers, :]
        if token_slice.size == 0:
            token_slice = arr[max(0, min(span.token_start, arr.shape[0] - 1)) : max(1, min(span.token_start + 1, arr.shape[0])), layers, :]
        if pooling == "last":
            pooled = token_slice[-1]
        elif pooling == "max":
            pooled = token_slice.max(axis=0)
        elif pooling == "attention":
            if attention_weights is None:
                raise ValueError("attention pooling requires attention_weights")
            w = attention_weights[span.token_start : span.token_start + token_slice.shape[0]]
            pooled = np.einsum("t,tlh->lh", w / max(w.sum(), 1e-12), token_slice)
        elif pooling == "mean":
            pooled = token_slice.mean(axis=0)
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        out.append({str(layer): pooled[j].astype(float).tolist() for j, layer in enumerate(layers)})
    return out


def steps_from_text(
    text: str,
    token_hidden_states: np.ndarray | None = None,
    layers: Sequence[int] | None = None,
    pooling: str = "mean",
    verifier_states: Sequence[VerifierState] | None = None,
) -> list[Step]:
    spans = parse_steps(text)
    hiddens = aggregate_step_hidden_states(token_hidden_states, spans, layers, pooling) if token_hidden_states is not None else [{} for _ in spans]
    states = list(verifier_states or [])
    steps = []
    for i, span in enumerate(spans):
        steps.append(
            Step(
                step_id=f"s{i + 1}",
                token_start=span.token_start,
                token_end=span.token_end,
                text=span.text,
                hidden_states=hiddens[i],
                verifier_state_optional=states[i] if i < len(states) else None,
                labels=span.labels,
            )
        )
    return steps
