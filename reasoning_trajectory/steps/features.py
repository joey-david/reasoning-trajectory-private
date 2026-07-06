"""Derive latent mean, direction, nudge, and scalar features for segmented reasoning steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from reasoning_trajectory.steps.segmentation import StepSegment


@dataclass(slots=True)
class StepFeature:
    """Pair one step's serializable metadata with its full latent feature vectors."""

    record: dict[str, Any]
    mean: np.ndarray
    direction: np.ndarray
    nudge: np.ndarray


@dataclass(slots=True)
class StepMatrices:
    """Hold aligned matrices of step mean, direction, and previous-step nudge vectors."""

    means: np.ndarray
    directions: np.ndarray
    nudges: np.ndarray


def build_step_features(
    *,
    states: np.ndarray,
    layer: int,
    layer_col: int,
    row: dict[str, Any],
    segments: list[StepSegment],
) -> list[StepFeature]:
    """Compute latent features for one row, layer, and segmentation.

    Args:
        states: Hidden states shaped ``[tokens, layers, hidden]``.
        layer: Decoder-layer ID written into output metadata.
        layer_col: Column of ``states`` corresponding to ``layer``.
        row: Source generation row.
        segments: Token-aligned steps to featurize in order.

    Returns:
        Step features with metadata and full latent vectors.
    """
    out: list[StepFeature] = []
    previous_mean: np.ndarray | None = None
    total_tokens = max(len(row.get("generated_token_ids", [])), 1)

    for segment in segments:
        token_start = min(max(segment.token_start, 0), states.shape[0] - 1)
        token_end = min(max(segment.token_end, token_start), states.shape[0] - 1)
        segment_states = states[token_start : token_end + 1, layer_col].astype(
            np.float32
        )
        if segment_states.size == 0:
            continue

        mean = segment_states.mean(axis=0)
        variance = float(np.var(segment_states, axis=0).mean())
        direction = segment_states[-1] - segment_states[0]
        nudge = np.zeros_like(mean) if previous_mean is None else mean - previous_mean
        previous_mean = mean

        record = {
            "sample_id": row.get("sample_id"),
            "seed": row.get("seed"),
            "trajectory_id": f"{row.get('sample_id')}::{row.get('seed')}",
            "layer": layer,
            "segmenter": segment.segmenter,
            "selector": segment.segmenter,
            "step_idx": segment.step_idx,
            "token_start": token_start,
            "token_end": token_end,
            "token_idx": round((token_start + token_end) / 2),
            "token_fraction": ((token_start + token_end) / 2)
            / max(total_tokens - 1, 1),
            "token_count": token_end - token_start + 1,
            "char_start": segment.char_start,
            "char_end": segment.char_end,
            "step_text": segment.text[:600],
            "labels": list(segment.labels),
            "variance": round(variance, 6),
            "direction_norm": round(float(np.linalg.norm(direction)), 6),
            "nudge_norm": round(float(np.linalg.norm(nudge)), 6),
            "is_correct": row.get("is_correct"),
            "produced_answer": row.get("produced_answer"),
            "reasoning_length": row.get("reasoning_length"),
        }
        out.append(
            StepFeature(record=record, mean=mean, direction=direction, nudge=nudge)
        )

    return out


def stack_features(features: list[StepFeature]) -> StepMatrices:
    """Stack aligned step feature vectors into dense matrices.

    Args:
        features: Non-empty ordered step features.

    Returns:
        Float32 mean, direction, and nudge matrices.
    """
    return StepMatrices(
        means=np.stack([item.mean for item in features]).astype(np.float32),
        directions=np.stack([item.direction for item in features]).astype(np.float32),
        nudges=np.stack([item.nudge for item in features]).astype(np.float32),
    )
