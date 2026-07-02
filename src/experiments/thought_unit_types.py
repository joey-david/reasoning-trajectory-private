"""Shared sentence-lattice trace contracts and objective names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.analysis.step_classification.segmentation import StepSegment
from src.experiments.symbolic import SymbolicUpdate

OBJECTIVES = ("answer", "object", "correctness", "compression")
ORACLE_NAMES = {objective: f"oracle_{objective}" for objective in OBJECTIVES}
PRIMARY_FRACTION = 0.2


@dataclass(slots=True)
class TraceSpec:
    """Hold one generation and its fixed token-aligned sentence lattice."""

    row: dict[str, Any]
    segments: list[StepSegment]
    updates: list[SymbolicUpdate]
    train: bool


@dataclass(slots=True)
class TraceView:
    """Expose compact per-sentence features for one cached trajectory."""

    sample_id: str
    seed: int
    is_correct: bool
    train: bool
    raw: np.ndarray
    pca: np.ndarray
    h4: np.ndarray
    gram: np.ndarray
    raw_geometry: np.ndarray
    answer_score: np.ndarray
    update_count: np.ndarray
    token_count: np.ndarray
