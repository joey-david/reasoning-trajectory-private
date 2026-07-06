"""Shared rollout selection, latent statistics, and grouped evaluation helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_trajectory.artifacts import read_generation_rows


@dataclass(slots=True)
class IntervalDynamics:
    """Summarize how an activation path evolves across one token interval."""

    token_count: int
    integrated_vector_norm: float
    mean_vector_norm: float
    path_length: float
    mean_derivative_magnitude: float
    net_displacement: float
    net_to_path_ratio: float
    cumulative_state_cosine_distance: float
    cumulative_derivative_cosine_distance: float
    peak_share: float
    effective_width_tokens: float
    effective_width_fraction: float
    temporal_centroid: float
    net_vector: np.ndarray

    def scalar_record(self) -> dict[str, float | int]:
        """Serialize scalar interval-dynamics metrics for reporting.

        Args:
            None.

        Returns:
            The resulting keyed records or metrics.
        """
        return {
            "interval_tokens": self.token_count,
            "integrated_vector_norm": self.integrated_vector_norm,
            "mean_vector_norm": self.mean_vector_norm,
            "path_length": self.path_length,
            "mean_derivative_magnitude": self.mean_derivative_magnitude,
            "net_displacement": self.net_displacement,
            "net_to_path_ratio": self.net_to_path_ratio,
            "cumulative_state_cosine_distance": (self.cumulative_state_cosine_distance),
            "cumulative_derivative_cosine_distance": (
                self.cumulative_derivative_cosine_distance
            ),
            "peak_share": self.peak_share,
            "effective_width_tokens": self.effective_width_tokens,
            "effective_width_fraction": self.effective_width_fraction,
            "temporal_centroid": self.temporal_centroid,
        }


def balanced_generation_rows(
    run_path: Path,
    *,
    per_sample: int = 10,
    require_hidden_states: bool = True,
    require_scored: bool = True,
) -> list[dict[str, Any]]:
    """Select a deterministic, uniformly capped set without changing raw data.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        per_sample: Maximum number of trajectories retained per sample.
        require_hidden_states: Whether rows without activation artifacts are excluded.
        require_scored: Whether rows lacking correctness labels are excluded.

    Returns:
        The resulting ordered records or values.
    """
    selected: list[dict[str, Any]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    rows = sorted(
        read_generation_rows(run_path),
        key=lambda row: (str(row.get("sample_id")), int(row.get("seed", 0))),
    )
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if counts[sample_id] >= per_sample:
            continue
        if require_hidden_states and not row.get("hidden_states_file"):
            continue
        if require_scored and row.get("is_correct") is None:
            continue
        selected.append(row)
        counts[sample_id] += 1
    return selected


def latent_deltas(states: np.ndarray) -> np.ndarray:
    """Return token-to-token state changes with a zero vector at token zero.

    Args:
        states: Token-aligned hidden-state vectors.

    Returns:
        The resulting numeric array or tensor.
    """
    states = np.asarray(states, dtype=np.float32)
    deltas = np.zeros_like(states)
    if len(states) > 1:
        deltas[1:] = states[1:] - states[:-1]
    return deltas


def robust_spike_indices(
    magnitudes: np.ndarray,
    *,
    z_threshold: float = 3.0,
    min_distance: int = 3,
) -> np.ndarray:
    """Find separated local maxima above a median absolute-deviation threshold.

    Args:
        magnitudes: Per-token activation-change magnitudes.
        z_threshold: Robust z-score threshold for selecting changes.
        min_distance: Minimum token separation between retained peaks.

    Returns:
        The resulting numeric array or tensor.
    """
    values = np.asarray(magnitudes, dtype=np.float32)
    if len(values) < 3:
        return np.empty(0, dtype=np.int32)
    core = values[1:]
    median = float(np.median(core))
    mad = float(np.median(np.abs(core - median)))
    scale = max(1.4826 * mad, float(np.std(core)) * 0.1, 1e-8)
    threshold = median + z_threshold * scale
    candidates = np.flatnonzero(
        (values >= threshold)
        & (values >= np.r_[values[0], values[:-1]])
        & (values >= np.r_[values[1:], values[-1]])
    )
    if not len(candidates):
        return candidates.astype(np.int32)

    retained: list[int] = []
    for idx in candidates[np.argsort(values[candidates])[::-1]]:
        if all(abs(int(idx) - existing) >= min_distance for existing in retained):
            retained.append(int(idx))
    return np.asarray(sorted(retained), dtype=np.int32)


def nearest_distance(indices: np.ndarray, target: int) -> int | None:
    """Return the distance from ``target`` to the nearest index.

    Args:
        indices: Token or record indices to process.
        target: Target value or index.

    Returns:
        The computed index, count, or status code.
    """
    if not len(indices):
        return None
    return int(np.min(np.abs(indices - int(target))))


def percentile_rank(values: np.ndarray, value: float) -> float:
    """Return the empirical percentile of a value in ``values``.

    Args:
        values: Values to summarize or transform.
        value: Value to rank, parse, or transform.

    Returns:
        The computed scalar metric.
    """
    values = np.asarray(values)
    if not len(values):
        return float("nan")
    return float(np.mean(values <= value))


def prefix_checkpoints(length: int) -> dict[int, int]:
    """Map 25/50/75 percent checkpoints to valid zero-based token indices.

    Args:
        length: Sequence length.

    Returns:
        The resulting keyed records or metrics.
    """
    return {
        percent: min(max(int(np.ceil(length * percent / 100)) - 1, 0), length - 1)
        for percent in (25, 50, 75)
    }


def update_phase_bounds(
    token_start: int,
    token_end: int,
    state_count: int,
) -> tuple[int, int]:
    """Map a textual token interval to pre-update and completed-state indices.

        Stored state ``t`` predicts generated token ``t``. A textual interval
        ending at token ``token_end`` is therefore fully represented at state
        ``token_end + 1``.

    Args:
        token_start: Inclusive first token index.
        token_end: Inclusive final token index.
        state_count: Number of stored token states.

    Returns:
        The computed aligned values described above.
    """
    if state_count < 2:
        raise ValueError("Interval dynamics require at least two states")
    start = min(max(int(token_start), 0), state_count - 2)
    end = min(max(int(token_end) + 1, start + 1), state_count - 1)
    return start, end


def interval_dynamics(
    states: np.ndarray,
    start: int,
    end: int,
) -> IntervalDynamics:
    """Integrate activation movement from ``start`` through completed ``end``.

    Args:
        states: Token-aligned hidden-state vectors.
        start: Inclusive start index.
        end: Inclusive end index.

    Returns:
        Integrated dynamics for the requested activation interval.
    """
    values = np.asarray(states, dtype=np.float32)
    if not 0 <= start < end < len(values):
        raise ValueError(f"Invalid interval [{start}, {end}] for {len(values)} states")
    step_deltas = values[start + 1 : end + 1] - values[start:end]
    interval_values = values[start + 1 : end + 1]
    vector_norms = np.linalg.norm(interval_values, axis=1)
    magnitudes = np.linalg.norm(step_deltas, axis=1)
    path_length = float(magnitudes.sum())
    net_vector = values[end] - values[start]
    net_displacement = float(np.linalg.norm(net_vector))
    squared_mass = float(np.square(magnitudes).sum())
    width = path_length * path_length / squared_mass if squared_mass > 0.0 else 0.0
    weights = (
        magnitudes / path_length if path_length > 0.0 else np.zeros_like(magnitudes)
    )
    positions = np.linspace(0.0, 1.0, len(magnitudes), dtype=np.float32)
    return IntervalDynamics(
        token_count=len(magnitudes),
        integrated_vector_norm=float(vector_norms.sum()),
        mean_vector_norm=float(vector_norms.mean()),
        path_length=path_length,
        mean_derivative_magnitude=float(magnitudes.mean()),
        net_displacement=net_displacement,
        net_to_path_ratio=net_displacement / max(path_length, 1e-8),
        cumulative_state_cosine_distance=cumulative_cosine_distance(
            values[start : end + 1]
        ),
        cumulative_derivative_cosine_distance=cumulative_cosine_distance(step_deltas),
        peak_share=float(magnitudes.max() / max(path_length, 1e-8)),
        effective_width_tokens=width,
        effective_width_fraction=width / len(magnitudes),
        temporal_centroid=float(np.sum(weights * positions)),
        net_vector=net_vector,
    )


def cumulative_cosine_distance(vectors: np.ndarray) -> float:
    """Sum adjacent cosine distances over a vector sequence.

    Args:
        vectors: Feature or activation vectors to process.

    Returns:
        The computed scalar metric.
    """
    values = np.asarray(vectors, dtype=np.float32)
    if len(values) < 2:
        return 0.0
    left = values[:-1]
    right = values[1:]
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-8)
    return float(np.sum(1.0 - np.clip(cosine, -1.0, 1.0)))


def matched_control_dynamics(
    states: np.ndarray,
    *,
    duration: int,
    excluded: list[tuple[int, int]],
    max_windows: int = 31,
) -> list[IntervalDynamics]:
    """Sample same-duration non-update windows as a conservative null.

    Args:
        states: Token-aligned hidden-state vectors.
        duration: Window duration in token transitions.
        excluded: Intervals that control windows must not overlap.
        max_windows: Maximum number of matched control windows.

    Returns:
        The resulting ordered records or values.
    """
    last_start = len(states) - duration - 1
    if duration < 1 or last_start < 0:
        return []
    candidates = [
        start
        for start in range(last_start + 1)
        if not any(
            start < excluded_end and start + duration > excluded_start
            for excluded_start, excluded_end in excluded
        )
    ]
    if not candidates:
        return []
    if len(candidates) > max_windows:
        indices = np.linspace(0, len(candidates) - 1, max_windows, dtype=int)
        candidates = [candidates[int(index)] for index in indices]
    return [interval_dynamics(states, start, start + duration) for start in candidates]


def control_percentile(
    value: float,
    controls: list[IntervalDynamics],
    field: str,
) -> float | None:
    """Rank an interval metric against its same-length control windows.

    Args:
        value: Value to rank, parse, or transform.
        controls: Matched control interval summaries.
        field: Record field to read or summarize.

    Returns:
        The computed scalar metric, or ``None`` when unavailable.
    """
    if not controls:
        return None
    control_values = np.asarray([getattr(control, field) for control in controls])
    return float(np.mean(control_values <= value))
