"""Sentence-level latent features and answer/object target construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
import torch

from src.analysis.step_classification.segmentation import StepSegment
from src.experiments.sentence_lattice import boundary_jaccard, top_boundaries
from src.experiments.symbolic import SymbolicUpdate
from src.experiments.thought_unit_types import PRIMARY_FRACTION, TraceSpec
from src.runtime.artifact_store import load_hidden_states_npz


def question_split(rows: list[dict[str, Any]]) -> set[str]:
    """Choose deterministic train questions without splitting their rollouts.

    Args:
        rows: Generation or analysis records to process.

    Returns:
        The resulting unique values.
    """
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    unique = np.asarray(sorted(set(sample_ids)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train, _test = next(splitter.split(unique, groups=unique))
    return set(unique[train])


def fit_sentence_pca(
    run_path: Path,
    specs: list[TraceSpec],
    *,
    pca_dim: int,
) -> PCA:
    """Fit a bounded training-only PCA sample over sentence means.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        specs: Selected trace specifications used to fit PCA.
        pca_dim: Maximum PCA output dimension.

    Returns:
        A PCA fitted to the sampled sentence activations.
    """
    sampled: list[np.ndarray] = []
    for spec in specs:
        if not spec.train:
            continue
        states, layers = load_hidden_states_npz(
            run_path / spec.row["hidden_states_file"]
        )
        means, _counts = sentence_means(
            states[:, layers.index(-1)].astype(np.float32),
            spec.segments,
        )
        indices = np.linspace(
            0,
            len(means) - 1,
            min(len(means), 24),
            dtype=int,
        )
        sampled.append(means[indices])
    fit_matrix = np.concatenate(sampled).astype(np.float32)
    dimension = min(pca_dim, len(fit_matrix) - 1, fit_matrix.shape[1])
    model = PCA(
        n_components=dimension,
        whiten=True,
        svd_solver="randomized",
        random_state=42,
    )
    model.fit(fit_matrix)
    return model


def load_h4_projection(
    run_path: Path,
    *,
    projection_path: Path | None,
) -> tuple[np.ndarray, Path]:
    """Load the existing operation-supervised H4 projection matrix.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        projection_path: Path to a saved projection artifact.

    Returns:
        The computed aligned values described above.
    """
    path = projection_path or (
        run_path
        / "analysis"
        / "experiments"
        / "h4_structural_contrast"
        / "layer-1_projection.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weight = payload["weight"].detach().cpu().numpy().astype(np.float32)
    return weight, path


def sentence_means(
    states: np.ndarray,
    segments: list[StepSegment],
) -> tuple[np.ndarray, np.ndarray]:
    """Average last-layer token states inside every sentence span.

    Args:
        states: Token-aligned hidden-state vectors.
        segments: Contiguous sentence or token segments.

    Returns:
        The computed aligned values described above.
    """
    means: list[np.ndarray] = []
    counts: list[int] = []
    for segment in segments:
        start = min(max(segment.token_start, 0), len(states) - 1)
        end = min(max(segment.token_end, start), len(states) - 1)
        values = states[start : end + 1]
        means.append(values.mean(axis=0))
        counts.append(len(values))
    return np.stack(means).astype(np.float32), np.asarray(counts, dtype=np.int32)


def accumulated_gram_spectra(
    states: np.ndarray,
    segments: list[StepSegment],
    pca: PCA,
    *,
    dimension: int,
) -> np.ndarray:
    """Compute Yu-style accumulated Gram spectra on the sentence lattice.

    Args:
        states: Token-aligned hidden-state vectors.
        segments: Contiguous sentence or token segments.
        pca: Fitted PCA transform, or ``None`` when no reduction is needed.
        dimension: Maximum number of spectrum dimensions to retain.

    Returns:
        The resulting numeric array or tensor.
    """
    dimension = min(dimension, pca.n_components_)
    scale = np.sqrt(np.maximum(pca.explained_variance_[:dimension], 1e-8))
    projected = ((states - pca.mean_) @ pca.components_[:dimension].T) / scale
    accumulated = np.zeros((dimension, dimension), dtype=np.float64)
    spectra: list[np.ndarray] = []
    for segment in segments:
        start = min(max(segment.token_start, 0), len(projected) - 1)
        end = min(max(segment.token_end, start), len(projected) - 1)
        values = projected[start : end + 1].astype(np.float64)
        accumulated += values.T @ values / max(len(values), 1)
        eigenvalues = np.linalg.eigvalsh(accumulated)[::-1]
        spectra.append(np.log1p(np.maximum(eigenvalues, 0.0)))
    return np.stack(spectra).astype(np.float32)


def raw_geometry(means: np.ndarray) -> np.ndarray:
    """Summarize raw-space magnitude, direction change, and curvature.

    Args:
        means: Sentence-mean activation vectors.

    Returns:
        The resulting numeric array or tensor.
    """
    count = len(means)
    output = np.zeros((count, 4), dtype=np.float32)
    output[:, 0] = np.linalg.norm(means, axis=1)
    if count < 2:
        return output
    deltas = means[1:] - means[:-1]
    output[1:, 1] = np.linalg.norm(deltas, axis=1)
    output[1:, 2] = cosine_distance(means[:-1], means[1:])
    if count > 2:
        output[2:, 3] = cosine_distance(deltas[:-1], deltas[1:])
    return output


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine distance with stable zero-norm handling.

    Args:
        left: Left operand or comparison input.
        right: Right operand or comparison input.

    Returns:
        The resulting numeric array or tensor.
    """
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.einsum("ij,ij->i", left, right) / np.maximum(denominator, 1e-8)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows while preserving all-zero vectors.

    Args:
        values: Values to summarize or transform.

    Returns:
        The resulting numeric array or tensor.
    """
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def linear_hsic_alignment(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute normalized linear-kernel HSIC against one target vector.

    Args:
        values: Values to summarize or transform.
        target: Target value or index.

    Returns:
        The resulting numeric array or tensor.
    """
    centered = values - values.mean(axis=1, keepdims=True)
    target_centered = target - target.mean()
    numerator = np.square(centered @ target_centered)
    denominator = np.einsum("ij,ij->i", centered, centered) * float(
        target_centered @ target_centered
    )
    return numerator / np.maximum(denominator, 1e-12)


def cross_rollout_answer_scores(
    raw_rows: list[np.ndarray],
    records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Pair each trace with a correct same-question terminal answer state.

    Args:
        raw_rows: Source generation rows aligned with cached traces.
        records: Aligned records to analyze or annotate.

    Returns:
        The computed aligned values described above.
    """
    by_question: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_question[str(record["sample_id"])].append(index)
    scores: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    for index, (means, record) in enumerate(zip(raw_rows, records)):
        candidates = [
            candidate
            for candidate in by_question[str(record["sample_id"])]
            if candidate != index and records[candidate]["is_correct"]
        ]
        if candidates:
            donor = candidates[index % len(candidates)]
            counts["cross_rollout_correct"] += 1
        else:
            donor = index
            counts["self_fallback_no_correct_donor"] += 1
        donor_answer = int(records[donor]["answer_sentence"])
        target = raw_rows[donor][donor_answer].astype(np.float32)
        scores.append(
            linear_hsic_alignment(means.astype(np.float32), target).astype(np.float32)
        )
        record["answer_target_seed"] = int(records[donor]["seed"])
        record["answer_target_is_self"] = donor == index
    return scores, dict(counts)


def compare_answer_curves(
    offsets: np.ndarray,
    proxy: np.ndarray,
    gold: np.ndarray,
    selected_indices: list[int],
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Compare cross-rollout and gold-solution answer curves on held-out traces.

    Args:
        offsets: Trace start offsets in the flattened sentence arrays.
        proxy: Proxy answer-information scores.
        gold: Gold-answer similarity scores.
        selected_indices: Indices of traces selected for evaluation.
        records: Aligned records to analyze or annotate.

    Returns:
        The resulting keyed records or metrics.
    """
    correlations: list[float] = []
    jaccards: list[float] = []
    for index in selected_indices:
        if bool(records[index]["train"]):
            continue
        start, end = offsets[index : index + 2]
        proxy_curve = proxy[int(start) : int(end)]
        gold_curve = gold[int(start) : int(end)]
        if np.std(proxy_curve) > 0 and np.std(gold_curve) > 0:
            correlation = spearmanr(proxy_curve, gold_curve).statistic
            if np.isfinite(correlation):
                correlations.append(float(correlation))
        boundary_count = max(int(round((len(proxy_curve) - 1) * PRIMARY_FRACTION)), 1)
        proxy_boundaries = top_boundaries(
            np.abs(np.diff(proxy_curve)),
            boundary_count,
        )
        gold_boundaries = top_boundaries(
            np.abs(np.diff(gold_curve)),
            boundary_count,
        )
        jaccards.append(boundary_jaccard(proxy_boundaries, gold_boundaries))
    return {
        "held_out_curve_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "held_out_top_boundary_jaccard": float(np.mean(jaccards)),
    }


def terminal_answer_sentence(spec: TraceSpec) -> int:
    """Locate the sentence containing the terminal extracted answer.

    Args:
        spec: Cached trace specification.

    Returns:
        The computed index, count, or status code.
    """
    extracts = [update for update in spec.updates if update.operator == "EXTRACT"]
    target_token = extracts[-1].token_end if extracts else None
    if target_token is not None:
        for index, segment in enumerate(spec.segments):
            if segment.token_start <= target_token <= segment.token_end:
                return index
    for index in range(len(spec.segments) - 1, -1, -1):
        text = spec.segments[index].text.lower()
        if "answer:" in text or "final answer" in text:
            return index
    return len(spec.segments) - 1


def sentence_update_counts(
    segments: list[StepSegment],
    updates: list[SymbolicUpdate],
) -> np.ndarray:
    """Count symbolic update completions inside each sentence.

    Args:
        segments: Contiguous sentence or token segments.
        updates: Symbolic solution-object updates.

    Returns:
        The resulting numeric array or tensor.
    """
    ends = np.asarray([segment.token_end for segment in segments])
    counts = np.zeros(len(segments), dtype=np.int32)
    for update in updates:
        sentence = int(np.searchsorted(ends, update.token_end, side="left"))
        if sentence < len(counts):
            counts[sentence] += 1
    return counts
