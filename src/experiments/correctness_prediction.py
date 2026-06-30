"""Compare prefix-level correctness probes across reasoning segmentations."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.step_classification.segmentation import build_segments
from src.analysis.token_alignment import build_token_spans
from src.experiments.common import (
    balanced_generation_rows,
    latent_deltas,
    prefix_checkpoints,
    robust_spike_indices,
)
from src.runtime.artifact_store import load_hidden_states_npz


REPRESENTATIONS = (
    "token_state",
    "sentence_mean",
    "step_mean",
    "step_mean_variance",
    "step_direction",
    "latent_segments",
)


def run_correctness_prediction(
    run_path: Path,
    *,
    per_sample: int = 10,
    folds: int = 5,
) -> Path:
    """Run H5 with question-disjoint out-of-fold predictions."""
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    token_spans = build_token_spans(run_path, rows)
    features: defaultdict[tuple[int, int, str], list[np.ndarray]] = defaultdict(list)
    labels: list[int] = []
    groups: list[str] = []
    identities: list[str] = []
    layers_seen: set[int] = set()

    for row, spans in zip(rows, token_spans):
        states, layers = load_hidden_states_npz(run_path / row["hidden_states_file"])
        count = min(len(row.get("generated_token_ids", [])), states.shape[0])
        if count < 4:
            continue
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        checkpoints = prefix_checkpoints(count)
        for layer_col, layer in enumerate(layers):
            layers_seen.add(layer)
            layer_states = states[:count, layer_col].astype(np.float32)
            for percent, checkpoint in checkpoints.items():
                representation_features = prefix_representations(
                    layer_states, segments, checkpoint
                )
                for name, vector in representation_features.items():
                    features[(layer, percent, name)].append(vector)
        labels.append(int(bool(row["is_correct"])))
        groups.append(str(row["sample_id"]))
        identities.append(f"{row['sample_id']}::{row['seed']}")

    y = np.asarray(labels, dtype=np.int8)
    group_array = np.asarray(groups)
    out_dir = run_path / "analysis" / "experiments" / "h5_correctness_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    feature_archive: dict[str, np.ndarray] = {}

    effective_folds = min(folds, len(set(groups)))
    for layer in sorted(layers_seen):
        for percent in (25, 50, 75):
            for name in REPRESENTATIONS:
                key = (layer, percent, name)
                if key not in features or len(features[key]) != len(y):
                    continue
                x = np.stack(features[key]).astype(np.float32)
                probabilities = grouped_oof_probabilities(
                    x, y, group_array, folds=effective_folds
                )
                archive_key = f"layer{layer}_{percent}_{name}"
                predictions[archive_key] = probabilities.astype(np.float32)
                feature_archive[archive_key] = x.astype(np.float16)
                results.append(
                    score_predictions(
                        y=y,
                        probabilities=probabilities,
                        groups=group_array,
                        layer=layer,
                        checkpoint=percent,
                        representation=name,
                        dimensions=x.shape[1],
                    )
                )

    np.savez_compressed(
        out_dir / "features.npz",
        labels=y,
        groups=group_array.astype(str),
        identities=np.asarray(identities, dtype=str),
        **feature_archive,
    )
    np.savez_compressed(
        out_dir / "predictions.npz",
        labels=y,
        groups=group_array.astype(str),
        **predictions,
    )
    comparisons = segmentation_comparisons(
        y=y,
        groups=group_array,
        predictions=predictions,
        layers=sorted(layers_seen),
    )
    report = {
        "hypothesis": "H5_correctness_prediction_by_segmentation",
        "source_run": run_path.as_posix(),
        "selection": {
            "trajectories": len(y),
            "questions": len(set(groups)),
            "per_sample_cap": per_sample,
            "class_counts": dict(Counter(labels)),
        },
        "evaluation": {
            "split": "GroupKFold by sample_id",
            "folds": effective_folds,
            "probe": "StandardScaler + L2 logistic regression",
        },
        "results": results,
        "comparisons": comparisons,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def prefix_representations(
    states: np.ndarray,
    segments: list[Any],
    checkpoint: int,
) -> dict[str, np.ndarray]:
    """Build fixed-width representations from one partial trace."""
    checkpoint = min(max(int(checkpoint), 1), len(states) - 1)
    prefix = states[: checkpoint + 1]
    sentence_means: list[np.ndarray] = []
    for segment in segments:
        if segment.token_start > checkpoint:
            break
        start = min(max(segment.token_start, 0), checkpoint)
        end = min(max(segment.token_end, start), checkpoint)
        sentence_means.append(states[start : end + 1].mean(axis=0))
    if not sentence_means:
        sentence_means = [prefix.mean(axis=0)]
    step_matrix = np.stack(sentence_means)
    step_mean = step_matrix.mean(axis=0)
    step_variance = step_matrix.var(axis=0)
    if len(step_matrix) > 1:
        step_direction = np.diff(step_matrix, axis=0).mean(axis=0)
    else:
        step_direction = np.zeros(states.shape[1], dtype=np.float32)

    deltas = latent_deltas(prefix)
    magnitudes = np.linalg.norm(deltas, axis=1)
    spikes = robust_spike_indices(magnitudes)
    update_vectors = deltas[spikes] if len(spikes) else deltas[[-1]]
    latent_mean = update_vectors.mean(axis=0)
    latent_variance = update_vectors.var(axis=0)

    return {
        "token_state": prefix[-1],
        "sentence_mean": sentence_means[-1],
        "step_mean": step_mean,
        "step_mean_variance": np.concatenate([step_mean, step_variance]),
        "step_direction": step_direction,
        "latent_segments": np.concatenate([latent_mean, latent_variance]),
    }


def grouped_oof_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
) -> np.ndarray:
    """Fit identical L2 probes and return question-disjoint predictions."""
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                solver="liblinear",
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        )
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Grouped probe did not produce every out-of-fold prediction")
    return probabilities


def score_predictions(
    *,
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    layer: int,
    checkpoint: int,
    representation: str,
    dimensions: int,
) -> dict[str, Any]:
    predicted = probabilities >= 0.5
    observed, fraction = calibration_curve(
        y, probabilities, n_bins=8, strategy="quantile"
    )
    auc = float(roc_auc_score(y, probabilities))
    auc_interval = grouped_bootstrap_auc(y, probabilities, groups)
    return {
        "layer": layer,
        "checkpoint_percent": checkpoint,
        "representation": representation,
        "dimensions": dimensions,
        "roc_auc": auc,
        "roc_auc_95ci": auc_interval,
        "accuracy": float(accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities)),
        "calibration": {
            "mean_predicted": fraction.tolist(),
            "fraction_positive": observed.tolist(),
        },
    }


def segmentation_comparisons(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    predictions: dict[str, np.ndarray],
    layers: list[int],
) -> list[dict[str, Any]]:
    """Compare segmentation probes against the sentence baseline by group."""
    comparisons: list[dict[str, Any]] = []
    for layer in layers:
        for checkpoint in (25, 50, 75):
            baseline_key = f"layer{layer}_{checkpoint}_sentence_mean"
            if baseline_key not in predictions:
                continue
            for representation in ("step_mean_variance", "latent_segments"):
                contender_key = f"layer{layer}_{checkpoint}_{representation}"
                if contender_key not in predictions:
                    continue
                point, interval = grouped_bootstrap_auc_difference(
                    y,
                    predictions[contender_key],
                    predictions[baseline_key],
                    groups,
                )
                comparisons.append(
                    {
                        "layer": layer,
                        "checkpoint_percent": checkpoint,
                        "representation": representation,
                        "baseline": "sentence_mean",
                        "roc_auc_difference": point,
                        "roc_auc_difference_95ci": interval,
                    }
                )
    return comparisons


def grouped_bootstrap_auc(
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 500,
) -> list[float]:
    values = grouped_bootstrap_values(
        y,
        groups,
        lambda indices: roc_auc_score(y[indices], probabilities[indices]),
        draws=draws,
    )
    return np.quantile(values, [0.025, 0.975]).tolist()


def grouped_bootstrap_auc_difference(
    y: np.ndarray,
    contender: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int = 500,
) -> tuple[float, list[float]]:
    point = float(roc_auc_score(y, contender) - roc_auc_score(y, baseline))
    values = grouped_bootstrap_values(
        y,
        groups,
        lambda indices: (
            roc_auc_score(y[indices], contender[indices])
            - roc_auc_score(y[indices], baseline[indices])
        ),
        draws=draws,
    )
    return point, np.quantile(values, [0.025, 0.975]).tolist()


def grouped_bootstrap_values(
    y: np.ndarray,
    groups: np.ndarray,
    scorer: Any,
    *,
    draws: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(42)
    values: list[float] = []
    for _ in range(draws):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([by_group[group] for group in sampled])
        if len(np.unique(y[indices])) < 2:
            continue
        values.append(float(scorer(indices)))
    if not values:
        raise ValueError("Grouped bootstrap produced no valid draws")
    return np.asarray(values)
