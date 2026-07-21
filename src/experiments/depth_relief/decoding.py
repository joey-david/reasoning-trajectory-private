"""Small, calibrated decoders for finite-state representation assays."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci


def label_entropy(labels: np.ndarray) -> float:
    """Return empirical base-two entropy for a finite label vector."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def conditional_label_entropy(
    labels: np.ndarray, conditions: np.ndarray
) -> float:
    """Return empirical H(labels | conditions) in bits."""
    targets = np.asarray(labels)
    given = np.asarray(conditions)
    if len(targets) != len(given) or not len(targets):
        raise ValueError("Conditional entropy arrays must be nonempty and aligned")
    total = 0.0
    for condition in np.unique(given):
        selected = targets[given == condition]
        total += len(selected) / len(targets) * label_entropy(selected)
    return float(total)


def fit_centroid_decoder(
    activations: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    rank: int,
) -> dict[str, np.ndarray]:
    """Fit a low-rank nearest-centroid decoder without hyperparameter search."""
    values = np.asarray(activations, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(targets):
        raise ValueError("Decoder activations and labels are not row-aligned")
    missing = sorted(set(range(class_count)) - set(map(int, targets)))
    if missing:
        raise ValueError(f"Decoder training split misses states: {missing}")
    mean = values.mean(axis=0)
    raw_centroids = np.stack(
        [values[targets == label].mean(axis=0) for label in range(class_count)]
    )
    _, _, vt = np.linalg.svd(
        raw_centroids - raw_centroids.mean(axis=0), full_matrices=False
    )
    fitted_rank = min(rank, class_count - 1, values.shape[1])
    basis = vt[:fitted_rank].T.astype(np.float32)
    projected = (values - mean) @ basis
    scale = projected.std(axis=0)
    scale[scale < 1e-6] = 1.0
    centroids = np.stack(
        [projected[targets == label].mean(axis=0) for label in range(class_count)]
    )
    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "scale": scale.astype(np.float32),
        "centroids": centroids.astype(np.float32),
    }


def decoder_logits(
    decoder: dict[str, np.ndarray], values: np.ndarray
) -> np.ndarray:
    """Return negative standardized centroid distances as class logits."""
    projected = (np.asarray(values, dtype=np.float32) - decoder["mean"]) @ decoder[
        "basis"
    ]
    projected = projected / decoder["scale"]
    centroids = decoder["centroids"] / decoder["scale"]
    return -((projected[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)


def predict_centroid(
    decoder: dict[str, np.ndarray], values: np.ndarray
) -> np.ndarray:
    """Predict the nearest calibrated centroid label."""
    return decoder_logits(decoder, values).argmax(axis=1)


def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    temperatures: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> float:
    """Select a fixed-grid softmax temperature on validation log loss."""
    targets = np.asarray(labels, dtype=np.int64)

    def loss(temperature: float) -> float:
        scaled = logits / temperature
        log_normalizer = np.logaddexp.reduce(scaled, axis=1)
        return float(np.mean(log_normalizer - scaled[np.arange(len(targets)), targets]))

    return min(temperatures, key=lambda value: (loss(value), value))


def decoder_report(
    decoder: dict[str, np.ndarray],
    values: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    temperature: float,
    seed: int,
    clusters: np.ndarray | None = None,
    entropy_bits: float | None = None,
) -> dict[str, Any]:
    """Report held-out accuracy and a variational information lower bound."""
    targets = np.asarray(labels, dtype=np.int64)
    logits = decoder_logits(decoder, values) / temperature
    log_normalizer = np.logaddexp.reduce(logits, axis=1)
    log_probabilities = logits - log_normalizer[:, None]
    entropy = label_entropy(targets) if entropy_bits is None else entropy_bits
    information = entropy + (
        log_probabilities[np.arange(len(targets)), targets] / np.log(2)
    )
    interval = bootstrap_mean_ci if clusters is None else cluster_bootstrap_mean_ci
    interval_kwargs = {} if clusters is None else {"clusters": clusters}
    return {
        "n": len(targets),
        "accuracy": interval(
            logits.argmax(axis=1) == targets,
            seed=seed,
            **interval_kwargs,
        ),
        "information_lower_bound_bits": interval(
            information,
            seed=seed + 1,
            **interval_kwargs,
        ),
        "temperature": float(temperature),
        "label_entropy_bits": float(entropy),
    }


def decoder_point(
    decoder: dict[str, np.ndarray],
    values: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    temperature: float,
    entropy_bits: float | None = None,
) -> dict[str, float]:
    """Return compact accuracy and information values for dense layer maps."""
    targets = np.asarray(labels, dtype=np.int64)
    logits = decoder_logits(decoder, values) / temperature
    log_normalizer = np.logaddexp.reduce(logits, axis=1)
    entropy = label_entropy(targets) if entropy_bits is None else entropy_bits
    information = entropy + np.mean(
        (logits[np.arange(len(targets)), targets] - log_normalizer) / np.log(2)
    )
    return {
        "accuracy": float(np.mean(logits.argmax(axis=1) == targets)),
        "information_lower_bound_bits": float(information),
        "label_entropy_bits": float(entropy),
    }
