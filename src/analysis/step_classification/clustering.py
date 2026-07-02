"""Cluster latent reasoning-step features and summarize representative text examples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.step_classification.features import StepMatrices


@dataclass(slots=True)
class ClusterModel:
    """Hold fitted transforms and K-means state for step assignment."""

    mean_pca: PCA
    direction_pca: PCA | None
    nudge_pca: PCA | None
    scaler: StandardScaler
    kmeans: KMeans


def fit_cluster_model(
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    cfg: dict[str, Any],
) -> ClusterModel | None:
    """Fit bounded latent transforms and K-means state.

    Args:
        records: Step metadata used to fit scalar normalization.
        vectors: Mean, direction, and nudge matrices used for fitting.
        cfg: Step-classification PCA, cluster-count, and random-seed options.

    Returns:
        Fitted cluster state, or ``None`` with fewer than three records.
    """
    if len(records) < 3:
        return None

    step_cfg = cfg.get("step_classification", {})
    k = min(int(step_cfg.get("cluster_k", 8)), len(records) - 1)
    pca_dim = min(
        int(step_cfg.get("cluster_pca_dim", 32)),
        len(records) - 1,
        vectors.means.shape[1],
    )
    random_state = int(step_cfg.get("random_state", 42))

    mean_pca = PCA(n_components=pca_dim, random_state=random_state).fit(
        vectors.means
    )
    direction_pca = fit_normalized_pca(
        vectors.directions, min(8, pca_dim, len(records) - 1), random_state
    )
    nudge_pca = fit_normalized_pca(
        vectors.nudges, min(8, pca_dim, len(records) - 1), random_state
    )
    scaler = StandardScaler()
    features = cluster_features(
        records,
        vectors,
        mean_pca,
        direction_pca,
        nudge_pca,
    )
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(scaled)
    return ClusterModel(
        mean_pca=mean_pca,
        direction_pca=direction_pca,
        nudge_pca=nudge_pca,
        scaler=scaler,
        kmeans=kmeans,
    )


def assign_clusters(
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    model: ClusterModel | None,
) -> None:
    """Assign fitted cluster IDs and distances to aligned step records.

    Args:
        records: Aligned records to analyze or annotate.
        vectors: Feature or activation vectors to process.
        model: Loaded model used for inference or transformation.

    Returns:
        None.
    """
    if model is None:
        return
    features = cluster_features(
        records,
        vectors,
        model.mean_pca,
        model.direction_pca,
        model.nudge_pca,
    )
    scaled = model.scaler.transform(features)
    labels = model.kmeans.predict(scaled)
    distances = model.kmeans.transform(scaled)[np.arange(len(labels)), labels]

    for rec, label, distance in zip(records, labels, distances):
        rec["cluster_id"] = int(label)
        rec["cluster_distance"] = round(float(distance), 6)


def cluster_metadata(
    records: list[dict[str, Any]],
    model: ClusterModel | None,
) -> dict[str, Any]:
    """Describe fitted feature dimensions and summarize assigned records.

    Args:
        records: Aligned records to analyze or annotate.
        model: Loaded model used for inference or transformation.

    Returns:
        The resulting keyed records or metrics.
    """
    if model is None:
        return {"clusters": [], "feature_columns": []}
    return {
        "cluster_count": int(model.kmeans.n_clusters),
        "feature_columns": {
            "mean_pca": model.mean_pca.n_components_,
            "direction_pca": pca_components(model.direction_pca),
            "nudge_pca": pca_components(model.nudge_pca),
            "scalars": [
                "variance",
                "direction_norm",
                "nudge_norm",
                "token_fraction",
                "token_count",
            ],
        },
        "clusters": cluster_summaries(records),
    }


def cluster_features(
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    mean_pca: PCA,
    direction_pca: PCA | None,
    nudge_pca: PCA | None,
) -> np.ndarray:
    """Transform aligned latent and scalar inputs into cluster features.

    Args:
        records: Aligned records to analyze or annotate.
        vectors: Feature or activation vectors to process.
        mean_pca: Optional PCA fitted to step-mean vectors.
        direction_pca: Optional PCA fitted to direction vectors.
        nudge_pca: Optional PCA fitted to nudge vectors.

    Returns:
        The resulting numeric array or tensor.
    """
    scalars = np.asarray(
        [
            [
                rec["variance"],
                rec["direction_norm"],
                rec["nudge_norm"],
                rec["token_fraction"],
                rec["token_count"],
            ]
            for rec in records
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            mean_pca.transform(vectors.means),
            transform_normalized(vectors.directions, direction_pca),
            transform_normalized(vectors.nudges, nudge_pca),
            scalars,
        ],
        axis=1,
    )


def fit_normalized_pca(
    x: np.ndarray,
    n_components: int,
    random_state: int,
) -> PCA | None:
    """Fit PCA after row-wise L2 normalization.

    Args:
        x: Two-dimensional sample-by-feature matrix.
        n_components: Requested output dimensions.
        random_state: PCA random seed.

    Returns:
        Fitted PCA, or ``None`` when no components are requested.
    """
    if n_components <= 0:
        return None
    return PCA(n_components=n_components, random_state=random_state).fit(
        normalized_rows(x)
    )


def transform_normalized(x: np.ndarray, pca: PCA | None) -> np.ndarray:
    """Normalize rows and apply an optional fitted PCA.

    Args:
        x: Input feature matrix.
        pca: Fitted PCA transform, or ``None`` when no reduction is needed.

    Returns:
        The resulting numeric array or tensor.
    """
    if pca is None:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    return pca.transform(normalized_rows(x))


def normalized_rows(x: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalized vectors.

    Args:
        x: Input feature matrix.

    Returns:
        The resulting numeric array or tensor.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, np.where(norms == 0.0, 1.0, norms))


def pca_components(pca: PCA | None) -> int:
    """Return the fitted PCA width, or zero without a transform.

    Args:
        pca: Fitted PCA transform, or ``None`` when no reduction is needed.

    Returns:
        The computed index, count, or status code.
    """
    return int(pca.n_components_) if pca is not None else 0


def cluster_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate clustered step records and select nearest exemplars.

    Args:
        records: Step records containing cluster IDs, distances, and scalar features.

    Returns:
        Cluster sizes, correctness counts, average metrics, and up to five exemplars.
    """
    clusters: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        clusters.setdefault(int(rec["cluster_id"]), []).append(rec)

    summaries: list[dict[str, Any]] = []
    for cluster_id, items in sorted(clusters.items()):
        exemplars = sorted(items, key=lambda x: x.get("cluster_distance", 0.0))[:5]
        summaries.append(
            {
                "cluster_id": cluster_id,
                "size": len(items),
                "correct": sum(item.get("is_correct") is True for item in items),
                "incorrect": sum(item.get("is_correct") is False for item in items),
                "unknown": sum(item.get("is_correct") is None for item in items),
                "avg_variance": round(
                    sum(item["variance"] for item in items) / len(items), 6
                ),
                "avg_direction_norm": round(
                    sum(item["direction_norm"] for item in items) / len(items), 6
                ),
                "avg_nudge_norm": round(
                    sum(item["nudge_norm"] for item in items) / len(items), 6
                ),
                "exemplars": [
                    {
                        "sample_id": item["sample_id"],
                        "seed": item["seed"],
                        "segmenter": item["segmenter"],
                        "step_idx": item["step_idx"],
                        "text": item["step_text"],
                    }
                    for item in exemplars
                ],
            }
        )
    return summaries
