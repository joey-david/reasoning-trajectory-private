from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.analysis.step_classification.features import StepMatrices


def assign_clusters(
    records: list[dict[str, Any]],
    vectors: StepMatrices,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if len(records) < 3:
        return {"clusters": [], "feature_columns": []}

    step_cfg = cfg.get("step_classification", {})
    k = min(int(step_cfg.get("cluster_k", 8)), len(records) - 1)
    pca_dim = min(
        int(step_cfg.get("cluster_pca_dim", 32)),
        len(records) - 1,
        vectors.means.shape[1],
    )
    random_state = int(step_cfg.get("random_state", 42))

    mean_components = PCA(
        n_components=pca_dim, random_state=random_state
    ).fit_transform(vectors.means)
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
    direction_normed = normalized_pca(
        vectors.directions, min(8, pca_dim, len(records) - 1), random_state
    )
    nudge_normed = normalized_pca(
        vectors.nudges, min(8, pca_dim, len(records) - 1), random_state
    )
    features = np.concatenate(
        [mean_components, direction_normed, nudge_normed, scalars], axis=1
    )
    features = StandardScaler().fit_transform(features)

    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(features)
    distances = model.transform(features)[np.arange(len(labels)), labels]

    for rec, label, distance in zip(records, labels, distances):
        rec["cluster_id"] = int(label)
        rec["cluster_distance"] = round(float(distance), 6)

    return {
        "cluster_count": k,
        "feature_columns": {
            "mean_pca": pca_dim,
            "direction_pca": direction_normed.shape[1],
            "nudge_pca": nudge_normed.shape[1],
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


def normalized_pca(x: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    if n_components <= 0:
        return np.zeros((x.shape[0], 0), dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x_normed = np.divide(x, np.where(norms == 0.0, 1.0, norms))
    return PCA(n_components=n_components, random_state=random_state).fit_transform(
        x_normed
    )


def cluster_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
