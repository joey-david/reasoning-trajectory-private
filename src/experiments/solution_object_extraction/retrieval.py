"""Nearest-centroid object-state retrieval and controlled baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows without producing NaNs."""
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-8)


def class_centroids(
    vectors: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted class labels and normalized class centroids."""
    classes = np.asarray(sorted(set(labels.astype(str))), dtype=str)
    centroids = np.stack(
        [vectors[labels == label].mean(axis=0) for label in classes]
    )
    return classes, normalize_rows(centroids)


def evaluate_retrieval(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    test_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate cosine nearest-centroid retrieval."""
    classes, centroids = class_centroids(train_vectors, train_labels)
    similarities = normalize_rows(test_vectors) @ centroids.T
    order = np.argsort(-similarities, axis=1)
    predicted = classes[order[:, 0]]
    true_positions = np.asarray(
        [
            int(np.flatnonzero(classes[indices] == label)[0])
            if np.any(classes[indices] == label)
            else len(classes)
            for indices, label in zip(order, test_labels, strict=True)
        ]
    )
    top1 = float(np.mean(true_positions == 0))
    top5 = float(np.mean(true_positions < min(5, len(classes))))
    margins = []
    hard_results = []
    class_answer = graph_answer_by_class(train_labels, train_vectors, classes, [])
    record_answer_by_graph: dict[str, float] = {}
    for record in test_records:
        result = record["observed"].get("result")
        if result is not None:
            record_answer_by_graph[str(record["canonical_graph_id"])] = float(result)
    class_answer.update(record_answer_by_graph)
    for row_index, label in enumerate(test_labels):
        matches = np.flatnonzero(classes == label)
        if len(matches):
            own = float(similarities[row_index, matches[0]])
            others = np.delete(similarities[row_index], matches[0])
            margins.append(own - float(np.max(others)) if len(others) else own)
            answer = class_answer.get(str(label))
            same_answer = [
                index
                for index, candidate in enumerate(classes)
                if candidate != label and class_answer.get(str(candidate)) == answer
            ]
            if same_answer:
                hard_results.append(
                    own > float(np.max(similarities[row_index, same_answer]))
                )
    report = {
        "count": int(len(test_labels)),
        "top1": top1,
        "top5": top5,
        "mean_retrieval_margin": float(np.mean(margins)) if margins else None,
        "same_answer_hard_negative_accuracy": (
            float(np.mean(hard_results)) if hard_results else None
        ),
        "candidate_graphs": int(len(classes)),
    }
    return report, predicted, similarities


def graph_answer_by_class(
    _labels: np.ndarray,
    _vectors: np.ndarray,
    classes: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Extract result values for graph IDs when records are available."""
    answers: dict[str, float] = {}
    for record in records:
        result = record["observed"].get("result")
        if result is not None:
            answers[str(record["canonical_graph_id"])] = float(result)
    for label in classes:
        if str(label).endswith("_bound"):
            continue
        try:
            answers.setdefault(str(label), float(str(label).rsplit("_r", 1)[1]))
        except (IndexError, ValueError):
            pass
    return answers


def lexical_features(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a character n-gram lexical baseline."""
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=12000,
        sublinear_tf=True,
    )
    train = vectorizer.fit_transform(record["anchor_text"] for record in train_records)
    test = vectorizer.transform(record["anchor_text"] for record in test_records)
    return (
        train.astype(np.float32).toarray(),
        test.astype(np.float32).toarray(),
        {"features": int(train.shape[1]), "analyzer": "char_wb_3_5"},
    )


def evaluate_by_split(
    *,
    train_vectors: np.ndarray,
    all_vectors: np.ndarray,
    records: list[dict[str, Any]],
    train_indices: np.ndarray,
    splits: tuple[str, ...],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Evaluate one representation on each strict held-out surface split."""
    train_labels = np.asarray(
        [records[index]["canonical_graph_id"] for index in train_indices], dtype=str
    )
    reports: dict[str, Any] = {}
    per_record: dict[int, dict[str, Any]] = {}
    for split in splits:
        indices = np.asarray(
            [index for index, row in enumerate(records) if row["split"] == split],
            dtype=int,
        )
        if not len(indices):
            continue
        labels = np.asarray(
            [records[index]["canonical_graph_id"] for index in indices], dtype=str
        )
        report, predicted, similarities = evaluate_retrieval(
            train_vectors,
            all_vectors[indices],
            train_labels,
            labels,
            [records[index] for index in indices],
        )
        reports[split] = report
        classes, _ = class_centroids(train_vectors, train_labels)
        for local, index in enumerate(indices):
            own = np.flatnonzero(classes == labels[local])
            sorted_scores = np.sort(similarities[local])
            margin = None
            if len(own):
                other = np.delete(similarities[local], own[0])
                margin = float(similarities[local, own[0]] - np.max(other))
            per_record[int(index)] = {
                "retrieved_graph_id": str(predicted[local]),
                "retrieval_margin": margin,
                "top_similarity": float(sorted_scores[-1]),
            }
    return reports, per_record


def surface_gap(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
    indices: np.ndarray,
) -> dict[str, float | None]:
    """Compare same-object/different-vocabulary and lexical-only similarities."""
    normalized = normalize_rows(vectors[indices])
    same_object: list[float] = []
    same_vocab_different_object: list[float] = []
    for left in range(len(indices)):
        a = records[int(indices[left])]
        for right in range(left + 1, len(indices)):
            b = records[int(indices[right])]
            similarity = float(normalized[left] @ normalized[right])
            if (
                a["canonical_graph_id"] == b["canonical_graph_id"]
                and a["surface"]["lexical_family"]
                != b["surface"]["lexical_family"]
            ):
                same_object.append(similarity)
            elif (
                a["canonical_graph_id"] != b["canonical_graph_id"]
                and a["surface"]["lexical_family"]
                == b["surface"]["lexical_family"]
            ):
                same_vocab_different_object.append(similarity)
    object_mean = float(np.mean(same_object)) if same_object else None
    lexical_mean = (
        float(np.mean(same_vocab_different_object))
        if same_vocab_different_object
        else None
    )
    return {
        "same_object_different_vocabulary": object_mean,
        "different_object_same_vocabulary": lexical_mean,
        "gap": (
            object_mean - lexical_mean
            if object_mean is not None and lexical_mean is not None
            else None
        ),
    }
