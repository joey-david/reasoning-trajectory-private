"""Decode semantic span labels from latent and lexical representations."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.experiments.token_segmentation.data import TraceKey, load_states
from src.experiments.token_segmentation.semantic_labels import SemanticTrace


def evaluate_semantic_labels(
    run_path: Path,
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    projection: Any,
    layer: int,
) -> dict[str, Any]:
    """Decode span types with question-disjoint latent and lexical models."""
    records: list[tuple[np.ndarray, np.ndarray, str, str, bool, str]] = []
    for key, annotation in semantic.items():
        trace = trace_map[key]
        states = projection.transform(load_states(run_path, trace, layer))
        for span in annotation.spans:
            end = min(span.token_end, len(states) - 1)
            start = min(span.token_start, end)
            block = states[start : end + 1]
            mean = block.mean(axis=0)
            latent = np.concatenate(
                [
                    mean,
                    block[0],
                    block[-1],
                    block[-1] - block[0],
                    [np.log1p(len(block))],
                ]
            )
            records.append(
                (
                    latent,
                    mean,
                    span.text,
                    span.label,
                    trace.train,
                    trace.sample_id,
                )
            )
    train_counts = Counter(record[3] for record in records if record[4])
    test_counts = Counter(record[3] for record in records if not record[4])
    classes = sorted(
        label
        for label, count in train_counts.items()
        if count >= 20 and test_counts[label] >= 2
    )
    filtered = [record for record in records if record[3] in classes]
    train = [record for record in filtered if record[4]]
    test = [record for record in filtered if not record[4]]
    y_train = np.asarray([record[3] for record in train])
    y_test = np.asarray([record[3] for record in test])

    scaler = StandardScaler().fit(np.stack([record[0] for record in train]))
    latent_train = scaler.transform(np.stack([record[0] for record in train]))
    latent_test = scaler.transform(np.stack([record[0] for record in test]))
    latent_model = _classifier().fit(latent_train, y_train)
    latent_scores = latent_model.predict_proba(latent_test)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=10_000, sublinear_tf=True
    )
    lexical_train = vectorizer.fit_transform([record[2] for record in train])
    lexical_test = vectorizer.transform([record[2] for record in test])
    lexical_model = _classifier().fit(lexical_train, y_train)
    lexical_scores = lexical_model.predict_proba(lexical_test)

    combined_model = _classifier().fit(
        hstack([lexical_train, csr_matrix(latent_train)]), y_train
    )
    combined_scores = combined_model.predict_proba(
        hstack([lexical_test, csr_matrix(latent_test)])
    )
    majority = train_counts.most_common(1)[0][0]
    return {
        "classes": classes,
        "train_spans": len(train),
        "test_spans": len(test),
        "train_label_counts": dict(Counter(y_train)),
        "test_label_counts": dict(Counter(y_test)),
        "majority_macro_f1": float(
            f1_score(y_test, np.full(len(y_test), majority), average="macro")
        ),
        "unsupervised_pca_cosine_auc": _pairwise_cosine_auc(test),
        "latent": _classification_metrics(
            y_test, latent_scores, latent_model.classes_
        ),
        "lexical_tfidf": _classification_metrics(
            y_test, lexical_scores, lexical_model.classes_
        ),
        "combined": _classification_metrics(
            y_test, combined_scores, combined_model.classes_
        ),
    }


def _classifier() -> LogisticRegression:
    """Construct the shared balanced multiclass classifier."""
    return LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=0
    )


def _classification_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    """Return compact multiclass metrics and per-label recall."""
    prediction = classes[np.argmax(scores, axis=1)]
    report = classification_report(
        target, prediction, labels=classes, output_dict=True, zero_division=0
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "macro_ovr_auc": float(
            roc_auc_score(target, scores, labels=classes, multi_class="ovr")
        ),
        "per_label_recall": {
            label: float(report[label]["recall"]) for label in classes
        },
    }


def _pairwise_cosine_auc(
    records: list[tuple[np.ndarray, np.ndarray, str, str, bool, str]],
) -> float:
    """Measure raw label clustering across different held-out questions."""
    means = np.stack([record[1] for record in records])
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-8)
    similarities = means @ means.T
    labels = np.asarray([record[3] for record in records])
    questions = np.asarray([record[5] for record in records])
    left, right = np.triu_indices(len(records), k=1)
    keep = questions[left] != questions[right]
    same_label = labels[left[keep]] == labels[right[keep]]
    return float(roc_auc_score(same_label, similarities[left[keep], right[keep]]))
