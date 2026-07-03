"""Text-only baselines for semantic token-boundary detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.experiments.token_segmentation.data import TraceKey
from src.experiments.token_segmentation.semantic_labels import SemanticTrace


@dataclass(slots=True)
class TextBoundaryModel:
    """A fitted local-context vectorizer and linear boundary classifier."""

    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


def evaluate_text_boundary_baselines(
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    *,
    context_chars: int = 96,
) -> dict[str, Any]:
    """Fit and evaluate full and non-sentence text-only boundary detectors."""
    output: dict[str, Any] = {
        "representation": (
            f"character 2-5 gram TF-IDF over {context_chars} characters "
            "on each side of the token transition"
        )
    }
    rng = np.random.default_rng(0)
    for name, residual in (("all", False), ("non_sentence", True)):
        model = _fit_model(
            trace_map,
            semantic,
            context_chars=context_chars,
            sentence_residual=residual,
        )
        scores: dict[str, float] = {}
        for key, annotation in semantic.items():
            trace = trace_map[key]
            if trace.train:
                continue
            boundaries = (
                _far_from(annotation.boundaries, trace.sentence_boundaries, 4)
                if residual
                else annotation.boundaries
            )
            contexts, valid_indices = transition_contexts(trace, context_chars)
            labels = np.isin(valid_indices, boundaries).astype(np.int8)
            if len(np.unique(labels)) < 2:
                continue
            probabilities = model.classifier.predict_proba(
                model.vectorizer.transform(contexts)
            )[:, 1]
            scores[trace.sample_id] = float(roc_auc_score(labels, probabilities))
        values = list(scores.values())
        output[name] = {
            "question_mean_auc": float(np.mean(values)),
            "question_bootstrap_95ci": _bootstrap_interval(values, rng),
            "questions": len(values),
            "question_aucs": scores,
        }
    return output


def paired_auc_difference(
    latent: dict[str, float],
    text: dict[str, float],
) -> dict[str, Any]:
    """Bootstrap the paired latent-minus-text AUC difference by question."""
    questions = sorted(set(latent) & set(text))
    differences = [latent[question] - text[question] for question in questions]
    return {
        "latent_minus_text": float(np.mean(differences)),
        "question_bootstrap_95ci": _bootstrap_interval(
            differences, np.random.default_rng(0)
        ),
        "questions": len(questions),
    }


def compare_latent_text_auc(
    report: dict[str, Any],
    text_baselines: dict[str, Any],
) -> dict[str, Any]:
    """Compare matched latent and text boundary AUCs for both targets."""
    latent_keys = {
        "all": "semantic_boundary_auc_by_question",
        "non_sentence": "non_sentence_semantic_boundary_auc_by_question",
    }
    return {
        name: paired_auc_difference(report[key], text_baselines[name]["question_aucs"])
        for name, key in latent_keys.items()
    }


def transition_contexts(
    trace: Any,
    context_chars: int,
) -> tuple[list[str], np.ndarray]:
    """Extract local text around each character-aligned token transition."""
    contexts: list[str] = []
    indices: list[int] = []
    for boundary in range(max(0, trace.token_count - 1)):
        if boundary >= len(trace.token_char_ends):
            break
        offset = int(trace.token_char_ends[boundary])
        if offset < 0:
            continue
        left = trace.text[max(0, offset - context_chars) : offset]
        right = trace.text[offset : min(len(trace.text), offset + context_chars)]
        contexts.append(f"{left} <<<CUT>>> {right}")
        indices.append(boundary)
    return contexts, np.asarray(indices, dtype=np.int32)


def _fit_model(
    trace_map: dict[TraceKey, Any],
    semantic: dict[TraceKey, SemanticTrace],
    *,
    context_chars: int,
    sentence_residual: bool,
) -> TextBoundaryModel:
    """Fit one balanced text-only detector on training questions."""
    positive_contexts: list[str] = []
    negative_contexts: list[str] = []
    for key, annotation in semantic.items():
        trace = trace_map[key]
        if not trace.train:
            continue
        boundaries = (
            _far_from(annotation.boundaries, trace.sentence_boundaries, 4)
            if sentence_residual
            else annotation.boundaries
        )
        contexts, indices = transition_contexts(trace, context_chars)
        labels = np.isin(indices, boundaries)
        positive_contexts.extend(
            context for context, label in zip(contexts, labels, strict=True) if label
        )
        negative_contexts.extend(
            context
            for context, label in zip(contexts, labels, strict=True)
            if not label
        )
    rng = np.random.default_rng(0)
    negative_indices = rng.choice(
        len(negative_contexts),
        size=min(len(negative_contexts), 4 * len(positive_contexts)),
        replace=False,
    )
    texts = positive_contexts + [negative_contexts[index] for index in negative_indices]
    labels = np.r_[
        np.ones(len(positive_contexts), dtype=np.int8),
        np.zeros(len(negative_indices), dtype=np.int8),
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=20_000,
        sublinear_tf=True,
    )
    features = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=500,
        random_state=0,
    ).fit(features, labels)
    return TextBoundaryModel(vectorizer, classifier)


def _far_from(
    values: np.ndarray,
    references: np.ndarray,
    tolerance: int,
) -> np.ndarray:
    """Return boundaries farther than a tolerance from every reference."""
    if not len(references):
        return values
    distances = np.min(np.abs(values[:, None] - references[None, :]), axis=1)
    return values[distances > tolerance]


def _bootstrap_interval(
    values: list[float],
    rng: np.random.Generator,
    *,
    samples: int = 2000,
) -> list[float]:
    """Bootstrap a mean over held-out questions."""
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
