"""Representational similarity analysis for object versus surface structure."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

from .retrieval import normalize_rows


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return pairwise entries above the diagonal."""
    indices = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix[indices], dtype=np.float64)


def categorical_similarity(values: list[Any]) -> np.ndarray:
    """Build an exact-match similarity matrix."""
    array = np.asarray(values)
    return (array[:, None] == array[None, :]).astype(np.float32)


def numeric_object_similarity(records: list[dict[str, Any]]) -> np.ndarray:
    """Build graded similarity from operation, role bindings, and result."""
    count = len(records)
    matrix = np.zeros((count, count), dtype=np.float32)
    for left, a in enumerate(records):
        av = a["observed"]
        for right, b in enumerate(records):
            bv = b["observed"]
            fields = (
                av["operation"] == bv["operation"],
                av["operand_a"] == bv["operand_a"],
                av["operand_b"] == bv["operand_b"],
                av.get("result") == bv.get("result"),
                a["edit_type"] == b["edit_type"],
            )
            matrix[left, right] = float(np.mean(fields))
    return matrix


def partial_spearman(
    target: np.ndarray,
    predictor: np.ndarray,
    controls: list[np.ndarray],
) -> float:
    """Compute rank correlation after linear residualization on controls."""
    y = rank_values(target)
    x = rank_values(predictor)
    design = np.column_stack(
        [np.ones(len(y))] + [rank_values(control) for control in controls]
    )
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    return float(np.corrcoef(x_residual, y_residual)[0, 1])


def rank_values(values: np.ndarray) -> np.ndarray:
    """Return stable average ranks through SciPy's Spearman implementation."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(values, dtype=np.float64), method="average")


def run_rsa(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare vector geometry with object, lexical, answer, and template geometry."""
    activation = normalize_rows(vectors) @ normalize_rows(vectors).T
    object_matrix = numeric_object_similarity(records)
    answer_matrix = categorical_similarity(
        [record["observed"].get("result") for record in records]
    )
    template_matrix = categorical_similarity(
        [record["surface"]["template_id"] for record in records]
    )
    lexical_family = categorical_similarity(
        [record["surface"]["lexical_family"] for record in records]
    )
    lexical_vectors = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1
    ).fit_transform(record["anchor_text"] for record in records)
    lexical_matrix = (lexical_vectors @ lexical_vectors.T).toarray()
    values = {
        "activation": upper_triangle(activation),
        "object": upper_triangle(object_matrix),
        "lexical": upper_triangle(lexical_matrix),
        "lexical_family": upper_triangle(lexical_family),
        "answer": upper_triangle(answer_matrix),
        "template": upper_triangle(template_matrix),
    }
    correlations = {
        name: float(spearmanr(values["activation"], values[name]).statistic)
        for name in ("object", "lexical", "lexical_family", "answer", "template")
    }
    partial_object = partial_spearman(
        values["activation"],
        values["object"],
        [values["lexical"], values["answer"], values["template"]],
    )
    return {
        "records": len(records),
        "pair_count": int(len(values["activation"])),
        "spearman": correlations,
        "partial_object_controlling_lexical_answer_template": partial_object,
        "object_minus_lexical": correlations["object"] - correlations["lexical"],
    }
