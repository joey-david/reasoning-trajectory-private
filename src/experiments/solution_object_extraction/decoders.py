"""Factorized linear decoding of object edits and graph fields."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CLASS_FIELDS = {
    "edit_type": lambda row: row["edit_type"],
    "operation": lambda row: row["observed"]["operation"],
    "target": lambda row: row["observed"]["target"],
}


def decode_representation(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fit independent linear probes for categorical and numeric graph fields."""
    report: dict[str, Any] = {}
    predicted_fields: dict[str, np.ndarray] = {}
    for name, getter in CLASS_FIELDS.items():
        train_y = np.asarray([getter(row) for row in train_records], dtype=str)
        test_y = np.asarray([getter(row) for row in test_records], dtype=str)
        if len(set(train_y)) == 1:
            predicted = np.full(len(test_y), train_y[0], dtype=str)
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=42
                ),
            )
            model.fit(train_vectors, train_y)
            predicted = model.predict(test_vectors)
        predicted_fields[name] = predicted
        report[name] = {
            "macro_f1": float(f1_score(test_y, predicted, average="macro")),
            "accuracy": float(accuracy_score(test_y, predicted)),
        }

    value_report: dict[str, Any] = {}
    predicted_values: dict[str, np.ndarray] = {}
    for field in ("operand_a", "operand_b", "result"):
        train_mask = np.asarray(
            [row["observed"].get(field) is not None for row in train_records]
        )
        test_mask = np.asarray(
            [row["observed"].get(field) is not None for row in test_records]
        )
        if not np.any(test_mask):
            continue
        train_y = np.asarray(
            [
                float(row["observed"][field])
                for row, keep in zip(train_records, train_mask, strict=True)
                if keep
            ]
        )
        test_y = np.asarray(
            [
                float(row["observed"][field])
                for row, keep in zip(test_records, test_mask, strict=True)
                if keep
            ]
        )
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train_vectors[train_mask], train_y)
        predicted = model.predict(test_vectors[test_mask])
        expanded = np.full(len(test_records), np.nan, dtype=np.float32)
        expanded[test_mask] = predicted
        predicted_values[field] = expanded
        value_report[field] = {
            "mae": float(mean_absolute_error(test_y, predicted)),
            "rounded_accuracy": float(np.mean(np.rint(predicted) == test_y)),
        }
    exact = np.ones(len(test_records), dtype=bool)
    for name, getter in CLASS_FIELDS.items():
        exact &= predicted_fields[name] == np.asarray(
            [getter(row) for row in test_records], dtype=str
        )
    for field, predicted in predicted_values.items():
        truth = np.asarray(
            [
                np.nan
                if row["observed"].get(field) is None
                else float(row["observed"][field])
                for row in test_records
            ]
        )
        mask = np.isfinite(truth)
        exact[mask] &= np.rint(predicted[mask]) == truth[mask]
    report["values"] = value_report
    report["exact_factorized_state_match"] = float(np.mean(exact))
    return report
