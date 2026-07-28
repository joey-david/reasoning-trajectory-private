"""Artifact-only reductions for the six causal reasoning questions."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.experiments.depth_relief.metrics import cluster_bootstrap_mean_ci
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["condition"]),
        str(row["layer_mode"]),
        "none" if row["layer"] is None else str(row["layer"]),
        int(row["token_width"]),
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: defaultdict[
        tuple[str, str, str, int], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for case in rows:
        for result in case["results"]:
            cells[_cell_key(result)].append((case, result))
    summaries = []
    for key, selected in sorted(cells.items()):
        condition, mode, layer, width = key
        groups = [str(case["group"]) for case, _ in selected]
        correct = [bool(result["is_expected"]) for _, result in selected]
        unconstrained = [
            bool(result["is_expected_unconstrained"]) for _, result in selected
        ]
        probability = [
            float(result["expected_probability"]) for _, result in selected
        ]
        changes = [
            float(result.get("expected_probability_change", 0.0))
            for _, result in selected
        ]
        accuracy_changes = [
            float(result.get("accuracy_change", 0.0))
            for _, result in selected
        ]
        candidate_mass = [
            float(result["candidate_probability_mass"])
            for _, result in selected
        ]
        invalid = [
            result["unconstrained_prediction"] is None
            for _, result in selected
        ]
        summaries.append(
            {
                "condition": condition,
                "layer_mode": mode,
                "layer": None if layer == "none" else int(layer),
                "token_width": width,
                "case_count": len(selected),
                "accuracy": cluster_bootstrap_mean_ci(
                    correct, groups, seed=86_101
                ),
                "unconstrained_accuracy": cluster_bootstrap_mean_ci(
                    unconstrained, groups, seed=86_102
                ),
                "expected_probability": cluster_bootstrap_mean_ci(
                    probability, groups, seed=86_103
                ),
                "expected_probability_change": cluster_bootstrap_mean_ci(
                    changes, groups, seed=86_104
                ),
                "accuracy_change_from_recipient": cluster_bootstrap_mean_ci(
                    accuracy_changes, groups, seed=86_105
                ),
                "candidate_probability_mass": cluster_bootstrap_mean_ci(
                    candidate_mass, groups, seed=86_106
                ),
                "invalid_output_rate": cluster_bootstrap_mean_ci(
                    invalid, groups, seed=86_107
                ),
            }
        )
    return {"cells": summaries}


def _summarize_representations(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cells: defaultdict[
        tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]
    ] = defaultdict(list)
    for case in rows:
        for comparison in case.get("representation", []):
            cells[
                (str(comparison["pair"]), int(comparison["layer"]))
            ].append((case, comparison))
    result = []
    for (pair, layer), selected in sorted(cells.items()):
        groups = [str(case["group"]) for case, _ in selected]
        result.append(
            {
                "pair": pair,
                "layer": layer,
                "case_count": len(selected),
                "cosine_similarity": cluster_bootstrap_mean_ci(
                    [
                        float(comparison["cosine_similarity"])
                        for _, comparison in selected
                    ],
                    groups,
                    seed=86_111,
                ),
                "normalized_distance": cluster_bootstrap_mean_ci(
                    [
                        float(comparison["normalized_distance"])
                        for _, comparison in selected
                    ],
                    groups,
                    seed=86_112,
                ),
            }
        )
    return result


def _best_layers(
    validation: dict[str, Any],
    test: dict[str, Any],
) -> list[dict[str, Any]]:
    by_condition: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in validation["cells"]:
        by_condition[str(cell["condition"])].append(cell)
    test_index = {
        (
            str(cell["condition"]),
            str(cell["layer_mode"]),
            cell["layer"],
            int(cell["token_width"]),
        ): cell
        for cell in test["cells"]
    }
    result = []
    for condition, cells in sorted(by_condition.items()):
        patched = [cell for cell in cells if cell["layer_mode"] != "baseline"]
        candidates = patched or cells
        selected = max(
            candidates,
            key=lambda cell: (
                float(cell["accuracy"]["mean"]),
                float(cell["expected_probability"]["mean"]),
                -1 if cell["layer"] is None else -int(cell["layer"]),
            ),
        )
        key = (
            condition,
            str(selected["layer_mode"]),
            selected["layer"],
            int(selected["token_width"]),
        )
        result.append(
            {
                "condition": condition,
                "selected_on": "validation accuracy then expected probability",
                "validation": selected,
                "test": test_index.get(key),
            }
        )
    return result


def _load_feature_rows(
    run_path: Path,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, list[int], list[dict[str, Any]]]:
    vectors = []
    labels = []
    layers: list[int] | None = None
    for row in rows:
        if not row.get("feature_file"):
            continue
        with np.load(run_path / row["feature_file"]) as stored:
            current_layers = stored["layers"].astype(int).tolist()
            if layers is None:
                layers = current_layers
            elif current_layers != layers:
                raise ValueError("Feature layer order changed within one run")
            vectors.append(stored["states"].astype(np.float32))
        labels.append(row)
    if not vectors or layers is None:
        return np.empty((0, 0, 0)), [], []
    return np.stack(vectors), layers, labels


def _categorical_probe(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[list[dict[str, float]], int]:
    curve = []
    models = []
    for layer_col in range(train_x.shape[1]):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=86_201,
            ),
        )
        model.fit(train_x[:, layer_col], train_y)
        validation_prediction = model.predict(validation_x[:, layer_col])
        curve.append(
            {
                "validation_accuracy": float(
                    accuracy_score(validation_y, validation_prediction)
                ),
                "test_accuracy": float(
                    accuracy_score(
                        test_y, model.predict(test_x[:, layer_col])
                    )
                ),
            }
        )
        models.append(model)
    selected = max(
        range(len(curve)),
        key=lambda index: (curve[index]["validation_accuracy"], -index),
    )
    return curve, selected


def _numeric_probe(
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    validation_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[list[dict[str, float]], int]:
    curve = []
    for layer_col in range(train_x.shape[1]):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(train_x[:, layer_col], train_y)
        validation_prediction = model.predict(validation_x[:, layer_col])
        test_prediction = model.predict(test_x[:, layer_col])
        curve.append(
            {
                "validation_mae": float(
                    mean_absolute_error(validation_y, validation_prediction)
                ),
                "test_mae": float(mean_absolute_error(test_y, test_prediction)),
                "test_r2": float(r2_score(test_y, test_prediction)),
            }
        )
    selected = min(
        range(len(curve)),
        key=lambda index: (curve[index]["validation_mae"], index),
    )
    return curve, selected


def _feature_probes(
    run_path: Path,
    rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any] | None:
    x, layers, records = _load_feature_rows(run_path, rows)
    if not len(records):
        return None
    indices = {
        split: np.asarray(
            [index for index, row in enumerate(records) if row["split"] == split]
        )
        for split in ("train", "validation", "test")
    }
    if any(not len(value) for value in indices.values()):
        raise ValueError("Feature probes require nonempty train/validation/test")
    common_fields = set(records[0]["labels"])
    for row in records[1:]:
        common_fields &= set(row["labels"])
    probes = {}
    cases_by_id = {str(case["id"]): case for case in cases}
    texts = np.asarray(
        [
            cases_by_id[str(row["id"])]["prompts"][
                cases_by_id[str(row["id"])]["feature_prompt"]
            ]["text"]
            for row in records
        ]
    )
    for field in sorted(common_fields):
        y = np.asarray([row["labels"][field] for row in records])
        train_y = y[indices["train"]]
        validation_y = y[indices["validation"]]
        test_y = y[indices["test"]]
        if len(set(train_y.tolist())) < 2:
            continue
        if np.issubdtype(y.dtype, np.integer) and len(set(y.tolist())) <= 5:
            curve, selected = _categorical_probe(
                x[indices["train"]],
                x[indices["validation"]],
                x[indices["test"]],
                train_y,
                validation_y,
                test_y,
            )
            metric = "accuracy"
            text_model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(2, 5)),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=86_202,
                ),
            )
            text_model.fit(texts[indices["train"]], train_y)
            text_result = {
                "validation_accuracy": float(
                    accuracy_score(
                        validation_y,
                        text_model.predict(texts[indices["validation"]]),
                    )
                ),
                "test_accuracy": float(
                    accuracy_score(
                        test_y, text_model.predict(texts[indices["test"]])
                    )
                ),
            }
        else:
            curve, selected = _numeric_probe(
                x[indices["train"]],
                x[indices["validation"]],
                x[indices["test"]],
                train_y.astype(float),
                validation_y.astype(float),
                test_y.astype(float),
            )
            metric = "mae"
            text_model = make_pipeline(
                TfidfVectorizer(analyzer="char", ngram_range=(2, 5)),
                Ridge(alpha=10.0),
            )
            text_model.fit(texts[indices["train"]], train_y.astype(float))
            text_result = {
                "validation_mae": float(
                    mean_absolute_error(
                        validation_y,
                        text_model.predict(texts[indices["validation"]]),
                    )
                ),
                "test_mae": float(
                    mean_absolute_error(
                        test_y,
                        text_model.predict(texts[indices["test"]]),
                    )
                ),
            }
        probes[field] = {
            "task": metric,
            "selected_layer": layers[selected],
            "selected_result": curve[selected],
            "text_only_control": text_result,
            "layer_curve": {
                str(layer): value for layer, value in zip(layers, curve)
            },
        }
    return {
        "split": "fixed group-disjoint train/validation/test",
        "model_selection": "validation only",
        "probes": probes,
    }


def reduce_experiment(run_path: Path) -> dict[str, Any]:
    """Reduce one experiment without rerunning model inference."""
    config = load_config(run_path)["causal_reasoning"]
    cases = load_samples(run_path / "dataset.jsonl")
    output_path = run_path / "evaluation" / "cases.jsonl"
    outputs = load_samples(output_path) if output_path.exists() else []
    expected_ids = {str(row["id"]) for row in cases}
    output_ids = [str(row["id"]) for row in outputs]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError(f"Duplicate case IDs in {output_path}")
    if not set(output_ids) <= expected_ids:
        raise ValueError("Evaluation contains unknown case IDs")
    by_split = {
        split: _summarize_rows(
            [row for row in outputs if row["split"] == split]
        )
        for split in ("train", "validation", "test")
    }
    complete = set(output_ids) == expected_ids
    has_all_splits = all(by_split[split]["cells"] for split in by_split)
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "hypothesis": config["hypothesis"],
        "status": "complete" if complete else "partial",
        "case_count": len(cases),
        "completed_case_count": len(outputs),
        "splits": by_split,
        "representational_alignment": {
            split: _summarize_representations(
                [row for row in outputs if row["split"] == split]
            )
            for split in ("train", "validation", "test")
        },
        "validation_selected_test": _best_layers(
            by_split["validation"], by_split["test"]
        )
        if has_all_splits
        else [],
        "feature_probes": _feature_probes(run_path, outputs, cases),
        "inference_contract": {
            "checkpoint": "exact marked token span",
            "patch": "residual replacement; recipient KV retained except in truncated-context tests",
            "probe_layers": config["probe_layers"],
        },
    }
    write_json(run_path / "evaluation" / "summary.json", summary)
    return summary


def reduce_suite(run_path: Path) -> dict[str, Any]:
    """Reduce every child run and write one compact model-level index."""
    config = load_config(run_path)
    summaries = [
        reduce_experiment(Path(str(child)))
        for child in config["causal_reasoning_suite"]["runs"]
    ]
    result = {
        "schema_version": 1,
        "model": config["model"]["name"],
        "status": (
            "complete"
            if all(summary["status"] == "complete" for summary in summaries)
            else "partial"
        ),
        "experiments": {
            str(summary["experiment"]): {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "summary": (
                    Path(str(child)) / "evaluation" / "summary.json"
                ).as_posix(),
            }
            for child, summary in zip(
                config["causal_reasoning_suite"]["runs"], summaries
            )
        },
    }
    write_json(run_path / "evaluation" / "suite_summary.json", result)
    return result
