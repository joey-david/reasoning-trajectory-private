"""Correct/incorrect object-trajectory diagnostics and reranking."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from .iso_dataset import arithmetic_result
from .retrieval import normalize_rows


EDIT_ORDER = {"BIND": 0, "OPERATE": 1, "VERIFY": 2, "EXTRACT": 3}


def trajectory_report(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure when corrupted trajectories separate from matched gold traces."""
    prototypes: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for index, row in enumerate(records):
        if row["split"] == "train" and row["canonical_graph_id"] == row["gold_graph_id"]:
            prototypes[(row["canonical_graph_id"], row["edit_type"])].append(
                vectors[index]
            )
    normalized_prototypes = {
        key: normalize_rows(np.stack(values).mean(axis=0, keepdims=True))[0]
        for key, values in prototypes.items()
    }
    distances: defaultdict[str, list[float]] = defaultdict(list)
    labels: defaultdict[str, list[int]] = defaultdict(list)
    trace_distances: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    normalized = normalize_rows(vectors)
    for index, row in enumerate(records):
        if row["split"] == "train":
            continue
        prototype = normalized_prototypes.get(
            (row["gold_graph_id"], row["edit_type"])
        )
        if prototype is None:
            continue
        distance = float(1.0 - normalized[index] @ prototype)
        distances[row["edit_type"]].append(distance)
        labels[row["edit_type"]].append(int(row["is_correct"]))
        trace_distances[row["trace_id"]].append(
            (EDIT_ORDER[row["edit_type"]], distance)
        )
    thresholds = {
        edit: float(
            np.quantile(
                [
                    distance
                    for distance, label in zip(
                        distances[edit], labels[edit], strict=True
                    )
                    if label
                ],
                0.95,
            )
        )
        for edit in distances
        if any(labels[edit])
    }
    earliest: list[int] = []
    for trace_id, values in trace_distances.items():
        if "-correct" in trace_id:
            continue
        exceeding = [
            order
            for order, distance in values
            if distance > thresholds.get(
                next(
                    edit
                    for edit, edit_order in EDIT_ORDER.items()
                    if edit_order == order
                ),
                float("inf"),
            )
        ]
        if exceeding:
            earliest.append(min(exceeding))
    edit_reports = {}
    for edit in sorted(distances, key=EDIT_ORDER.get):
        y = np.asarray(labels[edit])
        scores = -np.asarray(distances[edit])
        edit_reports[edit] = {
            "examples": len(y),
            "correct_mean_distance_to_gold": float(
                np.mean(np.asarray(distances[edit])[y == 1])
            ),
            "incorrect_mean_distance_to_gold": (
                float(np.mean(np.asarray(distances[edit])[y == 0]))
                if np.any(y == 0)
                else None
            ),
            "correctness_auc": (
                float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None
            ),
        }
    return {
        "edit_reports": edit_reports,
        "divergence_thresholds": thresholds,
        "incorrect_trajectories_with_detected_divergence": len(earliest),
        "median_earliest_divergence_edit_index": (
            float(np.median(earliest)) if earliest else None
        ),
        "edit_order": EDIT_ORDER,
    }


def reranking_report(
    vectors: np.ndarray,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare answer-only consensus with graph-validity/object consistency."""
    groups: defaultdict[
        tuple[str, str, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for index, row in enumerate(records):
        if row["edit_type"] != "EXTRACT":
            continue
        key = (
            row["question_id"],
            row["split"],
            row["surface"]["lexical_family"],
        )
        groups[key][row["trace_id"]] = {"row": row, "index": index}
    evaluated = []
    normalized = normalize_rows(vectors)
    for _key, candidates_by_trace in groups.items():
        candidates = list(candidates_by_trace.values())
        if len(candidates) < 2:
            continue
        answers = [candidate["row"]["observed"]["result"] for candidate in candidates]
        counts = {answer: answers.count(answer) for answer in set(answers)}
        maximum = max(counts.values())
        winners = [answer for answer, count in counts.items() if count == maximum]
        consensus = winners[0] if len(winners) == 1 else None
        gold = candidates[0]["row"]["expected"]["result"]
        centroid = normalized[
            [
                candidate["index"]
                for candidate in candidates
                if candidate["row"]["is_correct"]
            ]
        ].mean(axis=0)
        ranked = []
        for candidate in candidates:
            row = candidate["row"]
            observed = row["observed"]
            validity = float(
                arithmetic_result(
                    observed["operation"],
                    int(observed["operand_a"]),
                    int(observed["operand_b"]),
                )
                == observed["result"]
            )
            consistency = float(normalized[candidate["index"]] @ centroid)
            ranked.append((validity + 0.1 * consistency, observed["result"]))
        object_choice = max(ranked)[1]
        evaluated.append(
            {
                "gold": gold,
                "consensus": consensus,
                "object_choice": object_choice,
            }
        )
    consensus_covered = [row for row in evaluated if row["consensus"] is not None]
    return {
        "questions": len(evaluated),
        "answer_consensus_coverage": (
            len(consensus_covered) / len(evaluated) if evaluated else None
        ),
        "answer_consensus_accuracy_when_covered": (
            float(
                np.mean([row["consensus"] == row["gold"] for row in consensus_covered])
            )
            if consensus_covered
            else None
        ),
        "object_graph_validity_accuracy": (
            float(
                np.mean(
                    [row["object_choice"] == row["gold"] for row in evaluated]
                )
            )
            if evaluated
            else None
        ),
        "note": (
            "This local controlled-bank pilot tests ranking mechanics. The medium "
            "run must replace synthetic corruptions with sampled model rollouts."
        ),
    }
