"""Localization and causal self-handoff of an implicitly synthesized state."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .benchmark import apply_rule, candidate_token_ids, state_symbols
from .decoding import fit_centroid_decoder, predict_centroid
from .hf import patched_logits
from .metrics import bootstrap_mean_ci
from .qualification import score_logits
from .transfer import render_transfer_prompts


HANDOFF_MODES = ("self_state", "random_self", "full_self")
DEFAULT_LOCALIZATION_GATE = {
    "min_history_state_accuracy_lower": 0.30,
    "min_history_over_shuffled_lower": 0.15,
    "min_history_over_initial_lower": 0.10,
}
DEFAULT_CAUSAL_GATE = {
    "min_self_shift_lower": 0.10,
    "min_self_over_random_lower": 0.10,
    "min_margin_improvement_lower": 0.10,
}


def _position_map(capture: dict[str, Any]) -> dict[str, int]:
    return {
        str(row["name"]): index
        for index, row in enumerate(capture["compose_positions"])
    }


def trace_position(
    arrays: dict[str, np.ndarray], capture: dict[str, Any], name: str
) -> np.ndarray:
    """Return all-layer residuals for one named Compose position."""
    positions = _position_map(capture)
    if name not in positions:
        raise ValueError(f"Capture does not contain Compose position {name!r}")
    trace = np.asarray(arrays["compose_trace"], dtype=np.float32)
    return trace[positions[name]]


def analyze_state_localization(
    *,
    cases: dict[str, dict[str, Any]],
    split: dict[str, Any],
    captures: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, np.ndarray]],
    rank: int,
    seed: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Train on explicit states, select a layer on validation, and test once."""
    class_count = 2 ** int(next(iter(cases.values()))["bits"])
    layer_count = int(next(iter(activations.values()))["compose"].shape[0])
    train_ids = list(split["train"])

    explicit_values: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
    explicit_labels: list[int] = []
    for case_id in train_ids:
        case = cases[case_id]
        arrays = activations[case_id]
        explicit_labels.extend(
            [int(case["current_state"]), int(case["counterfactual_state"])]
        )
        for layer in range(layer_count):
            explicit_values[layer].extend(
                [arrays["materialized"][layer], arrays["counterfactual"][layer]]
            )
    labels = np.asarray(explicit_labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    decoders = [
        fit_centroid_decoder(
            np.stack(explicit_values[layer]),
            labels,
            class_count=class_count,
            rank=rank,
        )
        for layer in range(layer_count)
    ]
    shuffled_decoders = [
        fit_centroid_decoder(
            np.stack(explicit_values[layer]),
            shuffled,
            class_count=class_count,
            rank=rank,
        )
        for layer in range(layer_count)
    ]

    validation_ids = list(split["validation"])
    validation_labels = np.asarray(
        [int(cases[case_id]["current_state"]) for case_id in validation_ids]
    )
    validation_scores: dict[int, dict[str, float]] = {}
    for layer in range(layer_count):
        # Histories have different lengths, so collect each case's own final step.
        values = np.stack(
            [
                trace_position(
                    activations[case_id],
                    captures[case_id],
                    f"history_step_{int(cases[case_id]['history_steps'])}",
                )[layer]
                for case_id in validation_ids
            ]
        )
        predictions = predict_centroid(decoders[layer], values)
        shuffled_predictions = predict_centroid(shuffled_decoders[layer], values)
        accuracy = float(np.mean(predictions == validation_labels))
        shuffled_accuracy = float(np.mean(shuffled_predictions == validation_labels))
        validation_scores[layer] = {
            "current_state_accuracy": accuracy,
            "shuffled_accuracy": shuffled_accuracy,
            "selection_score": accuracy - shuffled_accuracy,
        }
    selected = max(
        range(layer_count),
        key=lambda layer: (validation_scores[layer]["selection_score"], -layer),
    )

    test_ids = list(split["test"])
    common_positions = ["start", "history_end", "final_rule", "answer"]
    predictions_by_position: dict[str, np.ndarray] = {}
    shuffled_by_position: dict[str, np.ndarray] = {}
    for position in common_positions:
        values = np.stack(
            [
                trace_position(
                    activations[case_id],
                    captures[case_id],
                    (
                        f"history_step_{int(cases[case_id]['history_steps'])}"
                        if position == "history_end"
                        else position
                    ),
                )[selected]
                for case_id in test_ids
            ]
        )
        predictions_by_position[position] = predict_centroid(
            decoders[selected], values
        )
        shuffled_by_position[position] = predict_centroid(
            shuffled_decoders[selected], values
        )

    label_rows = {
        "current_state": np.asarray(
            [int(cases[case_id]["current_state"]) for case_id in test_ids]
        ),
        "initial_state": np.asarray(
            [int(cases[case_id]["initial_state"]) for case_id in test_ids]
        ),
        "next_state": np.asarray(
            [int(cases[case_id]["next_state"]) for case_id in test_ids]
        ),
        "final_on_start": np.asarray(
            [
                apply_rule(
                    cases[case_id]["final_rule"],
                    int(cases[case_id]["initial_state"]),
                    class_count,
                )
                for case_id in test_ids
            ]
        ),
    }
    test_metrics: dict[str, Any] = {}
    for position_index, position in enumerate(common_positions):
        predictions = predictions_by_position[position]
        test_metrics[position] = {
            label: bootstrap_mean_ci(
                predictions == targets,
                seed=seed + 100 + position_index * 20 + label_index,
            )
            for label_index, (label, targets) in enumerate(label_rows.items())
        }
        test_metrics[position]["shuffled_current_state"] = bootstrap_mean_ci(
            shuffled_by_position[position] == label_rows["current_state"],
            seed=seed + 110 + position_index * 20,
        )
        test_metrics[position]["current_over_initial"] = bootstrap_mean_ci(
            (predictions == label_rows["current_state"]).astype(int)
            - (predictions == label_rows["initial_state"]).astype(int),
            seed=seed + 111 + position_index * 20,
        )
        test_metrics[position]["current_over_shuffled"] = bootstrap_mean_ci(
            (predictions == label_rows["current_state"]).astype(int)
            - (
                shuffled_by_position[position] == label_rows["current_state"]
            ).astype(int),
            seed=seed + 112 + position_index * 20,
        )

    thresholds = {**DEFAULT_LOCALIZATION_GATE, **gate}

    def lower(stat: dict[str, Any]) -> float:
        value = stat["ci95"][0]
        return float(value) if value is not None else float("-inf")

    history = test_metrics["history_end"]
    checks = {
        "history_state_decodable": lower(history["current_state"])
        >= float(thresholds["min_history_state_accuracy_lower"]),
        "history_beats_shuffled": lower(history["current_over_shuffled"])
        >= float(thresholds["min_history_over_shuffled_lower"]),
        "history_beats_initial": lower(history["current_over_initial"])
        >= float(thresholds["min_history_over_initial_lower"]),
    }
    answer = test_metrics["answer"]
    answer_floor = float(thresholds["min_history_state_accuracy_lower"])
    if not all(checks.values()):
        interpretation = "state_not_causally_locatable_at_history_endpoint"
    elif lower(answer["current_state"]) >= answer_floor:
        interpretation = "state_retained_to_answer_anchor"
    elif lower(answer["next_state"]) >= answer_floor:
        interpretation = "state_transformed_to_correct_output"
    elif lower(answer["final_on_start"]) >= answer_floor:
        interpretation = "state_replaced_by_final_on_start_shortcut"
    elif lower(answer["initial_state"]) >= answer_floor:
        interpretation = "initial_state_dominates_answer_anchor"
    else:
        interpretation = "state_not_preserved_to_answer_anchor"
    return {
        "schema_version": 1,
        "decoder": {
            "training_condition": "materialized and counterfactual answer anchors",
            "rank": int(decoders[selected]["basis"].shape[1]),
            "class_count": class_count,
        },
        "split_counts": {
            name: len(split[name]) for name in ("train", "validation", "test")
        },
        "layer_selection": {
            "selected": selected,
            "validation_scores": validation_scores,
        },
        "heldout": {"case_count": len(test_ids), "by_position": test_metrics},
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": all(checks.values()),
        },
        "interpretation": interpretation,
    }


def score_handoff_patches_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
    layer: int,
    answer_state: np.ndarray,
    history_state: np.ndarray,
    state_basis: np.ndarray,
    random_basis: np.ndarray,
) -> dict[str, Any]:
    """Move only the recipient's own earlier state component to its answer anchor."""
    prompt = render_transfer_prompts(tokenizer=tokenizer, case=case, config=config)[0]
    candidate_ids = candidate_token_ids(tokenizer, prompt["text"], state_symbols(case))
    delta = history_state - answer_state
    vectors = {
        "self_state": state_basis @ (state_basis.T @ delta),
        "random_self": random_basis @ (random_basis.T @ delta),
        "full_self": delta,
    }
    expected = int(case["next_state"])
    shortcut = apply_rule(
        case["final_rule"], int(case["initial_state"]), 2 ** int(case["bits"])
    )
    conditions = {}
    for mode in HANDOFF_MODES:
        replacement = torch.from_numpy(
            (answer_state + vectors[mode]).astype(np.float32)
        )[None, :]
        record = score_logits(
            patched_logits(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                patches={layer: ((-1,), replacement)},
            ),
            candidate_ids,
        )
        record.update(
            expected_next_state=expected,
            final_on_start_state=shortcut,
            is_expected_unconstrained=record["unconstrained_prediction"] == expected,
        )
        conditions[mode] = record
    return {
        "schema_version": 1,
        "id": case["id"],
        "layer": layer,
        "conditions": conditions,
    }


def summarize_handoff(
    *,
    cases: dict[str, dict[str, Any]],
    captures: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Report the prespecified held-out self-handoff causal test."""
    if not rows:
        raise ValueError("Cannot summarize empty self-handoff results")

    def probability(record: dict[str, Any], target: int) -> float:
        return float(record["final_candidate_probabilities"][target])

    shifts: dict[str, list[float]] = {mode: [] for mode in HANDOFF_MODES}
    shortcut_shifts: dict[str, list[float]] = {mode: [] for mode in HANDOFF_MODES}
    accuracies: dict[str, list[bool]] = {mode: [] for mode in HANDOFF_MODES}
    margin_improvements: list[float] = []
    for row in rows:
        case_id = str(row["id"])
        case = cases[case_id]
        correct = int(case["next_state"])
        shortcut = apply_rule(
            case["final_rule"],
            int(case["initial_state"]),
            2 ** int(case["bits"]),
        )
        baseline = captures[case_id]["conditions"]["compose"]
        base_correct = probability(baseline, correct)
        base_shortcut = probability(baseline, shortcut)
        for mode in HANDOFF_MODES:
            record = row["conditions"][mode]
            shifts[mode].append(probability(record, correct) - base_correct)
            shortcut_shifts[mode].append(
                probability(record, shortcut) - base_shortcut
            )
            accuracies[mode].append(bool(record["is_expected_unconstrained"]))
        state = row["conditions"]["self_state"]
        margin_improvements.append(
            (probability(state, correct) - probability(state, shortcut))
            - (base_correct - base_shortcut)
        )
    metrics = {
        mode: {
            "correct_probability_shift": bootstrap_mean_ci(
                shifts[mode], seed=1300 + index
            ),
            "shortcut_probability_shift": bootstrap_mean_ci(
                shortcut_shifts[mode], seed=1310 + index
            ),
            "accuracy": bootstrap_mean_ci(accuracies[mode], seed=1320 + index),
        }
        for index, mode in enumerate(HANDOFF_MODES)
    }
    state_over_random = bootstrap_mean_ci(
        [left - right for left, right in zip(shifts["self_state"], shifts["random_self"])],
        seed=1330,
    )
    margin = bootstrap_mean_ci(margin_improvements, seed=1331)
    thresholds = {**DEFAULT_CAUSAL_GATE, **gate}

    def lower(stat: dict[str, Any]) -> float:
        value = stat["ci95"][0]
        return float(value) if value is not None else float("-inf")

    checks = {
        "self_state_shift": lower(metrics["self_state"]["correct_probability_shift"])
        >= float(thresholds["min_self_shift_lower"]),
        "self_over_random": lower(state_over_random)
        >= float(thresholds["min_self_over_random_lower"]),
        "correct_over_shortcut": lower(margin)
        >= float(thresholds["min_margin_improvement_lower"]),
    }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "layer": int(rows[0]["layer"]),
        "metrics": metrics,
        "self_over_random": state_over_random,
        "correct_minus_shortcut_margin_improvement": margin,
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": all(checks.values()),
        },
    }
