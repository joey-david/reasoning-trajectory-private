"""Information-theoretic audits of history-to-state representations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .decoding import (
    calibrate_temperature,
    conditional_label_entropy,
    decoder_logits,
    decoder_point,
    decoder_report,
    fit_centroid_decoder,
    label_entropy,
)
from .handoff import trace_position


def _position_index(row: dict[str, Any], key: str, name: str) -> int:
    positions = {
        str(position["name"]): index
        for index, position in enumerate(row[key])
    }
    if name not in positions:
        raise ValueError(f"Missing {name!r} in {key}")
    return positions[name]


def abstraction_values(
    *,
    source: str,
    case_ids: list[str],
    cases: dict[str, dict[str, Any]],
    captures: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    """Stack one all-layer state-carrying representation per semantic case."""
    values = []
    for case_id in case_ids:
        arrays = activations[case_id]
        capture = captures[case_id]
        if source == "implicit_history":
            name = f"history_step_{int(cases[case_id]['history_steps'])}"
            value = trace_position(arrays, capture, name)
        elif source == "implicit_synthesis":
            value = arrays["synthesize_trace"][0]
        elif source == "explicit_state":
            index = _position_index(capture, "update_positions", "state")
            value = arrays["update_trace"][index]
        else:
            raise ValueError(f"Unknown abstraction source: {source!r}")
        values.append(np.asarray(value, dtype=np.float16))
    return np.stack(values)


def _residualize_state(
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.stack(
        [train[train_labels == label].mean(axis=0) for label in range(class_count)]
    )
    return (
        train - centroids[train_labels],
        validation - centroids[validation_labels],
        test - centroids[test_labels],
    )


def _analyze_layer_bank(
    *,
    train_values: np.ndarray,
    validation_values: np.ndarray,
    test_values: np.ndarray,
    train_states: np.ndarray,
    validation_states: np.ndarray,
    test_states: np.ndarray,
    train_paths: np.ndarray,
    validation_paths: np.ndarray,
    test_paths: np.ndarray,
    test_clusters: np.ndarray,
    state_count: int,
    path_count: int,
    rank: int,
    seed: int,
) -> dict[str, Any]:
    layer_count = int(train_values.shape[1])
    rng = np.random.default_rng(seed)
    shuffled_states = train_states.copy()
    rng.shuffle(shuffled_states)
    validation_state_entropy = label_entropy(validation_states)
    test_state_entropy = label_entropy(test_states)
    validation_path_entropy = conditional_label_entropy(
        validation_paths, validation_states
    )
    test_path_entropy = conditional_label_entropy(test_paths, test_states)
    curves: dict[str, Any] = {}
    fitted: list[
        tuple[
            dict[str, np.ndarray],
            float,
            dict[str, np.ndarray],
            float,
            dict[str, np.ndarray],
            float,
        ]
    ] = []
    for layer in range(layer_count):
        train = train_values[:, layer].astype(np.float32)
        validation = validation_values[:, layer].astype(np.float32)
        test = test_values[:, layer].astype(np.float32)
        state_decoder = fit_centroid_decoder(
            train, train_states, class_count=state_count, rank=rank
        )
        state_temperature = calibrate_temperature(
            decoder_logits(state_decoder, validation), validation_states
        )
        shuffled_decoder = fit_centroid_decoder(
            train, shuffled_states, class_count=state_count, rank=rank
        )
        shuffled_temperature = calibrate_temperature(
            decoder_logits(shuffled_decoder, validation), validation_states
        )
        residual_train, residual_validation, residual_test = _residualize_state(
            train,
            validation,
            test,
            train_states,
            validation_states,
            test_states,
            state_count,
        )
        path_decoder = fit_centroid_decoder(
            residual_train,
            train_paths,
            class_count=path_count,
            rank=min(rank, path_count - 1),
        )
        path_temperature = calibrate_temperature(
            decoder_logits(path_decoder, residual_validation), validation_paths
        )
        validation_state = decoder_point(
            state_decoder,
            validation,
            validation_states,
            class_count=state_count,
            temperature=state_temperature,
            entropy_bits=validation_state_entropy,
        )
        validation_path = decoder_point(
            path_decoder,
            residual_validation,
            validation_paths,
            class_count=path_count,
            temperature=path_temperature,
            entropy_bits=validation_path_entropy,
        )
        validation_shuffled = decoder_point(
            shuffled_decoder,
            validation,
            validation_states,
            class_count=state_count,
            temperature=shuffled_temperature,
            entropy_bits=validation_state_entropy,
        )
        test_state = decoder_point(
            state_decoder,
            test,
            test_states,
            class_count=state_count,
            temperature=state_temperature,
            entropy_bits=test_state_entropy,
        )
        test_path = decoder_point(
            path_decoder,
            residual_test,
            test_paths,
            class_count=path_count,
            temperature=path_temperature,
            entropy_bits=test_path_entropy,
        )
        state_normalized = validation_state["information_lower_bound_bits"] / max(
            validation_state_entropy, 1e-8
        )
        path_normalized = max(
            0.0, validation_path["information_lower_bound_bits"]
        ) / max(validation_path_entropy, 1e-8)
        curves[str(layer)] = {
            "validation": {
                "state": validation_state,
                "path_given_state": validation_path,
                "shuffled_state": validation_shuffled,
                "selection_score": float(state_normalized - path_normalized),
            },
            "test": {"state": test_state, "path_given_state": test_path},
        }
        fitted.append(
            (
                state_decoder,
                state_temperature,
                shuffled_decoder,
                shuffled_temperature,
                path_decoder,
                path_temperature,
            )
        )
    selected = max(
        range(layer_count),
        key=lambda layer: (
            curves[str(layer)]["validation"]["selection_score"],
            -layer,
        ),
    )
    (
        decoder,
        temperature,
        shuffled_decoder,
        shuffled_temperature,
        path_decoder,
        path_temperature,
    ) = fitted[selected]
    _, _, selected_residual_test = _residualize_state(
        train_values[:, selected].astype(np.float32),
        validation_values[:, selected].astype(np.float32),
        test_values[:, selected].astype(np.float32),
        train_states,
        validation_states,
        test_states,
        state_count,
    )
    return {
        "selected_layer": selected,
        "layer_curves": curves,
        "selected_test": {
            "state": decoder_report(
                decoder,
                test_values[:, selected].astype(np.float32),
                test_states,
                class_count=state_count,
                temperature=temperature,
                seed=seed + 100,
                clusters=test_clusters,
                entropy_bits=test_state_entropy,
            ),
            "path_given_state": decoder_report(
                path_decoder,
                selected_residual_test,
                test_paths,
                class_count=path_count,
                temperature=path_temperature,
                seed=seed + 102,
                clusters=test_clusters,
                entropy_bits=test_path_entropy,
            ),
            "shuffled_state": decoder_report(
                shuffled_decoder,
                test_values[:, selected].astype(np.float32),
                test_states,
                class_count=state_count,
                temperature=shuffled_temperature,
                seed=seed + 104,
                clusters=test_clusters,
                entropy_bits=test_state_entropy,
            ),
        },
        "selected_decoder": decoder,
        "selected_temperature": temperature,
    }


def analyze_matched_history_information(
    *,
    cases: dict[str, dict[str, Any]],
    captures: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, np.ndarray]],
    rank: int,
    seed: int,
) -> dict[str, Any]:
    """Map state formation and conditional path leakage across token and layer."""
    splits = {
        name: [
            case_id
            for case_id, case in cases.items()
            if case["abstraction_split"] == name
        ]
        for name in ("train", "validation", "test")
    }
    state_count = 2 ** int(next(iter(cases.values()))["bits"])
    path_count = len({int(case["path_code"]) for case in cases.values()})

    def labels(ids: list[str], key: str) -> np.ndarray:
        return np.asarray([int(cases[case_id][key]) for case_id in ids])

    state_labels = {
        name: labels(ids, "current_state") for name, ids in splits.items()
    }
    path_labels = {name: labels(ids, "path_code") for name, ids in splits.items()}
    group_labels = {
        name: np.asarray([str(cases[case_id]["abstraction_group"]) for case_id in ids])
        for name, ids in splits.items()
    }
    sources: dict[str, Any] = {}
    source_values: dict[str, dict[str, np.ndarray]] = {}
    for source_index, source in enumerate(
        ("implicit_history", "implicit_synthesis", "explicit_state")
    ):
        values = {
            name: abstraction_values(
                source=source,
                case_ids=ids,
                cases=cases,
                captures=captures,
                activations=activations,
            )
            for name, ids in splits.items()
        }
        source_values[source] = values
        sources[source] = _analyze_layer_bank(
            train_values=values["train"],
            validation_values=values["validation"],
            test_values=values["test"],
            train_states=state_labels["train"],
            validation_states=state_labels["validation"],
            test_states=state_labels["test"],
            train_paths=path_labels["train"],
            validation_paths=path_labels["validation"],
            test_paths=path_labels["test"],
            test_clusters=group_labels["test"],
            state_count=state_count,
            path_count=path_count,
            rank=rank,
            seed=seed + 1000 * source_index,
        )

    transfer_matrix: dict[str, Any] = {}
    for source_index, source in enumerate(sources):
        layer = int(sources[source]["selected_layer"])
        decoder = sources[source].pop("selected_decoder")
        temperature = float(sources[source].pop("selected_temperature"))
        transfer_matrix[source] = {
            target: decoder_report(
                decoder,
                source_values[target]["test"][:, layer].astype(np.float32),
                state_labels["test"],
                class_count=state_count,
                temperature=temperature,
                seed=seed + 5000 + 100 * source_index + target_index,
                clusters=group_labels["test"],
                entropy_bits=label_entropy(state_labels["test"]),
            )
            for target_index, target in enumerate(source_values)
        }

    formation: dict[str, Any] = {}
    for history_steps in sorted(
        {int(case["history_steps"]) for case in cases.values()}
    ):
        formation[str(history_steps)] = {}
        horizon_ids = {
            split: [
                case_id
                for case_id in ids
                if int(cases[case_id]["history_steps"]) == history_steps
            ]
            for split, ids in splits.items()
        }
        positions = [
            *[f"history_step_{index}" for index in range(1, history_steps + 1)],
            "history_joint",
            "final_rule",
            "answer",
        ]
        for position_index, position in enumerate(positions):
            if position == "history_joint":
                joint_positions = [
                    "start",
                    *[
                        f"history_step_{index}"
                        for index in range(1, history_steps + 1)
                    ],
                ]
                values = {
                    split: np.stack(
                        [
                            np.concatenate(
                                [
                                    trace_position(
                                        activations[case_id],
                                        captures[case_id],
                                        joint_position,
                                    )
                                    for joint_position in joint_positions
                                ],
                                axis=1,
                            )
                            for case_id in ids
                        ]
                    )
                    for split, ids in horizon_ids.items()
                }
            else:
                values = {
                    split: np.stack(
                        [
                            trace_position(
                                activations[case_id], captures[case_id], position
                            )
                            for case_id in ids
                        ]
                    )
                    for split, ids in horizon_ids.items()
                }

            def position_states(ids: list[str]) -> np.ndarray:
                if position.startswith("history_step_"):
                    step = int(position.rsplit("_", 1)[1])
                    return np.asarray(
                        [int(cases[case_id]["state_path"][step]) for case_id in ids]
                    )
                key = "next_state" if position == "answer" else "current_state"
                return labels(ids, key)

            position_state_labels = {
                split: position_states(ids) for split, ids in horizon_ids.items()
            }
            result = _analyze_layer_bank(
                train_values=values["train"],
                validation_values=values["validation"],
                test_values=values["test"],
                train_states=position_state_labels["train"],
                validation_states=position_state_labels["validation"],
                test_states=position_state_labels["test"],
                train_paths=labels(horizon_ids["train"], "path_code"),
                validation_paths=labels(horizon_ids["validation"], "path_code"),
                test_paths=labels(horizon_ids["test"], "path_code"),
                test_clusters=np.asarray(
                    [
                        str(cases[case_id]["abstraction_group"])
                        for case_id in horizon_ids["test"]
                    ]
                ),
                state_count=state_count,
                path_count=path_count,
                rank=rank,
                seed=seed + 10_000 + history_steps * 1000 + position_index * 100,
            )
            result.pop("selected_decoder")
            result.pop("selected_temperature")
            formation[str(history_steps)][position] = result

    implicit = sources["implicit_history"]["selected_test"]["state"]["accuracy"]
    explicit_to_implicit = transfer_matrix["explicit_state"]["implicit_history"][
        "accuracy"
    ]
    joint_lowers = {
        horizon: float(
            positions["history_joint"]["selected_test"]["state"]["accuracy"][
                "ci95"
            ][0]
        )
        for horizon, positions in formation.items()
    }
    if float(implicit["ci95"][0]) >= 0.30:
        code_diagnosis = "history_invariant_endpoint_state_detected"
    elif max(joint_lowers.values()) >= 0.30:
        code_diagnosis = "state_only_jointly_decodable_across_history_tokens"
    else:
        code_diagnosis = "no_heldout_state_code_detected"
    alignment_diagnosis = (
        "explicit_and_implicit_coordinates_align"
        if float(explicit_to_implicit["ci95"][0]) >= 0.30
        else "explicit_and_implicit_coordinates_do_not_align"
    )
    return {
        "schema_version": 1,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "program_context_counts": {
            name: int(len(np.unique(values)))
            for name, values in group_labels.items()
        },
        "inference_unit": "program context",
        "state_count": state_count,
        "path_count": path_count,
        "selection": "maximize validation normalized state information minus conditional path information",
        "sources": sources,
        "decoder_transfer_matrix": transfer_matrix,
        "formation_by_horizon": formation,
        "diagnosis": {
            "implicit_code": code_diagnosis,
            "coordinate_alignment": alignment_diagnosis,
            "joint_history_accuracy_lower_by_horizon": joint_lowers,
        },
    }
