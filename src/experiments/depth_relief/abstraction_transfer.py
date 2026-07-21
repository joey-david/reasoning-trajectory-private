"""Exploratory decoder transfer on the completed state-transfer captures."""

from __future__ import annotations

from typing import Any

import numpy as np

from .decoding import (
    calibrate_temperature,
    decoder_logits,
    decoder_point,
    decoder_report,
    fit_centroid_decoder,
)
from .handoff import trace_position


def analyze_existing_transfer_matrix(
    *,
    cases: dict[str, dict[str, Any]],
    split: dict[str, Any],
    captures: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, np.ndarray]],
    rank: int,
    seed: int,
) -> dict[str, Any]:
    """Distinguish implicit code absence from explicit/implicit misalignment."""
    state_count = 2 ** int(next(iter(cases.values()))["bits"])

    def implicit(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        values = np.stack(
            [
                trace_position(
                    activations[case_id],
                    captures[case_id],
                    f"history_step_{int(cases[case_id]['history_steps'])}",
                )
                for case_id in ids
            ]
        )
        labels = np.asarray(
            [int(cases[case_id]["current_state"]) for case_id in ids]
        )
        return values, labels

    def explicit(ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        values, labels = [], []
        for case_id in ids:
            case = cases[case_id]
            arrays = activations[case_id]
            values.extend([arrays["materialized"], arrays["counterfactual"]])
            labels.extend(
                [int(case["current_state"]), int(case["counterfactual_state"])]
            )
        return np.stack(values), np.asarray(labels)

    banks = {
        "implicit_history": {
            name: implicit(list(split[name]))
            for name in ("train", "validation", "test")
        },
        "explicit_answer": {
            name: explicit(list(split[name]))
            for name in ("train", "validation", "test")
        },
    }
    report: dict[str, Any] = {}
    for source_index, (source, bank) in enumerate(banks.items()):
        curves = {}
        fitted = []
        layer_count = int(bank["train"][0].shape[1])
        for layer in range(layer_count):
            decoder = fit_centroid_decoder(
                bank["train"][0][:, layer],
                bank["train"][1],
                class_count=state_count,
                rank=rank,
            )
            temperature = calibrate_temperature(
                decoder_logits(decoder, bank["validation"][0][:, layer]),
                bank["validation"][1],
            )
            curves[str(layer)] = decoder_point(
                decoder,
                bank["validation"][0][:, layer],
                bank["validation"][1],
                class_count=state_count,
                temperature=temperature,
            )
            fitted.append((decoder, temperature))
        selected = max(
            range(layer_count),
            key=lambda layer: (
                curves[str(layer)]["information_lower_bound_bits"],
                -layer,
            ),
        )
        decoder, temperature = fitted[selected]
        report[source] = {
            "selected_layer": selected,
            "validation_curve": curves,
            "heldout_transfer": {
                target: decoder_report(
                    decoder,
                    target_bank["test"][0][:, selected],
                    target_bank["test"][1],
                    class_count=state_count,
                    temperature=temperature,
                    seed=seed + source_index * 100 + target_index,
                )
                for target_index, (target, target_bank) in enumerate(banks.items())
            },
        }
    implicit_self = report["implicit_history"]["heldout_transfer"][
        "implicit_history"
    ]["accuracy"]
    explicit_cross = report["explicit_answer"]["heldout_transfer"][
        "implicit_history"
    ]["accuracy"]
    if float(implicit_self["ci95"][0]) >= 0.30 and float(
        explicit_cross["ci95"][0]
    ) < 0.30:
        interpretation = "prompt_specific_implicit_state"
    elif float(implicit_self["ci95"][0]) < 0.30:
        interpretation = "no_history_invariant_implicit_state_detected"
    else:
        interpretation = "explicit_and_implicit_coordinates_align"
    return {
        "schema_version": 1,
        "split_counts": {
            name: len(split[name]) for name in ("train", "validation", "test")
        },
        "state_count": state_count,
        "sources": report,
        "interpretation": interpretation,
        "caveat": (
            "The 94-case audit is exploratory; the matched-history assay is "
            "confirmatory."
        ),
    }
