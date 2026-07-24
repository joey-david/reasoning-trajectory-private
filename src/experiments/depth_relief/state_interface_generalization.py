"""Paired OOD and rate analysis for state-interface generalization runs."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .state_handoff_data import (
    COMPUTE_MANIFEST_PATH,
    DATA_MANIFEST_PATH,
    TEST_PATH,
    read_programs,
)
from .state_handoff_evaluation import read_evaluation_cases
from .state_handoff_information import (
    conditional_entropy,
    discrete_entropy,
    mutual_information,
)
from .state_interface_contract import (
    interface_codebook_size,
    semantic_states_for_code,
)
from .state_interface_evaluation import read_interface_evaluation_cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _answer_correct(row: dict[str, Any], *, interface: bool) -> bool:
    if interface:
        result = row.get("predicted_final")
        return bool(result and result["is_expected_unconstrained"])
    return bool(
        row["conditions"]["one_pass_compose"]["is_expected_unconstrained"]
    )


def _quotient_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[str, int], list[tuple[int, ...]]] = defaultdict(list)
    for row in rows:
        groups[
            (str(row["program_context"]), int(row["current_state"]))
        ].append(tuple(int(value) for value in row["predicted_semantic_states"]))
    values = []
    for group in groups.values():
        for index, left in enumerate(group):
            values.extend(
                bool(left) and left == right for right in group[index + 1 :]
            )
    return bootstrap_mean_ci(values, seed=81_401)


def _information_metrics(
    rows: list[dict[str, Any]], semantic_count: int
) -> dict[str, Any]:
    pairs = [
        (int(row["current_state"]), int(row["predicted_code"]))
        for row in rows
        if row["predicted_code"] is not None
    ]
    state_information = mutual_information(pairs)
    state_given_code = conditional_entropy(pairs)
    support = len({state for state, _ in pairs})
    state_entropy = discrete_entropy(state for state, _ in pairs)
    denominator = math.log2(max(support - 1, 2))
    fano_error_lower = max(0.0, (state_given_code - 1.0) / denominator)
    return {
        "semantic_state_count": semantic_count,
        "observed_state_support": support,
        "state_entropy_bits": state_entropy,
        "state_information_bits": state_information,
        "state_information_fraction": (
            state_information / state_entropy if state_entropy else 0.0
        ),
        "state_given_code_bits": state_given_code,
        "fano_state_error_lower_bound": fano_error_lower,
    }


def _cell_keys(row: dict[str, Any]) -> tuple[tuple[str, str, int], ...]:
    base = (
        str(row.get("domain", "addition")),
        str(row.get("composition_split", "seen")),
        int(row["history_steps"]),
    )
    if bool(row.get("proof_composition_active", False)):
        return (
            base,
            (
                "horn_proof_causal_conjunction",
                base[1],
                base[2],
            ),
        )
    return (base,)


def _conditional_transition_metrics(
    *,
    rows: list[dict[str, Any]],
    programs: dict[str, dict[str, Any]],
    condition: str,
    interface_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Score saved recursive steps against the state they actually received."""
    values: defaultdict[str, list[bool]] = defaultdict(list)
    clusters: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        case = programs[str(row["id"])]
        block_size = int(row.get("block_size", 1))
        for index, step in enumerate(row.get("steps", ())):
            if index == 0:
                continue
            prediction = step.get("unconstrained_prediction")
            predicted_states = (
                set()
                if prediction is None
                else set(
                    semantic_states_for_code(
                        condition=condition,
                        case=case,
                        code_index=int(prediction),
                        interface_config=interface_config,
                    )
                )
            )
            compatible_states: set[int] = set()
            for code in step["compatible_output_codes"]:
                compatible_states.update(
                    semantic_states_for_code(
                        condition=condition,
                        case=case,
                        code_index=int(code),
                        interface_config=interface_config,
                    )
                )
            correct = bool(predicted_states) and predicted_states == compatible_states
            start = index * block_size
            active = (
                int(case["state_path"][start])
                != int(case["state_path"][start + block_size])
            )
            for name in ("all", "active" if active else "dormant"):
                values[name].append(correct)
                clusters[name].append(str(row["program_context"]))

    return {
        "conditional_semantic_transition_accuracy": cluster_bootstrap_mean_ci(
            values["all"], clusters["all"], seed=seed
        ),
        "active_conditional_semantic_transition_accuracy": (
            cluster_bootstrap_mean_ci(
                values["active"], clusters["active"], seed=seed + 1
            )
        ),
        "dormant_conditional_semantic_transition_accuracy": (
            cluster_bootstrap_mean_ci(
                values["dormant"], clusters["dormant"], seed=seed + 2
            )
        ),
    }


def compare_state_interface_generalization(run_path: Path) -> dict[str, Any]:
    """Compare matched interface and one-pass controls on every OOD cell."""
    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    comparison = config.get("state_interface_generalization", {})
    control_run = Path(str(comparison["outcome_control_run"]))
    if _sha256(run_path / TEST_PATH) != _sha256(control_run / TEST_PATH):
        raise ValueError("Interface and outcome control test programs differ")
    if _sha256(run_path / DATA_MANIFEST_PATH) != _sha256(
        control_run / DATA_MANIFEST_PATH
    ):
        raise ValueError("Interface and outcome control data manifests differ")

    programs = {
        str(row["id"]): row for row in read_programs(run_path / TEST_PATH)
    }
    semantic_count = 2 ** int(next(iter(programs.values()))["bits"])
    outcome = {
        str(row["id"]): row
        for row in read_evaluation_cases(control_run, "outcome_only")
    }
    if set(outcome) != set(programs):
        raise RuntimeError("Outcome-control evaluation is incomplete")

    interface_compute = json.loads(
        (run_path / COMPUTE_MANIFEST_PATH).read_text()
    )
    outcome_compute = json.loads(
        (control_run / COMPUTE_MANIFEST_PATH).read_text()
    )["conditions"]["outcome_only"]
    conditions = tuple(str(value) for value in experiment["conditions"])
    interface_config = experiment.get("interfaces", {})
    cells: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        interface_rows = read_interface_evaluation_cases(run_path, condition)
        indexed = {str(row["id"]): row for row in interface_rows}
        if set(indexed) != set(programs):
            raise RuntimeError(f"Interface evaluation is incomplete for {condition}")
        compute = interface_compute["conditions"][condition]
        for key in (
            "semantic_programs",
            "forward_passes",
            "fixed_padding_compute_tokens",
            "active_input_tokens",
            "target_tokens",
        ):
            if compute[key] != outcome_compute[key]:
                raise ValueError(f"Matched compute differs for {condition}: {key}")
        for domain, composition_split, horizon in sorted(
            {key for row in interface_rows for key in _cell_keys(row)}
        ):
            selected = [
                row
                for row in interface_rows
                if (domain, composition_split, horizon) in _cell_keys(row)
            ]
            ids = [str(row["id"]) for row in selected]
            clusters = [str(row["program_context"]) for row in selected]
            interface_values = [
                _answer_correct(indexed[case_id], interface=True)
                for case_id in ids
            ]
            outcome_values = [
                _answer_correct(outcome[case_id], interface=False)
                for case_id in ids
            ]
            semantic_values = [
                bool(indexed[case_id]["semantic_state_correct"])
                for case_id in ids
            ]
            cell = {
                "domain": domain,
                "composition_split": composition_split,
                "history_steps": horizon,
                "case_count": len(ids),
                "program_context_count": len(set(clusters)),
                "interface_answer_accuracy": cluster_bootstrap_mean_ci(
                    interface_values, clusters, seed=81_100 + horizon
                ),
                "outcome_answer_accuracy": cluster_bootstrap_mean_ci(
                    outcome_values, clusters, seed=81_200 + horizon
                ),
                "interface_minus_outcome": cluster_bootstrap_mean_ci(
                    [
                        int(left) - int(right)
                        for left, right in zip(interface_values, outcome_values)
                    ],
                    clusters,
                    seed=81_300 + horizon,
                ),
                "semantic_state_accuracy": cluster_bootstrap_mean_ci(
                    semantic_values, clusters, seed=81_400 + horizon
                ),
                "same_state_quotient_agreement": _quotient_agreement(selected),
                **_conditional_transition_metrics(
                    rows=selected,
                    programs=programs,
                    condition=condition,
                    interface_config=interface_config,
                    seed=81_500 + horizon,
                ),
                **_information_metrics(selected, semantic_count),
            }
            cells[
                f"{condition}/{domain}/{composition_split}/h{horizon}"
            ] = cell

    primary = str(comparison.get("primary_condition", conditions[-1]))
    gate = comparison.get("gate", {})
    gate_domains = tuple(
        str(value)
        for value in gate.get(
            "domains",
            sorted({str(case.get("domain", "addition")) for case in programs.values()}),
        )
    )
    gate_split = str(gate.get("composition_split", "heldout"))
    gate_horizon = int(gate.get("history_steps", 16))
    selected_gate_cells = [
        cells[f"{primary}/{domain}/{gate_split}/h{gate_horizon}"]
        for domain in gate_domains
    ]
    min_accuracy = float(gate.get("min_accuracy", 0.80))
    min_improvement = float(gate.get("min_improvement", 0.10))
    min_information_fraction = float(
        gate.get("min_state_information_fraction", 0.90)
    )
    min_quotient_agreement = float(
        gate.get("min_quotient_agreement", 0.90)
    )
    checks = {
        "heldout_accuracy": all(
            float(cell["interface_answer_accuracy"]["mean"]) >= min_accuracy
            for cell in selected_gate_cells
        ),
        "matched_control_improvement": all(
            float(cell["interface_minus_outcome"]["mean"]) >= min_improvement
            and float(cell["interface_minus_outcome"]["ci95"][0]) > 0
            for cell in selected_gate_cells
        ),
        "state_information": all(
            float(cell["state_information_fraction"]) >= min_information_fraction
            for cell in selected_gate_cells
        ),
        "quotient_agreement": all(
            float(cell["same_state_quotient_agreement"]["mean"])
            >= min_quotient_agreement
            for cell in selected_gate_cells
        ),
    }
    rate_control = comparison.get("rate_control_condition")
    if rate_control:
        expected = min(
            1.0,
            interface_codebook_size(
                str(rate_control), experiment.get("interfaces", {})
            )
            / semantic_count,
        )
        tolerance = float(gate.get("rate_control_tolerance", 0.05))
        rate_cells = [
            cells[f"{rate_control}/{domain}/{gate_split}/h{gate_horizon}"]
            for domain in gate_domains
        ]
        checks["rate_control_calibration"] = all(
            abs(float(cell["interface_answer_accuracy"]["mean"]) - expected)
            <= tolerance
            for cell in rate_cells
        )

    result = {
        "schema_version": 1,
        "interface_run": str(run_path),
        "outcome_control_run": str(control_run),
        "semantic_state_count": semantic_count,
        "semantic_state_entropy_bits": math.log2(semantic_count),
        "conditions": list(conditions),
        "matched_forward_passes_and_compute": True,
        "cells": cells,
        "gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "primary_condition": primary,
            "selected_domains": list(gate_domains),
            "composition_split": gate_split,
            "history_steps": gate_horizon,
            "checks": checks,
            "thresholds": {
                "min_accuracy": min_accuracy,
                "min_improvement": min_improvement,
                "min_state_information_fraction": min_information_fraction,
                "min_quotient_agreement": min_quotient_agreement,
            },
        },
    }
    write_json(run_path / "evaluation/generalization_summary.json", result)
    return result
