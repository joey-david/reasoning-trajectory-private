from __future__ import annotations

import json

from src.experiments.depth_relief.state_handoff_programs import (
    build_test_programs,
)
from src.experiments.depth_relief.state_interface_generalization import (
    _conditional_transition_metrics,
    _information_metrics,
    compare_state_interface_generalization,
)
from src.experiments.depth_relief.state_interface_replication import (
    compare_state_interface_replications,
)


def _write_jsonl(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def test_generalization_analysis_is_paired_and_artifact_only(tmp_path) -> None:
    interface_run = tmp_path / "interface"
    control_run = tmp_path / "control"
    interface_run.mkdir()
    control_run.mkdir()
    (interface_run / "config.yaml").write_text(
        f"""
state_handoff_training:
  conditions: [redundant_4bit]
state_interface_generalization:
  outcome_control_run: {control_run}
  primary_condition: redundant_4bit
  gate:
    domains: [mixed_algebra]
    composition_split: heldout
    history_steps: 16
    min_accuracy: 0.9
    min_improvement: 0.5
    min_state_information_fraction: 0.9
    min_quotient_agreement: 0.9
""".strip()
        + "\n"
    )
    (control_run / "config.yaml").write_text(
        "state_handoff_training:\n  conditions: [outcome_only]\n"
    )
    programs = build_test_programs(
        horizons=(16,),
        context_count=2,
        paths_per_state=2,
        width=3,
        seed=43,
        dataset={
            "domain": "mixed_algebra",
            "test_composition_splits": ["seen", "heldout"],
        },
    )
    _write_jsonl(interface_run / "evaluation/test_programs.jsonl", programs)
    _write_jsonl(control_run / "evaluation/test_programs.jsonl", programs)
    manifest = {"schema_version": 1, "shared": True}
    for run in (interface_run, control_run):
        path = run / "training/data/manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    budget = {
        "semantic_programs": 32,
        "forward_passes": 64,
        "fixed_padding_compute_tokens": 16_384,
        "active_input_tokens": 16_384,
        "target_tokens": 64,
    }
    interface_compute = {
        "conditions": {"redundant_4bit": budget},
    }
    outcome_compute = {
        "conditions": {"outcome_only": budget},
    }
    for run, compute in (
        (interface_run, interface_compute),
        (control_run, outcome_compute),
    ):
        path = run / "training/compute_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(compute, sort_keys=True) + "\n")

    interface_rows = []
    outcome_rows = []
    for case in programs:
        state = int(case["current_state"])
        code = 2 * state + int(case["path_code"]) % 2
        interface_rows.append(
            {
                "id": case["id"],
                "bits": 3,
                "history_steps": 16,
                "program_context": case["program_context"],
                "current_state": state,
                "domain": "mixed_algebra",
                "composition_split": case["composition_split"],
                "predicted_code": code,
                "predicted_semantic_states": [state],
                "semantic_state_correct": True,
                "predicted_final": {"is_expected_unconstrained": True},
            }
        )
        outcome_rows.append(
            {
                "id": case["id"],
                "history_steps": 16,
                "program_context": case["program_context"],
                "domain": "mixed_algebra",
                "composition_split": case["composition_split"],
                "conditions": {
                    "one_pass_compose": {
                        "is_expected_unconstrained": False,
                    }
                },
            }
        )
    _write_jsonl(
        interface_run
        / "evaluation/interfaces/redundant_4bit/cases.jsonl",
        interface_rows,
    )
    _write_jsonl(
        control_run / "evaluation/outcome_only/cases.jsonl",
        outcome_rows,
    )

    first = compare_state_interface_generalization(interface_run)
    second = compare_state_interface_generalization(interface_run)

    assert first == second
    assert first["gate"]["status"] == "passed"
    cell = first["cells"]["redundant_4bit/mixed_algebra/heldout/h16"]
    assert cell["interface_answer_accuracy"]["mean"] == 1.0
    assert cell["outcome_answer_accuracy"]["mean"] == 0.0
    assert cell["state_information_bits"] == 3.0
    assert cell["same_state_quotient_agreement"]["mean"] == 1.0


def test_replication_summary_requires_identical_three_seed_programs(
    tmp_path,
) -> None:
    runs = [tmp_path / f"seed{seed}" for seed in range(3)]
    for index, run in enumerate(runs):
        run.mkdir()
        config = {
            "model": {"name": "tiny", "revision": "fixed"},
        }
        if index == 0:
            config["state_interface_replication"] = {
                "runs": [str(value) for value in runs],
                "primary_condition": "redundant_4bit",
                "domain": "mixed_algebra",
                "composition_split": "heldout",
                "history_steps": 16,
                "min_accuracy": 0.8,
                "min_improvement": 0.1,
            }
        (run / "config.yaml").write_text(json.dumps(config))
        _write_jsonl(
            run / "evaluation/test_programs.jsonl",
            [{"id": "shared"}],
        )
        manifest = run / "training/data/manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{"shared": true}\n')
        summary = {
            "gate": {"status": "passed"},
            "cells": {
                "redundant_4bit/mixed_algebra/heldout/h16": {
                    "interface_answer_accuracy": {
                        "mean": 0.90 + 0.01 * index
                    },
                    "outcome_answer_accuracy": {"mean": 0.50},
                    "interface_minus_outcome": {
                        "mean": 0.40 + 0.01 * index
                    },
                    "semantic_state_accuracy": {"mean": 0.92},
                    "same_state_quotient_agreement": {"mean": 0.95},
                    "state_information_fraction": 0.96,
                    "state_given_code_bits": 0.12,
                    "fano_state_error_lower_bound": 0.0,
                }
            },
        }
        path = run / "evaluation/generalization_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary))

    first = compare_state_interface_replications(runs[0])
    second = compare_state_interface_replications(runs[0])
    assert first == second
    assert first["gate"]["status"] == "passed"
    assert first["metrics"]["interface_answer_accuracy"]["per_seed"] == [
        0.9,
        0.91,
        0.92,
    ]


def test_information_fraction_uses_the_selected_stratum_entropy() -> None:
    rows = [
        {"current_state": state, "predicted_code": index}
        for index, state in enumerate((3, 5, 6, 7))
        for _ in range(2)
    ]
    metrics = _information_metrics(rows, semantic_count=8)
    assert metrics["semantic_state_count"] == 8
    assert metrics["observed_state_support"] == 4
    assert metrics["state_entropy_bits"] == 2.0
    assert metrics["state_information_bits"] == 2.0
    assert metrics["state_information_fraction"] == 1.0


def test_conditional_transition_metrics_ignore_redundant_code_variant() -> None:
    case = {
        "id": "case",
        "bits": 3,
        "path_code": 0,
        "state_path": [0, 1, 2],
    }
    metrics = _conditional_transition_metrics(
        rows=[
            {
                "id": "case",
                "program_context": "context",
                "block_size": 1,
                "steps": [
                    {
                        "unconstrained_prediction": 2,
                        "compatible_output_codes": [2],
                    },
                    {
                        "unconstrained_prediction": 5,
                        "compatible_output_codes": [4],
                    },
                ],
            }
        ],
        programs={"case": case},
        condition="redundant_4bit",
        interface_config={},
        seed=1,
    )
    assert metrics["conditional_semantic_transition_accuracy"]["mean"] == 1.0
    assert (
        metrics["active_conditional_semantic_transition_accuracy"]["mean"]
        == 1.0
    )
