from __future__ import annotations

import json

from src.experiments.depth_relief.state_handoff_programs import (
    build_test_programs,
)
from src.experiments.depth_relief.state_interface_generalization import (
    compare_state_interface_generalization,
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
