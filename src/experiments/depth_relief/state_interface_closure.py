"""Paired comparison of transition-closure and endpoint-only fine-tuning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import cluster_bootstrap_mean_ci
from .state_handoff_data import (
    COMPUTE_MANIFEST_PATH,
    DATA_MANIFEST_PATH,
    TEST_PATH,
    read_programs,
)
from .state_interface_data import semantic_states_for_code
from .state_interface_evaluation import read_interface_evaluation_cases


def _semantic_local_accuracy(
    row: dict[str, Any],
    case: dict[str, Any],
    condition: str,
    interface_config: dict[str, Any],
) -> float:
    values = []
    for step in row["steps"]:
        predicted = step.get("unconstrained_prediction")
        if predicted is None:
            values.append(False)
            continue
        expected = int(step["local_expected_code"])
        values.append(
            semantic_states_for_code(
                condition=condition,
                case=case,
                code_index=int(predicted),
                interface_config=interface_config,
            )
            == semantic_states_for_code(
                condition=condition,
                case=case,
                code_index=expected,
                interface_config=interface_config,
            )
        )
    return sum(values) / len(values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_closure_finetuning(run_path: Path) -> dict[str, Any]:
    """Compare matched fine-tunes on shared test cases and apply the closure gate."""
    config = load_config(run_path)
    comparison = config.get("state_interface_closure_comparison", {})
    control_run = Path(str(comparison["control_run"]))
    experiment = config.get("state_handoff_training", {})
    control_experiment = load_config(control_run).get("state_handoff_training", {})
    conditions = tuple(str(value) for value in experiment["conditions"])
    if conditions != tuple(str(value) for value in control_experiment["conditions"]):
        raise ValueError("Closure and endpoint controls must use identical conditions")
    data_hashes = {
        "closure": _sha256(run_path / DATA_MANIFEST_PATH),
        "control": _sha256(control_run / DATA_MANIFEST_PATH),
        "closure_test": _sha256(run_path / TEST_PATH),
        "control_test": _sha256(control_run / TEST_PATH),
    }
    if data_hashes["closure"] != data_hashes["control"]:
        raise ValueError("Closure and endpoint data manifests differ")
    if data_hashes["closure_test"] != data_hashes["control_test"]:
        raise ValueError("Closure and endpoint test programs differ")
    closure_compute = json.loads((run_path / COMPUTE_MANIFEST_PATH).read_text())
    control_compute = json.loads((control_run / COMPUTE_MANIFEST_PATH).read_text())
    for condition in conditions:
        left = closure_compute["conditions"][condition]
        right = control_compute["conditions"][condition]
        for key in (
            "semantic_programs",
            "forward_passes",
            "fixed_padding_compute_tokens",
            "target_tokens",
        ):
            if left[key] != right[key]:
                raise ValueError(f"Fine-tune compute differs for {condition}: {key}")

    programs = {
        str(case["id"]): case for case in read_programs(run_path / TEST_PATH)
    }
    summaries = {}
    for condition in conditions:
        closure_rows = {
            str(row["id"]): row
            for row in read_interface_evaluation_cases(run_path, condition)
        }
        control_rows = {
            str(row["id"]): row
            for row in read_interface_evaluation_cases(control_run, condition)
        }
        shared = sorted(set(closure_rows) & set(control_rows))
        if len(shared) != len(programs):
            raise ValueError(f"Paired closure evaluation is incomplete for {condition}")
        by_horizon = {}
        for horizon in sorted(
            {int(programs[case_id]["history_steps"]) for case_id in shared}
        ):
            ids = [
                case_id
                for case_id in shared
                if int(programs[case_id]["history_steps"]) == horizon
            ]
            clusters = [str(programs[case_id]["program_context"]) for case_id in ids]
            closure_answer = [
                bool(
                    closure_rows[case_id]["predicted_final"]
                    and closure_rows[case_id]["predicted_final"][
                        "is_expected_unconstrained"
                    ]
                )
                for case_id in ids
            ]
            control_answer = [
                bool(
                    control_rows[case_id]["predicted_final"]
                    and control_rows[case_id]["predicted_final"][
                        "is_expected_unconstrained"
                    ]
                )
                for case_id in ids
            ]
            closure_local = [
                _semantic_local_accuracy(
                    closure_rows[case_id],
                    programs[case_id],
                    condition,
                    experiment.get("interfaces", {}),
                )
                for case_id in ids
            ]
            control_local = [
                _semantic_local_accuracy(
                    control_rows[case_id],
                    programs[case_id],
                    condition,
                    control_experiment.get("interfaces", {}),
                )
                for case_id in ids
            ]
            by_horizon[str(horizon)] = {
                "answer_accuracy": {
                    "closure": sum(closure_answer) / len(ids),
                    "endpoint_control": sum(control_answer) / len(ids),
                },
                "answer_difference": cluster_bootstrap_mean_ci(
                    [
                        int(left) - int(right)
                        for left, right in zip(closure_answer, control_answer)
                    ],
                    clusters,
                    seed=74_000 + horizon,
                ),
                "local_semantic_closure": {
                    "closure": sum(closure_local) / len(ids),
                    "endpoint_control": sum(control_local) / len(ids),
                },
                "local_closure_difference": cluster_bootstrap_mean_ci(
                    [
                        left - right
                        for left, right in zip(closure_local, control_local)
                    ],
                    clusters,
                    seed=74_100 + horizon,
                ),
            }
        summaries[condition] = {"case_count": len(shared), "by_horizon": by_horizon}

    canonical = summaries["canonical_opaque"]["by_horizon"]
    thresholds = {
        "minimum_h8_improvement": float(
            comparison.get("minimum_h8_improvement", 0.10)
        ),
        "minimum_h16_improvement": float(
            comparison.get("minimum_h16_improvement", 0.10)
        ),
    }
    checks = {
        "h8_improvement": canonical["8"]["answer_difference"]["mean"]
        >= thresholds["minimum_h8_improvement"],
        "h8_positive_interval": canonical["8"]["answer_difference"]["ci95"][0] > 0,
        "h16_improvement": canonical["16"]["answer_difference"]["mean"]
        >= thresholds["minimum_h16_improvement"],
        "h16_positive_interval": canonical["16"]["answer_difference"]["ci95"][0] > 0,
    }
    result = {
        "schema_version": 1,
        "closure_run": str(run_path),
        "control_run": str(control_run),
        "data_hashes": data_hashes,
        "matched_compute": True,
        "conditions": summaries,
        "gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "thresholds": thresholds,
            "checks": checks,
        },
    }
    write_json(run_path / "evaluation/closure_comparison.json", result)
    return result
