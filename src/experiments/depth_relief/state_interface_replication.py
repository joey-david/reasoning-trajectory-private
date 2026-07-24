"""Aggregate prespecified state-interface results across training seeds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci
from .state_handoff_data import DATA_MANIFEST_PATH, TEST_PATH, read_programs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_state_interface_replications(run_path: Path) -> dict[str, Any]:
    """Require matching data and summarize one OOD cell across seeds."""
    config = load_config(run_path)
    replication = config["state_interface_replication"]
    runs = tuple(Path(str(value)) for value in replication["runs"])
    if len(runs) < 3:
        raise ValueError("Confirmation requires at least three seed runs")
    summaries = [
        json.loads(
            (member / "evaluation/generalization_summary.json").read_text()
        )
        for member in runs
    ]
    test_hashes = [_sha256(member / TEST_PATH) for member in runs]
    data_hashes = [_sha256(member / DATA_MANIFEST_PATH) for member in runs]
    program_contract = str(replication.get("program_contract", "identical"))
    if program_contract == "identical":
        programs_valid = (
            len(set(test_hashes)) == 1 and len(set(data_hashes)) == 1
        )
    elif program_contract == "disjoint":
        id_sets = [
            {str(row["id"]) for row in read_programs(member / TEST_PATH)}
            for member in runs
        ]
        programs_valid = (
            len(set(test_hashes)) == len(runs)
            and len(set(data_hashes)) == len(runs)
            and all(
                not id_sets[left] & id_sets[right]
                for left in range(len(id_sets))
                for right in range(left + 1, len(id_sets))
            )
        )
    else:
        raise ValueError(
            "Replication program_contract must be identical or disjoint"
        )
    if not programs_valid:
        raise ValueError(
            f"Replication runs violate their {program_contract} program contract"
        )
    models = [
        {
            "name": load_config(member)["model"]["name"],
            "revision": load_config(member)["model"].get("revision"),
        }
        for member in runs
    ]
    if len({(row["name"], row["revision"]) for row in models}) != 1:
        raise ValueError("Replication runs do not pin the same base model")

    primary = str(replication["primary_condition"])
    domain = str(replication["domain"])
    composition_split = str(replication.get("composition_split", "heldout"))
    horizon = int(replication.get("history_steps", 16))
    key = f"{primary}/{domain}/{composition_split}/h{horizon}"
    cells = [summary["cells"][key] for summary in summaries]
    metric_paths = {
        "interface_answer_accuracy": ("interface_answer_accuracy", "mean"),
        "outcome_answer_accuracy": ("outcome_answer_accuracy", "mean"),
        "interface_minus_outcome": ("interface_minus_outcome", "mean"),
        "semantic_state_accuracy": ("semantic_state_accuracy", "mean"),
        "same_state_quotient_agreement": (
            "same_state_quotient_agreement",
            "mean",
        ),
        "state_information_fraction": ("state_information_fraction",),
        "state_given_code_bits": ("state_given_code_bits",),
        "fano_state_error_lower_bound": ("fano_state_error_lower_bound",),
    }
    metrics = {}
    for name, path in metric_paths.items():
        values = []
        for cell in cells:
            value: Any = cell
            for part in path:
                value = value[part]
            values.append(float(value))
        metrics[name] = {
            "per_seed": values,
            **bootstrap_mean_ci(values, seed=82_100 + len(metrics)),
            "minimum": min(values),
            "maximum": max(values),
        }

    minimum_accuracy = float(replication.get("min_accuracy", 0.80))
    minimum_improvement = float(replication.get("min_improvement", 0.10))
    checks = {
        "three_or_more_seeds": len(runs) >= 3,
        f"{program_contract}_programs": True,
        "all_individual_gates_pass": all(
            summary["gate"]["status"] == "passed" for summary in summaries
        ),
        "minimum_seed_accuracy": (
            metrics["interface_answer_accuracy"]["minimum"]
            >= minimum_accuracy
        ),
        "minimum_seed_improvement": (
            metrics["interface_minus_outcome"]["minimum"]
            >= minimum_improvement
        ),
    }
    result = {
        "schema_version": 1,
        "runs": [str(member) for member in runs],
        "models": models,
        "program_contract": program_contract,
        "test_sha256": (
            test_hashes[0] if program_contract == "identical" else test_hashes
        ),
        "data_manifest_sha256": (
            data_hashes[0] if program_contract == "identical" else data_hashes
        ),
        "cell": {
            "condition": primary,
            "domain": domain,
            "composition_split": composition_split,
            "history_steps": horizon,
        },
        "metrics": metrics,
        "gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "thresholds": {
                "min_accuracy_each_seed": minimum_accuracy,
                "min_improvement_each_seed": minimum_improvement,
            },
        },
        "uncertainty_contract": (
            "Each run retains its context-clustered paired interval. The "
            "aggregate interval resamples the three independent training-seed "
            "point estimates and is reported with the full seed range."
        ),
    }
    write_json(run_path / "evaluation/replication_summary.json", result)
    return result
