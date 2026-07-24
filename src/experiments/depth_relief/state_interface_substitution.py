"""Cross-adapter tests for a shared discrete state interface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config

from .metrics import cluster_bootstrap_mean_ci
from .state_handoff_data import TEST_PATH, read_programs
from .state_handoff_evaluation import _load_evaluation_model
from .state_interface_contract import interface_code_symbols
from .state_interface_evaluation import (
    evaluate_interface_program_hf,
    read_interface_evaluation_cases,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_dir(run_path: Path, consumer_run: Path) -> Path:
    return run_path / "evaluation/substitution" / consumer_run.name


def _read_rows(run_path: Path, consumer_run: Path) -> list[dict[str, Any]]:
    path = _output_dir(run_path, consumer_run) / "cases.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate substitution IDs in {path}")
    return rows


def _balanced_prefix(
    cases: list[dict[str, Any]], maximum: int
) -> list[dict[str, Any]]:
    """Select complete cells across contexts before adding more paths."""
    ordered = sorted(
        cases,
        key=lambda case: (
            int(case["path_code"]),
            str(case["program_context"]),
            int(case["history_steps"]),
            int(case["current_state"]),
            str(case.get("composition_split", "seen")),
        ),
    )
    return ordered[:maximum]


def evaluate_interface_substitution(
    run_path: Path,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Use one adapter's producer and an independent adapter's consumer."""
    config = load_config(run_path)
    spec = config["state_interface_substitution"]
    condition = str(spec["condition"])
    consumer_run = Path(str(spec["consumer_run"]))
    consumer_config = load_config(consumer_run)
    if config["model"] != consumer_config["model"]:
        raise ValueError("Substitution requires the same pinned base model")
    if _sha256(run_path / TEST_PATH) != _sha256(consumer_run / TEST_PATH):
        raise ValueError("Substitution runs must share an exact test bank")
    interface_config = config["state_handoff_training"].get("interfaces", {})
    consumer_interface_config = consumer_config["state_handoff_training"].get(
        "interfaces", {}
    )
    if interface_code_symbols(
        condition, interface_config
    ) != interface_code_symbols(condition, consumer_interface_config):
        raise ValueError("Substitution adapters do not share a code alphabet")

    producer_model, producer_tokenizer = _load_evaluation_model(
        run_path, condition
    )
    consumer_model, consumer_tokenizer = _load_evaluation_model(
        consumer_run, condition
    )
    cases = read_programs(run_path / TEST_PATH)
    maximum = min(len(cases), int(spec.get("max_cases", len(cases))))
    selected = _balanced_prefix(cases, maximum)
    complete = {str(row["id"]) for row in _read_rows(run_path, consumer_run)}
    pending = [case for case in selected if str(case["id"]) not in complete]
    output = _output_dir(run_path, consumer_run) / "cases.jsonl"
    block_size = int(
        config["state_handoff_training"].get("evaluation", {}).get(
            "block_size", 2
        )
    )
    for index, case in enumerate(pending, 1):
        append_jsonl(
            output,
            evaluate_interface_program_hf(
                model=producer_model,
                tokenizer=producer_tokenizer,
                consumer_model=consumer_model,
                consumer_tokenizer=consumer_tokenizer,
                case=case,
                prompt_config=config["state_handoff_training"].get("prompt", {}),
                condition=condition,
                interface_config=interface_config,
                block_size=block_size,
            ),
        )
        if on_progress and (index == 1 or index == len(pending) or index % 10 == 0):
            on_progress(f"adapter substitution {index}/{len(pending)} cases")

    rows = _read_rows(run_path, consumer_run)
    rows = [row for row in rows if str(row["id"]) in {str(case["id"]) for case in selected}]
    source_rows = {
        str(row["id"]): row
        for row in read_interface_evaluation_cases(run_path, condition)
    }
    if any(str(row["id"]) not in source_rows for row in rows):
        raise RuntimeError("Source self-evaluation must finish before substitution")
    clusters = [str(row["program_context"]) for row in rows]
    substituted = [
        bool(
            row["predicted_final"]
            and row["predicted_final"]["is_expected_unconstrained"]
        )
        for row in rows
    ]
    self_values = [
        bool(
            source_rows[str(row["id"])]["predicted_final"]
            and source_rows[str(row["id"])]["predicted_final"][
                "is_expected_unconstrained"
            ]
        )
        for row in rows
    ]
    gold_values = [
        bool(row["gold_final"]["is_expected_unconstrained"]) for row in rows
    ]
    differences = [
        int(left) - int(right)
        for left, right in zip(substituted, self_values)
    ]
    maximum_drop = float(spec.get("max_accuracy_drop", 0.05))
    difference = cluster_bootstrap_mean_ci(differences, clusters, seed=82_101)
    result = {
        "schema_version": 1,
        "producer_run": str(run_path),
        "consumer_run": str(consumer_run),
        "condition": condition,
        "case_count": len(rows),
        "expected_case_count": maximum,
        "complete": len(rows) == maximum,
        "substituted_accuracy": cluster_bootstrap_mean_ci(
            substituted, clusters, seed=82_102
        ),
        "source_self_accuracy": cluster_bootstrap_mean_ci(
            self_values, clusters, seed=82_103
        ),
        "consumer_gold_code_accuracy": cluster_bootstrap_mean_ci(
            gold_values, clusters, seed=82_104
        ),
        "substituted_minus_self": difference,
        "gate": {
            "max_accuracy_drop": maximum_drop,
            "passed": (
                len(rows) == maximum
                and float(difference["mean"]) >= -maximum_drop
                and float(
                    cluster_bootstrap_mean_ci(
                        gold_values, clusters, seed=82_105
                    )["mean"]
                )
                >= 0.90
            ),
        },
    }
    write_json(_output_dir(run_path, consumer_run) / "summary.json", result)
    return result
