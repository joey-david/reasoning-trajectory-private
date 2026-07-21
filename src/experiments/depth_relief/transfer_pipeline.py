"""Run-folder contracts for causal state-transfer capture and patching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl

from .metrics import bootstrap_mean_ci
from .handoff import analyze_state_localization, summarize_handoff
from .transfer import (
    build_transfer_split,
    fit_state_subspaces,
    summarize_transfer,
    validate_transfer_case,
)


ROOT = Path("depth_relief/state_transfer")
CAPTURE_PATH = ROOT / "captures.jsonl"
PATCH_PATH = ROOT / "patches.jsonl"
ACTIVATION_DIR = ROOT / "activations"
SPLIT_PATH = ROOT / "split.json"
PROJECTION_PATH = ROOT / "projection.npz"
LOCALIZATION_PATH = ROOT / "localization_summary.json"
HANDOFF_PATH = ROOT / "handoff_patches.jsonl"
HANDOFF_MANIFEST_PATH = ROOT / "handoff_manifest.json"
HANDOFF_SUMMARY_PATH = ROOT / "handoff_summary.json"


def capture_path(run_path: Path) -> Path:
    return run_path / CAPTURE_PATH


def patch_path(run_path: Path) -> Path:
    return run_path / PATCH_PATH


def handoff_path(run_path: Path) -> Path:
    return run_path / HANDOFF_PATH


def activation_dir(run_path: Path) -> Path:
    return run_path / ACTIVATION_DIR


def _read_unique(path: Path, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [tuple(row[key] for key in keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate result identities in {path}")
    return rows


def read_captures(run_path: Path) -> list[dict[str, Any]]:
    return _read_unique(capture_path(run_path), ("id",))


def reconcile_capture_artifacts(
    run_path: Path, cases: list[dict[str, Any]]
) -> set[str]:
    """Discard capture index rows whose activation artifact is incomplete."""
    indexed = {str(case["id"]): case for case in cases}
    rows = read_captures(run_path)
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["id"])
        case = indexed.get(case_id)
        if case is None:
            continue
        expected_positions = [
            "start",
            *[
                f"history_step_{index}"
                for index in range(1, int(case["history_steps"]) + 1)
            ],
            "final_rule",
            "answer",
        ]
        positions = row.get("compose_positions")
        if (
            not isinstance(positions, list)
            or not all(isinstance(position, dict) for position in positions)
            or [position.get("name") for position in positions]
            != expected_positions
        ):
            continue
        path = activation_dir(run_path) / f"{case_id}.npz"
        try:
            with np.load(path) as arrays:
                if not {
                    "compose",
                    "materialized",
                    "counterfactual",
                    "compose_trace",
                }.issubset(arrays.files):
                    continue
        except (OSError, ValueError):
            continue
        valid_rows.append(row)
    if len(valid_rows) != len(rows):
        write_jsonl(capture_path(run_path), valid_rows)
    return {str(row["id"]) for row in valid_rows}


def read_patches(run_path: Path) -> list[dict[str, Any]]:
    return _read_unique(patch_path(run_path), ("id", "layer"))


def read_handoff_patches(run_path: Path) -> list[dict[str, Any]]:
    return _read_unique(handoff_path(run_path), ("id", "layer"))


def read_split(run_path: Path) -> dict[str, Any]:
    return json.loads((run_path / SPLIT_PATH).read_text())


def prepare_transfer(run_path: Path) -> dict[str, Any]:
    """Pin confirmed routing cases and an answer-disjoint evaluation split."""
    config = load_config(run_path)
    experiment = config.get("state_transfer", {})
    source_run = Path(str(experiment["source_run"]))
    routing_config = load_config(source_run).get("state_routing", {})
    factorization_run = Path(str(routing_config["source_run"]))
    factorization_config = load_config(factorization_run).get(
        "state_materialization", {}
    )
    prompt = experiment.get("prompt", {})
    if prompt != routing_config.get("prompt", {}) or prompt != factorization_config.get(
        "prompt", {}
    ):
        raise ValueError(
            "Transfer prompt config must exactly match discovery and confirmation"
        )
    source_summary = json.loads(
        (source_run / "depth_relief/routing_summary.json").read_text()
    )
    if not source_summary["gate"]["passed"]:
        raise ValueError("Source routing confirmation did not pass its gate")
    cases = load_samples(source_run / "dataset.jsonl")
    split = build_transfer_split(
        cases,
        seed=int(experiment.get("seed", 47)),
        train_fraction=float(experiment.get("train_fraction", 0.55)),
    )
    write_jsonl(run_path / "dataset.jsonl", cases)
    write_json(run_path / SPLIT_PATH, split)
    manifest = {
        "schema_version": 1,
        "source_run": str(source_run),
        "source_factorization_run": str(factorization_run),
        "source_routing_gate_passed": True,
        "case_count": len(cases),
        "split_counts": {name: len(split[name]) for name in ("train", "validation", "test")},
        "donor_contract": "same format and state; different case and expected answer",
        "conditions": ["compose", "materialized", "counterfactual"],
        "capture_positions": [
            "start",
            "each history-step endpoint",
            "final-rule endpoint",
            "final answer anchor",
        ],
        "patch_modes": [
            "state_gold",
            "state_counterfactual",
            "full_gold",
            "random_gold",
            "random_counterfactual",
        ],
        "layers": [int(layer) for layer in experiment["layers"]],
    }
    write_json(run_path / ROOT / "manifest.json", manifest)
    return manifest


def validate_transfer(run_path: Path) -> dict[str, Any]:
    """Validate the tokenizer and every anti-answer-leakage donor contract."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("state_transfer", {})
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    cases = load_samples(run_path / "dataset.jsonl")
    indexed = {str(case["id"]): case for case in cases}
    records = [
        validate_transfer_case(tokenizer=tokenizer, case=case, config=experiment)
        for case in cases
    ]
    split = read_split(run_path)
    for case_id, donor_set in split["donors"].items():
        recipient = indexed[case_id]
        for branch, state_key, target_key in (
            ("gold", "current_state", "next_state"),
            ("counterfactual", "counterfactual_state", "counterfactual_next_state"),
        ):
            donor_spec = donor_set[branch]
            donor = indexed[donor_spec["case_id"]]
            condition = donor_spec["condition"]
            donor_state_key = "current_state" if condition == "materialized" else "counterfactual_state"
            donor_target_key = "next_state" if condition == "materialized" else "counterfactual_next_state"
            if donor["format"] != recipient["format"]:
                raise ValueError("Donor and recipient formats differ")
            if int(donor[donor_state_key]) != int(recipient[state_key]):
                raise ValueError("Donor and recipient states differ")
            if int(donor[donor_target_key]) == int(recipient[target_key]):
                raise ValueError("Donor answer leaks the recipient target")
    return {
        "run_path": str(run_path),
        "case_count": len(records),
        "condition_count": sum(record["condition_count"] for record in records),
        "token_count_range": [
            min(record["token_count_range"][0] for record in records),
            max(record["token_count_range"][1] for record in records),
        ],
        "answer_anchor_token_ids": sorted(
            {token_id for record in records for token_id in record["answer_anchor_token_ids"]}
        ),
        "donor_count": 2 * (len(split["validation"]) + len(split["test"])),
        "validated": True,
    }


def summarize_capture(run_path: Path) -> dict[str, Any]:
    """Gate causal fitting on retained failure and explicit-state competence."""
    config = load_config(run_path).get("state_transfer", {})
    rows = read_captures(run_path)
    cases = load_samples(run_path / "dataset.jsonl")
    if len(rows) != len(cases):
        raise ValueError(f"Capture is incomplete: {len(rows)}/{len(cases)} cases")
    gate = {
        "min_explicit_accuracy_lower": 0.85,
        "max_compose_accuracy_upper": 0.25,
        "min_candidate_mass_lower": 0.80,
        **config.get("capture_gate", {}),
    }
    accuracy = {
        condition: bootstrap_mean_ci(
            [row["conditions"][condition]["is_expected_unconstrained"] for row in rows],
            seed=1100 + index,
        )
        for index, condition in enumerate(("compose", "materialized", "counterfactual"))
    }
    mass = bootstrap_mean_ci(
        [row["conditions"][condition]["candidate_probability_mass"] for row in rows for condition in ("compose", "materialized", "counterfactual")],
        seed=1110,
    )
    checks = {
        "compose_failure_retained": float(accuracy["compose"]["ci95"][1]) <= float(gate["max_compose_accuracy_upper"]),
        "materialized_competence": float(accuracy["materialized"]["ci95"][0]) >= float(gate["min_explicit_accuracy_lower"]),
        "counterfactual_competence": float(accuracy["counterfactual"]["ci95"][0]) >= float(gate["min_explicit_accuracy_lower"]),
        "candidate_mass": float(mass["ci95"][0]) >= float(gate["min_candidate_mass_lower"]),
    }
    report = {
        "schema_version": 1,
        "case_count": len(rows),
        "accuracy": accuracy,
        "candidate_probability_mass": mass,
        "gate": {"thresholds": gate, "checks": checks, "passed": all(checks.values())},
    }
    write_json(run_path / ROOT / "capture_summary.json", report)
    return report


def fit_transfer(run_path: Path) -> dict[str, Any]:
    """Fit only after the behavioral capture gate passes."""
    capture_summary = summarize_capture(run_path)
    if not capture_summary["gate"]["passed"]:
        raise RuntimeError("State-transfer capture gate failed; patching is inadmissible")
    config = load_config(run_path).get("state_transfer", {})
    cases = {str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")}
    arrays = fit_state_subspaces(
        cases=cases,
        split=read_split(run_path),
        activation_dir=activation_dir(run_path),
        rank=int(config.get("rank", 7)),
        seed=int(config.get("seed", 47)),
    )
    path = run_path / PROJECTION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    report = {
        "schema_version": 1,
        "rank": int(arrays["rank"]),
        "layer_count": int(arrays["state_basis"].shape[0]),
        "hidden_size": int(arrays["state_basis"].shape[1]),
        "orthonormal_max_error": float(np.max(np.abs(np.einsum("lhr,lhs->lrs", arrays["state_basis"], arrays["state_basis"]) - np.eye(int(arrays["rank"]))))),
    }
    write_json(run_path / ROOT / "projection_summary.json", report)
    return report


def analyze_transfer(run_path: Path) -> dict[str, Any]:
    config = load_config(run_path).get("state_transfer", {})
    cases = {str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")}
    captures = {str(row["id"]): row for row in read_captures(run_path)}
    summary = summarize_transfer(
        cases=cases,
        split=read_split(run_path),
        captures=captures,
        patches=read_patches(run_path),
        gate=config.get("causal_gate", {}),
    )
    write_json(run_path / ROOT / "summary.json", summary)
    return summary


def analyze_localization(run_path: Path) -> dict[str, Any]:
    """Decode explicit-state coordinates at held-out Compose positions."""
    config = load_config(run_path)
    experiment = config.get("state_handoff", {})
    cases = {
        str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")
    }
    captures = {str(row["id"]): row for row in read_captures(run_path)}
    if set(captures) != set(cases):
        raise ValueError("State localization requires a complete capture set")
    activations: dict[str, dict[str, np.ndarray]] = {}
    for case_id in cases:
        with np.load(activation_dir(run_path) / f"{case_id}.npz") as arrays:
            if "compose_trace" not in arrays.files:
                raise ValueError("Capture is missing semantic Compose positions")
            activations[case_id] = {
                name: arrays[name].astype(np.float32) for name in arrays.files
            }
    summary = analyze_state_localization(
        cases=cases,
        split=read_split(run_path),
        captures=captures,
        activations=activations,
        rank=int(experiment.get("rank", 7)),
        seed=int(experiment.get("seed", 53)),
        gate=experiment.get("localization_gate", {}),
    )
    write_json(run_path / LOCALIZATION_PATH, summary)
    return summary


def prepare_handoff(run_path: Path) -> dict[str, Any]:
    """Authorize self-handoff only after both prerequisite causal gates pass."""
    eligibility = handoff_eligibility(run_path)
    if not eligibility["localization_gate_passed"]:
        raise RuntimeError("State-localization gate failed; self-handoff is inadmissible")
    if not eligibility["transfer_gate_passed"]:
        raise RuntimeError("State-transfer causal gate failed; self-handoff is inadmissible")
    localization = json.loads((run_path / LOCALIZATION_PATH).read_text())
    split = read_split(run_path)
    manifest = {
        "schema_version": 1,
        "case_count": len(split["test"]),
        "split": "held-out test only",
        "layer": int(localization["layer_selection"]["selected"]),
        "source_position": "final history-step endpoint",
        "target_position": "final answer anchor",
        "patch_modes": ["self_state", "random_self", "full_self"],
        "label_access": "none at intervention time",
    }
    write_json(run_path / HANDOFF_MANIFEST_PATH, manifest)
    return manifest


def handoff_eligibility(run_path: Path) -> dict[str, Any]:
    """Expose whether both prespecified gates admit self-handoff."""
    localization = json.loads((run_path / LOCALIZATION_PATH).read_text())
    transfer = json.loads((run_path / ROOT / "summary.json").read_text())
    localization_passed = bool(localization["gate"]["passed"])
    transfer_passed = bool(transfer["gate"]["passed"])
    return {
        "run_path": str(run_path),
        "eligible": localization_passed and transfer_passed,
        "localization_gate_passed": localization_passed,
        "transfer_gate_passed": transfer_passed,
        "selected_localization_layer": int(
            localization["layer_selection"]["selected"]
        ),
        "selected_transfer_layer": int(transfer["layer_selection"]["selected"]),
    }


def analyze_handoff(run_path: Path) -> dict[str, Any]:
    config = load_config(run_path).get("state_handoff", {})
    cases = {
        str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")
    }
    captures = {str(row["id"]): row for row in read_captures(run_path)}
    rows = read_handoff_patches(run_path)
    expected = set(read_split(run_path)["test"])
    if {str(row["id"]) for row in rows} != expected:
        raise ValueError(f"Self-handoff is incomplete: {len(rows)}/{len(expected)} cases")
    summary = summarize_handoff(
        cases=cases,
        captures=captures,
        rows=rows,
        gate=config.get("causal_gate", {}),
    )
    write_json(run_path / HANDOFF_SUMMARY_PATH, summary)
    return summary


def transfer_status(run_path: Path) -> dict[str, Any]:
    cases = load_samples(run_path / "dataset.jsonl") if (run_path / "dataset.jsonl").exists() else []
    split = read_split(run_path) if (run_path / SPLIT_PATH).exists() else {"validation": [], "test": []}
    layers = load_config(run_path).get("state_transfer", {}).get("layers", [])
    expected_patches = (len(split["validation"]) + len(split["test"])) * len(layers)
    return {
        "run_path": str(run_path),
        "case_count": len(cases),
        "capture_count": len(read_captures(run_path)),
        "patch_count": len(read_patches(run_path)),
        "expected_patch_count": expected_patches,
        "projection_exists": (run_path / PROJECTION_PATH).exists(),
        "summary_exists": (run_path / ROOT / "summary.json").exists(),
        "localization_exists": (run_path / LOCALIZATION_PATH).exists(),
        "handoff_count": len(read_handoff_patches(run_path)),
        "handoff_summary_exists": (run_path / HANDOFF_SUMMARY_PATH).exists(),
    }
