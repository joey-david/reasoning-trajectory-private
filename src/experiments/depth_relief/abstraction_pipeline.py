"""Run-folder contracts for matched-history state abstraction experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl

from .abstraction import (
    build_state_abstraction_benchmark,
    validate_abstraction_case,
)
from .abstraction_behavior import summarize_abstraction_behavior
from .abstraction_information import analyze_matched_history_information
from .abstraction_interchange import (
    build_interchange_pairs,
    fit_implicit_state_subspaces,
    select_interchange_pairs,
    summarize_interchange,
)
from .abstraction_transfer import analyze_existing_transfer_matrix
from .benchmark import state_symbols
from .pipeline import factorization_output_path, read_factorization_results
from .transfer_pipeline import (
    activation_dir as transfer_activation_dir,
    read_captures as read_transfer_captures,
    read_split as read_transfer_split,
)


ROOT = Path("depth_relief/state_abstraction")
ACTIVATION_DIR = ROOT / "activations"
MANIFEST_PATH = ROOT / "manifest.json"
INFORMATION_PATH = ROOT / "information_summary.json"
PAIR_PATH = ROOT / "pairs.jsonl"
PROJECTION_PATH = ROOT / "projection.npz"
INTERCHANGE_PATH = ROOT / "interchange.jsonl"
INTERCHANGE_MANIFEST_PATH = ROOT / "interchange_manifest.json"
INTERCHANGE_SUMMARY_PATH = ROOT / "interchange_summary.json"
TRANSFER_MATRIX_PATH = Path("depth_relief/state_transfer/decoder_transfer_summary.json")


def abstraction_activation_dir(run_path: Path) -> Path:
    return run_path / ACTIVATION_DIR


def pair_path(run_path: Path) -> Path:
    return run_path / PAIR_PATH


def interchange_path(run_path: Path) -> Path:
    return run_path / INTERCHANGE_PATH


def _read_unique(path: Path, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [tuple(row[key] for key in keys) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate artifact identities in {path}")
    return rows


def read_pairs(run_path: Path) -> list[dict[str, Any]]:
    return _read_unique(pair_path(run_path), ("id",))


def read_interchange(run_path: Path) -> list[dict[str, Any]]:
    return _read_unique(interchange_path(run_path), ("id", "layer"))


def prepare_state_abstraction(run_path: Path) -> dict[str, Any]:
    """Materialize balanced history equivalence classes and group splits."""
    config = load_config(run_path)
    experiment = config.get("state_abstraction", {})
    cases = build_state_abstraction_benchmark(experiment.get("benchmark", {}))
    dataset_path = run_path / "dataset.jsonl"
    write_jsonl(dataset_path, cases)
    manifest = {
        "schema_version": 1,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "groups": sorted({str(case["abstraction_group"]) for case in cases}),
        "split_counts": {
            split: sum(case["abstraction_split"] == split for case in cases)
            for split in ("train", "validation", "test")
        },
        "group_split_counts": {
            split: len(
                {
                    str(case["abstraction_group"])
                    for case in cases
                    if case["abstraction_split"] == split
                }
            )
            for split in ("train", "validation", "test")
        },
        "state_count": len(state_symbols(cases[0])),
        "path_count": len({int(case["path_code"]) for case in cases}),
        "history_steps": sorted({int(case["history_steps"]) for case in cases}),
        "formats": sorted({str(case["format"]) for case in cases}),
        "pair_contract": {
            "same_state": "different path; identical current state and final rule",
            "different_state": "identical path prefix; only final history operation differs",
        },
        "captured_conditions": ["compose", "synthesize", "update"],
    }
    write_json(run_path / MANIFEST_PATH, manifest)
    write_json(
        run_path / "depth_relief/factorization_manifest.json",
        {
            "schema_version": 1,
            "case_count": len(cases),
            "history_families": ["add"],
            "final_families": ["pointer"],
            "formats": manifest["formats"],
            "bits": sorted({int(case["bits"]) for case in cases}),
            "state_representations": ["decimal"],
            "state_symbols": [list(state_symbols(cases[0]))],
            "history_steps": manifest["history_steps"],
            "assays": ["read", "update", "synthesize", "compose"],
            "constituent_controls": "one actual-input control per history transition",
            "diagnostic_targets": [
                "correct_composition",
                "history_only",
                "final_on_start",
                "identity",
            ],
        },
    )
    return manifest


def validate_state_abstraction(run_path: Path) -> dict[str, Any]:
    """Validate token contracts, balanced classes, and aligned causal sites."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    experiment = config.get("state_abstraction", {})
    prompt = config.get("state_materialization", {})
    cases = load_samples(run_path / "dataset.jsonl")
    records = {
        str(case["id"]): validate_abstraction_case(
            tokenizer=tokenizer, case=case, config=prompt
        )
        for case in cases
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(
            (str(case["abstraction_group"]), str(case["format"])), []
        ).append(case)
    context_signatures = {
        (
            representation,
            int(group_cases[0]["initial_state"]),
            tuple(int(value) for value in group_cases[0]["final_rule"]["mapping"]),
            tuple(
                int(rule["value"])
                for rule in next(
                    case for case in group_cases if int(case["path_code"]) == 0
                )["history"][:-1]
            ),
        )
        for (_group, representation), group_cases in grouped.items()
    }
    if len(context_signatures) != len(grouped):
        raise ValueError("Program contexts are not independently distinct")
    state_count = 2 ** int(cases[0]["bits"])
    path_count = len({int(case["path_code"]) for case in cases})
    groups_by_horizon_split = {
        (history_steps, split): {
            str(case["abstraction_group"])
            for case in cases
            if int(case["history_steps"]) == history_steps
            and case["abstraction_split"] == split
        }
        for history_steps in {int(case["history_steps"]) for case in cases}
        for split in ("train", "validation", "test")
    }
    if min(map(len, groups_by_horizon_split.values())) < 5:
        raise ValueError(
            "Confirmatory abstraction requires five independent groups per horizon and split"
        )
    for (group, _format), group_cases in grouped.items():
        cells = {
            (int(case["current_state"]), int(case["path_code"]))
            for case in group_cases
        }
        if len(cells) != state_count * path_count:
            raise ValueError(f"Group {group} is not a complete state-by-path grid")
        compose_sites = {
            tuple(
                int(position["token_index"])
                for position in records[str(case["id"])]["compose_positions"]
            )
            for case in group_cases
        }
        if len(compose_sites) != 1:
            raise ValueError(f"Group {group} does not share causal token positions")
        endpoint_name = f"history_step_{int(group_cases[0]['history_steps'])}"
        endpoint_token_ids = {
            int(
                records[str(case["id"])]["compose_position_token_ids"][
                    endpoint_name
                ]
            )
            for case in group_cases
        }
        if len(endpoint_token_ids) != 1:
            raise ValueError(f"Group {group} changes the causal anchor token")
        for case in group_cases:
            different = next(
                candidate
                for candidate in group_cases
                if int(candidate["path_code"]) == int(case["path_code"])
                and int(candidate["current_state"])
                == (int(case["current_state"]) + 1) % state_count
            )
            if case["history"][:-1] != different["history"][:-1]:
                raise ValueError("Minimal state pair differs before its final history step")
    causal_anchor_token_ids = {
        int(
            records[str(case["id"])]["compose_position_token_ids"][
                f"history_step_{int(case['history_steps'])}"
            ]
        )
        for case in cases
    }
    if len(causal_anchor_token_ids) != 1:
        raise ValueError("Causal endpoint is not one shared lexical token")
    return {
        "run_path": str(run_path),
        "case_count": len(cases),
        "condition_count": sum(
            int(record["condition_count"]) for record in records.values()
        ),
        "group_count": len(grouped),
        "group_counts_by_horizon_split": {
            f"h{history_steps}_{split}": len(groups)
            for (history_steps, split), groups in sorted(groups_by_horizon_split.items())
        },
        "token_count_range": [
            min(record["token_count_range"][0] for record in records.values()),
            max(record["token_count_range"][1] for record in records.values()),
        ],
        "state_count": state_count,
        "path_count": path_count,
        "capture_arrays": ["compose_trace", "synthesize_trace", "update_trace"],
        "causal_anchor_token_ids": sorted(causal_anchor_token_ids),
        "causal_layers": [int(layer) for layer in experiment["causal_layers"]],
        "validated": True,
    }


def reconcile_abstraction_artifacts(
    run_path: Path, cases: list[dict[str, Any]]
) -> set[str]:
    """Keep only behavior rows with complete semantic activation artifacts."""
    indexed = {str(case["id"]): case for case in cases}
    rows = read_factorization_results(run_path)
    valid = []
    for row in rows:
        case_id = str(row["id"])
        case = indexed.get(case_id)
        if case is None:
            continue
        expected = [
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
            or any(not isinstance(position, dict) for position in positions)
            or [position.get("name") for position in positions] != expected
        ):
            continue
        try:
            with np.load(abstraction_activation_dir(run_path) / f"{case_id}.npz") as arrays:
                if not {
                    "compose_trace",
                    "synthesize_trace",
                    "update_trace",
                }.issubset(arrays.files):
                    continue
                shapes = {
                    name: arrays[name].shape
                    for name in (
                        "compose_trace",
                        "synthesize_trace",
                        "update_trace",
                    )
                }
                if (
                    any(len(shape) != 3 for shape in shapes.values())
                    or shapes["compose_trace"][0] != len(expected)
                    or shapes["synthesize_trace"][0] != 1
                    or shapes["update_trace"][0] != 2
                    or len({shape[1:] for shape in shapes.values()}) != 1
                ):
                    continue
        except (OSError, ValueError):
            continue
        valid.append(row)
    if len(valid) != len(rows):
        write_jsonl(factorization_output_path(run_path), valid)
    return {str(row["id"]) for row in valid}


def _load_activations(
    run_path: Path, case_ids: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    activations = {}
    for case_id in case_ids:
        with np.load(abstraction_activation_dir(run_path) / f"{case_id}.npz") as arrays:
            activations[case_id] = {
                name: arrays[name].astype(np.float16) for name in arrays.files
            }
    return activations


def analyze_state_abstraction_information(run_path: Path) -> dict[str, Any]:
    """Run the confirmatory state-information and path-leakage analysis."""
    config = load_config(run_path).get("state_abstraction", {})
    cases = {
        str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")
    }
    rows = {str(row["id"]): row for row in read_factorization_results(run_path)}
    if set(rows) != set(cases):
        raise ValueError("State abstraction capture is incomplete")
    summary = analyze_matched_history_information(
        cases=cases,
        captures=rows,
        activations=_load_activations(run_path, list(cases)),
        rank=int(config.get("decoder_rank", 7)),
        seed=int(config.get("seed", 73)),
    )
    summary["behavior"] = summarize_abstraction_behavior(cases, rows)
    write_json(run_path / INFORMATION_PATH, summary)
    return summary


def analyze_transfer_decoder_matrix(run_path: Path) -> dict[str, Any]:
    """Reanalyse the completed 94 captures without model inference."""
    config = load_config(run_path).get("state_handoff", {})
    cases = {
        str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")
    }
    captures = {str(row["id"]): row for row in read_transfer_captures(run_path)}
    activations = {}
    for case_id in cases:
        with np.load(transfer_activation_dir(run_path) / f"{case_id}.npz") as arrays:
            activations[case_id] = {
                name: arrays[name].astype(np.float16) for name in arrays.files
            }
    summary = analyze_existing_transfer_matrix(
        cases=cases,
        split=read_transfer_split(run_path),
        captures=captures,
        activations=activations,
        rank=int(config.get("rank", 7)),
        seed=int(config.get("seed", 53)) + 100,
    )
    write_json(run_path / TRANSFER_MATRIX_PATH, summary)
    return summary


def interchange_eligibility(run_path: Path) -> dict[str, Any]:
    """Expose whether behavior supports matched causal interchange."""
    config = load_config(run_path).get("state_abstraction", {})
    cases = load_samples(run_path / "dataset.jsonl")
    rows = read_factorization_results(run_path)
    pairs = select_interchange_pairs(
        build_interchange_pairs(cases, rows),
        max_per_group=int(config.get("max_pairs_per_group", 8)),
        seed=int(config.get("seed", 73)) + 1,
    )
    state_count = 2 ** int(cases[0]["bits"])
    train_state_counts = {
        str(state): sum(
            case["abstraction_split"] == "train"
            and int(case["current_state"]) == state
            for case in cases
        )
        for state in range(state_count)
    }
    pair_counts = {
        split: sum(pair["split"] == split for pair in pairs)
        for split in ("train", "validation", "test")
    }
    minimum_pairs = int(config.get("min_pairs_per_evaluation_split", 24))
    checks = {
        "validation_pairs": pair_counts["validation"] >= minimum_pairs,
        "test_pairs": pair_counts["test"] >= minimum_pairs,
        "balanced_training_states": len(set(train_state_counts.values())) == 1,
    }
    return {
        "run_path": str(run_path),
        "eligible": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_pairs_per_evaluation_split": minimum_pairs,
        },
        "pair_counts": pair_counts,
        "training_cases_by_state": train_state_counts,
    }


def prepare_interchange(run_path: Path) -> dict[str, Any]:
    """Pin matched pairs and fit implicit state directions on train groups."""
    eligibility = interchange_eligibility(run_path)
    if not eligibility["eligible"]:
        raise RuntimeError("Behavioral pair gate failed; interchange is inadmissible")
    config = load_config(run_path).get("state_abstraction", {})
    cases_list = load_samples(run_path / "dataset.jsonl")
    rows_list = read_factorization_results(run_path)
    cases = {str(case["id"]): case for case in cases_list}
    pairs = select_interchange_pairs(
        build_interchange_pairs(cases_list, rows_list),
        max_per_group=int(config.get("max_pairs_per_group", 8)),
        seed=int(config.get("seed", 73)) + 1,
    )
    write_jsonl(pair_path(run_path), pairs)
    arrays = fit_implicit_state_subspaces(
        cases=cases,
        captures={str(row["id"]): row for row in rows_list},
        activations=_load_activations(
            run_path,
            [
                case_id
                for case_id, case in cases.items()
                if case["abstraction_split"] == "train"
            ],
        ),
        rank=int(config.get("decoder_rank", 7)),
        seed=int(config.get("seed", 73)),
    )
    projection_path = run_path / PROJECTION_PATH
    projection_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(projection_path, **arrays)
    manifest = {
        "schema_version": 1,
        **eligibility,
        "layers": [int(layer) for layer in config["causal_layers"]],
        "patch_modes": [
            "state_different",
            "full_different",
            "random_different",
            "state_same",
            "full_same",
        ],
        "random_control": "orthogonal to the state subspace and norm-matched per patch",
        "source_position": "final history-step endpoint",
        "target_position": "same endpoint in recipient",
        "answer_leakage": "impossible: source site precedes the shared final rule",
    }
    write_json(run_path / INTERCHANGE_MANIFEST_PATH, manifest)
    return manifest


def analyze_interchange(run_path: Path) -> dict[str, Any]:
    """Select on validation pairs and report once on held-out groups."""
    config = load_config(run_path).get("state_abstraction", {})
    captures = {str(row["id"]): row for row in read_factorization_results(run_path)}
    pairs = read_pairs(run_path)
    expected = {
        (str(pair["id"]), int(layer))
        for pair in pairs
        if pair["split"] in {"validation", "test"}
        for layer in config["causal_layers"]
    }
    patches = read_interchange(run_path)
    actual = {(str(row["id"]), int(row["layer"])) for row in patches}
    if actual != expected:
        raise ValueError(f"Interchange is incomplete: {len(actual)}/{len(expected)}")
    summary = summarize_interchange(
        captures=captures,
        pairs=pairs,
        patches=patches,
        gate=config.get("causal_gate", {}),
    )
    write_json(run_path / INTERCHANGE_SUMMARY_PATH, summary)
    return summary


def state_abstraction_status(run_path: Path) -> dict[str, Any]:
    cases = load_samples(run_path / "dataset.jsonl") if (run_path / "dataset.jsonl").exists() else []
    rows = read_factorization_results(run_path)
    return {
        "run_path": str(run_path),
        "case_count": len(cases),
        "capture_count": len(rows),
        "information_exists": (run_path / INFORMATION_PATH).exists(),
        "pair_count": len(read_pairs(run_path)),
        "interchange_count": len(read_interchange(run_path)),
        "interchange_summary_exists": (run_path / INTERCHANGE_SUMMARY_PATH).exists(),
    }
