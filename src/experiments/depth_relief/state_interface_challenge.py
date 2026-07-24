"""Small length-extrapolation challenges for trained state interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import write_jsonl

from .factorization import render_factorization_prompts
from .metrics import bootstrap_mean_ci
from .state_handoff_evaluation import (
    _load_evaluation_model,
    evaluate_program_hf,
)
from .state_handoff_programs import build_test_programs
from .state_interface_evaluation import evaluate_interface_program_hf


def challenge_dir(run_path: Path, profile: str) -> Path:
    """Return the artifact owner for one long-horizon profile."""
    return run_path / "evaluation/challenges" / profile


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate length-extrapolation IDs in {path}")
    return rows


def prepare_interface_challenges(run_path: Path) -> dict[str, Any]:
    """Write compact, balanced horizon-64/128 challenge banks."""
    profiles = load_config(run_path).get("state_interface_challenges", {})
    result = {}
    for profile, spec in profiles.items():
        dataset = {
            "domain": str(spec["domain"]),
            **(
                {"proof_final": str(spec["proof_final"])}
                if "proof_final" in spec
                else {}
            ),
            **(
                {
                    "test_composition_splits": list(
                        spec["test_composition_splits"]
                    )
                }
                if "test_composition_splits" in spec
                else {}
            ),
        }
        cases = build_test_programs(
            horizons=tuple(int(value) for value in spec["horizons"]),
            context_count=int(spec["program_contexts"]),
            paths_per_state=int(spec["paths_per_state"]),
            width=int(spec["bits"]),
            seed=int(spec["seed"]),
            split=f"challenge_{profile}",
            dataset=dataset,
        )
        output = challenge_dir(run_path, profile) / "programs.jsonl"
        write_jsonl(output, cases)
        result[profile] = {
            "case_count": len(cases),
            "horizons": sorted({int(case["history_steps"]) for case in cases}),
            "states": sorted({int(case["current_state"]) for case in cases}),
            "domain": str(spec["domain"]),
        }
    manifest = {"schema_version": 1, "profiles": result}
    write_json(run_path / "evaluation/challenges/manifest.json", manifest)
    return manifest


def _side_path(run_path: Path, profile: str, side: str) -> Path:
    return challenge_dir(run_path, profile) / f"{side}_cases.jsonl"


def _interface_cost(row: dict[str, Any]) -> tuple[int, int, int]:
    deployed = [
        int(step["prompt_token_count"]) for step in row["steps"]
    ]
    if row["predicted_final"] is not None:
        deployed.append(int(row["predicted_final"]["prompt_token_count"]))
    return sum(deployed), max(deployed, default=0), len(deployed)


def _write_summary(run_path: Path, profile: str) -> dict[str, Any]:
    interface = _read(_side_path(run_path, profile, "interface"))
    outcome = _read(_side_path(run_path, profile, "outcome"))
    programs = _read(challenge_dir(run_path, profile) / "programs.jsonl")
    expected = len(programs)
    result: dict[str, Any] = {
        "schema_version": 1,
        "profile": profile,
        "expected_case_count": expected,
        "interface_case_count": len(interface),
        "outcome_case_count": len(outcome),
        "complete": len(interface) == expected and len(outcome) == expected,
    }
    if len(interface) == expected:
        values = [
            bool(
                row["predicted_final"]
                and row["predicted_final"]["is_expected_unconstrained"]
            )
            for row in interface
        ]
        costs = [_interface_cost(row) for row in interface]
        result["interface"] = {
            "accuracy": bootstrap_mean_ci(values, seed=83_101),
            "mean_total_prompt_tokens": sum(row[0] for row in costs) / len(costs),
            "max_prompt_tokens_per_call": max(row[1] for row in costs),
            "mean_model_calls": sum(row[2] for row in costs) / len(costs),
        }
    if len(outcome) == expected:
        values = [
            bool(
                row["conditions"]["one_pass_compose"][
                    "is_expected_unconstrained"
                ]
            )
            for row in outcome
        ]
        result["outcome"] = {
            "accuracy": bootstrap_mean_ci(values, seed=83_102),
            "mean_total_prompt_tokens": sum(
                int(row["prompt_token_count"]) for row in outcome
            )
            / len(outcome),
            "max_prompt_tokens_per_call": max(
                int(row["prompt_token_count"]) for row in outcome
            ),
            "mean_model_calls": 1.0,
        }
    if result["complete"]:
        indexed = {str(row["id"]): row for row in outcome}
        differences = []
        for row in interface:
            interface_correct = bool(
                row["predicted_final"]
                and row["predicted_final"]["is_expected_unconstrained"]
            )
            outcome_correct = bool(
                indexed[str(row["id"])]["conditions"]["one_pass_compose"][
                    "is_expected_unconstrained"
                ]
            )
            differences.append(int(interface_correct) - int(outcome_correct))
        result["interface_minus_outcome"] = bootstrap_mean_ci(
            differences, seed=83_103
        )
    write_json(challenge_dir(run_path, profile) / "summary.json", result)
    return result


def evaluate_interface_challenge(
    run_path: Path,
    profile: str,
    side: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one side of a saved long-horizon challenge."""
    if side not in {"interface", "outcome"}:
        raise ValueError("Challenge side must be interface or outcome")
    root_config = load_config(run_path)
    spec = root_config["state_interface_challenges"][profile]
    source_run = Path(str(spec[f"{side}_run"]))
    condition = (
        str(spec["interface_condition"]) if side == "interface" else "outcome_only"
    )
    source_config = load_config(source_run)
    model, tokenizer = _load_evaluation_model(source_run, condition)
    experiment = source_config["state_handoff_training"]
    cases = _read(challenge_dir(run_path, profile) / "programs.jsonl")
    completed = {
        str(row["id"]) for row in _read(_side_path(run_path, profile, side))
    }
    pending = [case for case in cases if str(case["id"]) not in completed]
    for index, case in enumerate(pending, 1):
        if side == "interface":
            row = evaluate_interface_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=experiment.get("prompt", {}),
                condition=condition,
                interface_config=experiment.get("interfaces", {}),
                block_size=int(spec.get("block_size", 2)),
            )
        else:
            row = evaluate_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=experiment.get("prompt", {}),
                condition=condition,
            )
            compose = next(
                prompt
                for prompt in render_factorization_prompts(
                    tokenizer=tokenizer,
                    case=case,
                    config=experiment.get("prompt", {}),
                )
                if prompt["name"] == "compose"
            )
            row["prompt_token_count"] = len(
                tokenizer.encode(compose["text"], add_special_tokens=False)
            )
        append_jsonl(_side_path(run_path, profile, side), row)
        if on_progress and (index == 1 or index == len(pending)):
            on_progress(f"{profile} {side} {index}/{len(pending)} cases")
    return _write_summary(run_path, profile)
