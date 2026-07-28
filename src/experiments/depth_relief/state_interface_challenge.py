"""Small length-extrapolation challenges for trained state interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import write_jsonl

from .factorization import render_factorization_prompts
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .state_handoff_challenge_programs import (
    build_full_support_proof_programs,
    build_proof_depth_programs,
)
from .state_handoff_evaluation import (
    _load_evaluation_model,
    evaluate_program_hf,
)
from .state_handoff_programs import build_test_programs
from .state_interface_evaluation import evaluate_interface_program_hf
from .state_interface_proof_metrics import (
    proof_step_diagnostics,
    proof_transition_class_summary,
)


def challenge_dir(run_path: Path, profile: str) -> Path:
    """Return the artifact owner for one long-horizon profile."""
    return run_path / "evaluation/challenges" / profile


def configured_challenge_profiles(run_path: Path) -> dict[str, dict[str, Any]]:
    """Expand explicit profiles and compact source/condition/template matrices."""
    config = load_config(run_path)
    profiles = {
        str(name): dict(spec)
        for name, spec in config.get("state_interface_challenges", {}).items()
    }
    matrix = config.get("state_interface_challenge_matrix")
    if not matrix:
        return profiles
    templates = matrix["templates"]
    outcome_owners: dict[tuple[str, str], str] = {}
    for source in matrix["sources"]:
        source_name = str(source["name"])
        for condition_spec in source["conditions"]:
            if isinstance(condition_spec, str):
                condition = condition_spec
                template_names = tuple(templates)
            else:
                condition = str(condition_spec["name"])
                template_names = tuple(
                    str(value) for value in condition_spec["templates"]
                )
            for template_name in template_names:
                template = templates[template_name]
                profile = f"{source_name}__{condition}__{template_name}"
                if profile in profiles:
                    raise ValueError(f"Duplicate challenge profile: {profile}")
                owner_key = (source_name, template_name)
                outcome_owner = outcome_owners.setdefault(owner_key, profile)
                spec = {
                    **template,
                    "interface_run": str(source["interface_run"]),
                    "interface_condition": str(condition),
                    "outcome_run": str(source["outcome_run"]),
                    "program_split": (f"proof_weekend_{source_name}_{template_name}"),
                    "seed": int(source["challenge_seed"]),
                    "outcome_owner_profile": outcome_owner,
                }
                profiles[profile] = spec
    return profiles


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
    profiles = configured_challenge_profiles(run_path)
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
                {"test_composition_splits": list(spec["test_composition_splits"])}
                if "test_composition_splits" in spec
                else {}
            ),
        }
        if spec.get("full_state_support"):
            if str(spec["domain"]) != "horn_proof":
                raise ValueError(
                    "Full-state-support challenges require the Horn-proof domain"
                )
            horizons = tuple(int(value) for value in spec["horizons"])
            cases = []
            for horizon in horizons:
                cases.extend(
                    build_full_support_proof_programs(
                        horizon=horizon,
                        context_count=int(spec["program_contexts"]),
                        width=int(spec["bits"]),
                        seed=int(spec["seed"]),
                        split=str(spec.get("program_split", f"challenge_{profile}")),
                    )
                )
        elif "active_depths" in spec:
            horizons = tuple(int(value) for value in spec["horizons"])
            if str(spec["domain"]) != "horn_proof":
                raise ValueError(
                    "Active-depth challenges require the Horn-proof domain"
                )
            cases = []
            for horizon in horizons:
                cases.extend(
                    build_proof_depth_programs(
                        active_depths=tuple(
                            int(value) for value in spec["active_depths"]
                        ),
                        horizon=horizon,
                        context_count=int(spec["program_contexts"]),
                        paths_per_depth=int(spec["paths_per_depth"]),
                        width=int(spec["bits"]),
                        seed=int(spec["seed"]),
                        split=str(spec.get("program_split", f"challenge_{profile}")),
                        proof_final=str(spec.get("proof_final", "action")),
                        proof_topologies=tuple(
                            str(value)
                            for value in spec.get("proof_topologies", ("mixed",))
                        ),
                        balanced_queries=bool(spec.get("balanced_queries", False)),
                        endpoint_cardinality=(
                            int(spec["endpoint_cardinality"])
                            if "endpoint_cardinality" in spec
                            else None
                        ),
                    )
                )
        else:
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
            "active_depths": sorted(
                {
                    int(case.get("active_transition_count", -1))
                    for case in cases
                    if "active_transition_count" in case
                }
            ),
            "proof_topologies": sorted(
                {
                    str(case["proof_topology"])
                    for case in cases
                    if "proof_topology" in case
                }
            ),
            "endpoint_cardinalities": sorted(
                {
                    int(case["endpoint_cardinality"])
                    for case in cases
                    if "endpoint_cardinality" in case
                }
            ),
            "outcome_owner_profile": str(spec.get("outcome_owner_profile", profile)),
        }
    manifest = {"schema_version": 1, "profiles": result}
    write_json(run_path / "evaluation/challenges/manifest.json", manifest)
    return manifest


def _side_path(run_path: Path, profile: str, side: str) -> Path:
    if side == "outcome":
        manifest_path = run_path / "evaluation/challenges/manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            profile = str(
                manifest.get("profiles", {})
                .get(profile, {})
                .get("outcome_owner_profile", profile)
            )
    return challenge_dir(run_path, profile) / f"{side}_cases.jsonl"


def _interface_cost(row: dict[str, Any]) -> tuple[int, int, int]:
    deployed = [int(step["prompt_token_count"]) for step in row["steps"]]
    if row["predicted_final"] is not None:
        deployed.append(int(row["predicted_final"]["prompt_token_count"]))
    return sum(deployed), max(deployed, default=0), len(deployed)


def _summarize_challenge_ids(
    ids: list[str],
    *,
    program_index: dict[str, dict[str, Any]],
    interface_index: dict[str, dict[str, Any]],
    outcome_index: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    interface_values = [
        bool(
            interface_index[case_id]["predicted_final"]
            and interface_index[case_id]["predicted_final"]["is_expected_unconstrained"]
        )
        for case_id in ids
    ]
    constrained_values = [
        bool(
            interface_index[case_id]["predicted_final"]
            and interface_index[case_id]["predicted_final"]["is_expected"]
        )
        for case_id in ids
    ]
    valid_values = [
        bool(
            interface_index[case_id]["predicted_final"]
            and interface_index[case_id]["predicted_final"]["unconstrained_prediction"]
            is not None
        )
        for case_id in ids
    ]
    outcome_values = [
        bool(
            outcome_index[case_id]["conditions"]["one_pass_compose"][
                "is_expected_unconstrained"
            ]
        )
        for case_id in ids
    ]
    semantic_values = [
        int(program_index[case_id]["current_state"])
        in interface_index[case_id]["predicted_semantic_states"]
        for case_id in ids
    ]
    exact_state_values = [
        (
            interface_index[case_id]["predicted_semantic_states"]
            == [int(program_index[case_id]["current_state"])]
        )
        for case_id in ids
    ]
    local_values = [
        bool(step["locally_correct"])
        for case_id in ids
        for step in interface_index[case_id]["steps"]
        if step.get("locally_correct") is not None
    ]
    local_exact_values = [
        (
            step.get("predicted_semantic_states")
            == [int(step["true_output_state"])]
        )
        for case_id in ids
        for step in interface_index[case_id]["steps"]
    ]
    clusters = [str(program_index[case_id]["program_context"]) for case_id in ids]
    result: dict[str, Any] = {
        "case_count": len(ids),
        "interface_accuracy": cluster_bootstrap_mean_ci(
            interface_values, clusters, seed=seed
        ),
        "interface_constrained_accuracy": cluster_bootstrap_mean_ci(
            constrained_values, clusters, seed=seed + 1
        ),
        "interface_valid_output_rate": cluster_bootstrap_mean_ci(
            valid_values, clusters, seed=seed + 2
        ),
        "outcome_accuracy": cluster_bootstrap_mean_ci(
            outcome_values, clusters, seed=seed + 3
        ),
        "semantic_state_accuracy": cluster_bootstrap_mean_ci(
            semantic_values, clusters, seed=seed + 4
        ),
        "exact_state_accuracy": cluster_bootstrap_mean_ci(
            exact_state_values, clusters, seed=seed + 5
        ),
        "local_semantic_closure": bootstrap_mean_ci(local_values, seed=seed + 6),
        "local_exact_state_closure": bootstrap_mean_ci(
            local_exact_values, seed=seed + 7
        ),
        "interface_minus_outcome": cluster_bootstrap_mean_ci(
            [
                int(left) - int(right)
                for left, right in zip(interface_values, outcome_values)
            ],
            clusters,
            seed=seed + 8,
        ),
    }
    result.update(
        proof_step_diagnostics(
            ids,
            program_index=program_index,
            interface_index=interface_index,
            seed=seed + 9,
        )
    )
    if all("next_state" in program_index[case_id] for case_id in ids):
        result["answer_positive_rate"] = sum(
            int(program_index[case_id]["next_state"]) for case_id in ids
        ) / len(ids)
    return result


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
        constrained = [
            bool(row["predicted_final"] and row["predicted_final"]["is_expected"])
            for row in interface
        ]
        valid = [
            bool(
                row["predicted_final"]
                and row["predicted_final"]["unconstrained_prediction"] is not None
            )
            for row in interface
        ]
        costs = [_interface_cost(row) for row in interface]
        result["interface"] = {
            "accuracy": bootstrap_mean_ci(values, seed=83_101),
            "constrained_accuracy": bootstrap_mean_ci(constrained, seed=83_104),
            "valid_output_rate": bootstrap_mean_ci(valid, seed=83_105),
            "mean_candidate_probability_mass": sum(
                float(row["predicted_final"]["candidate_probability_mass"])
                for row in interface
                if row["predicted_final"]
            )
            / len(interface),
            "mean_total_prompt_tokens": sum(row[0] for row in costs) / len(costs),
            "max_prompt_tokens_per_call": max(row[1] for row in costs),
            "mean_model_calls": sum(row[2] for row in costs) / len(costs),
        }
    if len(outcome) == expected:
        values = [
            bool(row["conditions"]["one_pass_compose"]["is_expected_unconstrained"])
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
        result["interface_minus_outcome"] = bootstrap_mean_ci(differences, seed=83_103)
        program_index = {str(row["id"]): row for row in programs}
        interface_index = {str(row["id"]): row for row in interface}
        strata = {
            "by_active_transition_count": (
                "active_transition_count",
                sorted(
                    {
                        int(row["active_transition_count"])
                        for row in programs
                        if "active_transition_count" in row
                    }
                ),
            ),
            "by_surface_horizon": (
                "history_steps",
                sorted(
                    {
                        int(row["history_steps"])
                        for row in programs
                        if "history_steps" in row
                    }
                ),
            ),
            "by_proof_topology": (
                "proof_topology",
                sorted(
                    {
                        str(row["proof_topology"])
                        for row in programs
                        if "proof_topology" in row
                    }
                ),
            ),
            "by_proof_query_mode": (
                "proof_query_mode",
                sorted(
                    {
                        str(row["final_rule"]["mode"])
                        for row in programs
                        if row.get("final_rule", {}).get("kind") == "proof_query"
                    }
                ),
            ),
            "by_proof_consumer": (
                "proof_consumer",
                sorted(
                    {
                        str(row["proof_consumer"])
                        for row in programs
                        if "proof_consumer" in row
                    }
                ),
            ),
        }
        for row in programs:
            if row.get("final_rule", {}).get("kind") == "proof_query":
                row["proof_query_mode"] = str(row["final_rule"]["mode"])
        result["overall"] = _summarize_challenge_ids(
            [str(row["id"]) for row in programs],
            program_index=program_index,
            interface_index=interface_index,
            outcome_index=indexed,
            seed=83_150,
        )
        for offset, (output_key, (field, values)) in enumerate(strata.items()):
            if values:
                result[output_key] = {
                    str(value): _summarize_challenge_ids(
                        [str(row["id"]) for row in programs if row.get(field) == value],
                        program_index=program_index,
                        interface_index=interface_index,
                        outcome_index=indexed,
                        seed=83_200 + 100 * offset + index,
                    )
                    for index, value in enumerate(values)
                }
        transition_summary = proof_transition_class_summary(interface)
        if transition_summary:
            result["by_proof_transition_class"] = transition_summary
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
    spec = configured_challenge_profiles(run_path)[profile]
    source_run = Path(str(spec[f"{side}_run"]))
    condition = (
        str(spec["interface_condition"]) if side == "interface" else "outcome_only"
    )
    source_config = load_config(source_run)
    model, tokenizer = _load_evaluation_model(source_run, condition)
    experiment = source_config["state_handoff_training"]
    cases = _read(challenge_dir(run_path, profile) / "programs.jsonl")
    completed = {str(row["id"]) for row in _read(_side_path(run_path, profile, side))}
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
