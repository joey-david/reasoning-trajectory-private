"""Recursive reuse of a trained short-horizon state interface."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import write_jsonl

from .benchmark import apply_rule, state_symbols
from .factorization import (
    render_factorization_prompts,
    render_factorization_update_prompt,
)
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .qualification import evaluate_prompt_conditions_hf
from .state_handoff_data import build_test_programs
from .state_handoff_evaluation import _load_evaluation_model


CONTINUATION_ROOT = Path("evaluation/continuation")
DEFAULT_PROFILES = {
    "probe": {
        "horizons": [2, 4, 8, 16],
        "context_count": 10,
        "paths_per_state": 2,
        "block_sizes": [2],
        "seed": 721_501,
    },
    "confirmation": {
        "horizons": [2, 4, 8, 16, 32],
        "context_count": 30,
        "paths_per_state": 4,
        "block_sizes": [1, 2],
        "seed": 721_601,
    },
}


def continuation_dir(run_path: Path, profile: str) -> Path:
    """Return the artifact owner for one continuation profile."""
    if not profile or any(part in {"", ".", ".."} for part in Path(profile).parts):
        raise ValueError(f"Invalid continuation profile: {profile!r}")
    return run_path / CONTINUATION_ROOT / profile


def continuation_profile(run_path: Path, profile: str) -> dict[str, Any]:
    """Resolve a named profile from defaults plus run configuration."""
    configured = (
        load_config(run_path)
        .get("state_handoff_continuation", {})
        .get("profiles", {})
        .get(profile, {})
    )
    if profile not in DEFAULT_PROFILES and not configured:
        raise ValueError(f"Unknown continuation profile: {profile!r}")
    values = {**DEFAULT_PROFILES.get(profile, {}), **configured}
    values["horizons"] = [int(value) for value in values["horizons"]]
    values["block_sizes"] = [int(value) for value in values["block_sizes"]]
    if any(value < 1 for value in values["horizons"] + values["block_sizes"]):
        raise ValueError("Continuation horizons and block sizes must be positive")
    if any(
        horizon % block_size
        for horizon in values["horizons"]
        for block_size in values["block_sizes"]
    ):
        raise ValueError("Every continuation horizon must divide into complete blocks")
    return values


def prepare_continuation_programs(run_path: Path, profile: str) -> dict[str, Any]:
    """Write a deterministic, balanced bank without changing pilot test data."""
    values = continuation_profile(run_path, profile)
    width = int(
        load_config(run_path)
        .get("state_handoff_training", {})
        .get("dataset", {})
        .get("bits", 3)
    )
    cases = build_test_programs(
        horizons=tuple(values["horizons"]),
        context_count=int(values["context_count"]),
        paths_per_state=int(values["paths_per_state"]),
        width=width,
        seed=int(values["seed"]),
        split=f"continuation_{profile}",
    )
    path = continuation_dir(run_path, profile) / "programs.jsonl"
    write_jsonl(path, cases)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "profile": profile,
        "case_count": len(cases),
        "program_context_count": len({row["program_context"] for row in cases}),
        "horizons": values["horizons"],
        "block_sizes": values["block_sizes"],
        "paths_per_state": int(values["paths_per_state"]),
        "state_count": 2**width,
        "capacity_bits": width,
        "sha256": digest,
    }
    write_json(continuation_dir(run_path, profile) / "manifest.json", manifest)
    return manifest


def read_continuation_programs(run_path: Path, profile: str) -> list[dict[str, Any]]:
    """Read a prepared profile and reject duplicate semantic case IDs."""
    path = continuation_dir(run_path, profile) / "programs.jsonl"
    if not path.exists():
        prepare_continuation_programs(run_path, profile)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate continuation program IDs in {path}")
    return rows


def continuation_case_id(case: dict[str, Any], block_size: int) -> str:
    """Return the stable execution ID for one case and block size."""
    return f"{case['id']}__block{block_size}"


def read_continuation_cases(run_path: Path, profile: str) -> list[dict[str, Any]]:
    """Read append-only continuation rows and reject duplicate executions."""
    path = continuation_dir(run_path, profile) / "cases.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate continuation execution IDs in {path}")
    return rows


def block_case(
    case: dict[str, Any], *, input_state: int, start: int, block_size: int
) -> dict[str, Any]:
    """Materialize one in-distribution block from a longer saved program."""
    history = list(case["history"][start : start + block_size])
    if len(history) != block_size:
        raise ValueError("Continuation block extends past the saved history")
    modulus = 2 ** int(case["bits"])
    states = [int(input_state)]
    for rule in history:
        states.append(apply_rule(rule, states[-1], modulus))
    result = {
        **case,
        "id": f"{case['id']}__steps{start + 1}-{start + block_size}",
        "initial_state": int(input_state),
        "history": history,
        "history_steps": block_size,
        "state_path": states,
        "current_state": states[-1],
        "next_state": apply_rule(case["final_rule"], states[-1], modulus),
    }
    return result


def _score(
    *, model: Any, tokenizer: Any, prompt: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_prompt_conditions_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        candidate_symbols=state_symbols(case),
    )[str(prompt["name"])]


def _prediction(result: dict[str, Any]) -> int | None:
    value = result.get("unconstrained_prediction")
    return int(value) if value is not None else None


def evaluate_recursive_program_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    block_size: int,
) -> dict[str, Any]:
    """Reuse a short state mapping until the full history is consumed."""
    horizon = int(case["history_steps"])
    if horizon % block_size:
        raise ValueError("Recursive evaluation requires complete blocks")
    predicted_state = int(case["initial_state"])
    predicted_steps = []
    globally_valid = True
    for start in range(0, horizon, block_size):
        supplied = predicted_state
        local = block_case(
            case, input_state=supplied, start=start, block_size=block_size
        )
        prompt = next(
            row
            for row in render_factorization_prompts(
                tokenizer=tokenizer, case=local, config=prompt_config
            )
            if row["name"] == "synthesize"
        )
        result = _score(model=model, tokenizer=tokenizer, prompt=prompt, case=local)
        predicted = _prediction(result)
        true_endpoint = int(case["state_path"][start + block_size])
        result.update(
            block_index=start // block_size,
            history_start=start,
            supplied_state=supplied,
            local_expected_state=int(local["current_state"]),
            true_endpoint_state=true_endpoint,
            locally_correct=predicted == int(local["current_state"]),
            globally_correct=predicted == true_endpoint,
        )
        predicted_steps.append(result)
        if predicted is None:
            globally_valid = False
            break
        predicted_state = predicted

    final_prompt = render_factorization_update_prompt(
        tokenizer=tokenizer,
        case=case,
        config=prompt_config,
        state=predicted_state,
        rule=case["final_rule"],
        name="recursive_final",
        label="FINAL",
    )
    final = _score(model=model, tokenizer=tokenizer, prompt=final_prompt, case=case)
    provided_target = apply_rule(
        case["final_rule"], predicted_state, 2 ** int(case["bits"])
    )
    final.update(
        supplied_state=predicted_state,
        provided_state_expected_next_state=provided_target,
        follows_supplied_state=_prediction(final) == provided_target,
        expected_next_state=int(case["next_state"]),
        is_expected=_prediction(final) == int(case["next_state"]),
        is_expected_unconstrained=_prediction(final) == int(case["next_state"]),
    )
    return {
        "schema_version": 1,
        "id": continuation_case_id(case, block_size),
        "case_id": str(case["id"]),
        "profile_split": str(case["program_context_split"]),
        "program_context": str(case["program_context"]),
        "history_steps": horizon,
        "block_size": block_size,
        "block_count": math.ceil(horizon / block_size),
        "path_code": int(case["path_code"]),
        "current_state": int(case["current_state"]),
        "predicted_state": predicted_state if globally_valid else None,
        "state_correct": globally_valid
        and predicted_state == int(case["current_state"]),
        "predicted_steps": predicted_steps,
        "final": final,
    }


def summarize_continuation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report recursive accuracy, closure, and path invariance by horizon."""
    if not rows:
        raise ValueError("Cannot summarize empty continuation results")
    by_cell = {}
    for block_size, horizon in sorted(
        {(int(row["block_size"]), int(row["history_steps"])) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if int(row["block_size"]) == block_size
            and int(row["history_steps"]) == horizon
        ]
        clusters = [str(row["program_context"]) for row in selected]
        step_results = [step for row in selected for step in row["predicted_steps"]]
        same_groups: defaultdict[tuple[str, int], list[int | None]] = defaultdict(list)
        for row in selected:
            same_groups[(str(row["program_context"]), int(row["current_state"]))].append(
                row["predicted_state"]
            )
        agreement = []
        for codes in same_groups.values():
            for left_index, left in enumerate(codes):
                agreement.extend(
                    left is not None and left == right
                    for right in codes[left_index + 1 :]
                )
        by_cell[f"block{block_size}_h{horizon}"] = {
            "case_count": len(selected),
            "block_count": horizon // block_size,
            "state_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["state_correct"]) for row in selected],
                clusters,
                seed=8100 + 100 * block_size + horizon,
            ),
            "answer_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["final"]["is_expected_unconstrained"]) for row in selected],
                clusters,
                seed=8200 + 100 * block_size + horizon,
            ),
            "local_closure_accuracy": bootstrap_mean_ci(
                [bool(step["locally_correct"]) for step in step_results],
                seed=8300 + 100 * block_size + horizon,
            ),
            "final_follows_supplied_state": bootstrap_mean_ci(
                [bool(row["final"]["follows_supplied_state"]) for row in selected],
                seed=8400 + 100 * block_size + horizon,
            ),
            "same_state_code_agreement": bootstrap_mean_ci(
                agreement, seed=8500 + 100 * block_size + horizon
            ),
        }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "program_context_count": len({row["program_context"] for row in rows}),
        "by_cell": by_cell,
    }


def evaluate_continuation_profile(
    run_path: Path,
    profile: str,
    *,
    max_cases: int | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resume a recursive profile using the trained explicit adapter."""
    manifest_path = continuation_dir(run_path, profile) / "manifest.json"
    if not manifest_path.exists():
        prepare_continuation_programs(run_path, profile)
    manifest = json.loads(manifest_path.read_text())
    cases = read_continuation_programs(run_path, profile)
    if model is None or tokenizer is None:
        model, tokenizer = _load_evaluation_model(run_path, "explicit_handoff")
    prompt_config = load_config(run_path).get("state_handoff_training", {}).get(
        "prompt", {}
    )
    completed = {row["id"] for row in read_continuation_cases(run_path, profile)}
    pending = [
        (case, block_size)
        for case in cases
        for block_size in manifest["block_sizes"]
        if continuation_case_id(case, int(block_size)) not in completed
    ]
    if max_cases is not None:
        pending = pending[:max_cases]
    path = continuation_dir(run_path, profile) / "cases.jsonl"
    for index, (case, block_size) in enumerate(pending, 1):
        append_jsonl(
            path,
            evaluate_recursive_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=prompt_config,
                block_size=int(block_size),
            ),
        )
        if on_progress is not None and (
            index == 1 or index == len(pending) or index % 10 == 0
        ):
            on_progress(
                f"recursive handoff {profile} {index}/{len(pending)} cases"
            )
    rows = read_continuation_cases(run_path, profile)
    summary = summarize_continuation_rows(rows)
    expected = int(manifest["case_count"]) * len(manifest["block_sizes"])
    summary.update(
        profile=profile,
        expected_case_count=expected,
        complete=len(rows) == expected,
        manifest_sha256=manifest["sha256"],
    )
    write_json(continuation_dir(run_path, profile) / "summary.json", summary)
    return summary


def continuation_status(run_path: Path) -> dict[str, Any]:
    """Report prepared and completed continuation profiles without loading a model."""
    root = run_path / CONTINUATION_ROOT
    profiles = {}
    for path in sorted(root.iterdir()) if root.exists() else []:
        if not path.is_dir():
            continue
        manifest = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else None
        summary = json.loads((path / "summary.json").read_text()) if (path / "summary.json").exists() else None
        profiles[path.name] = {"manifest": manifest, "summary": summary}
    return {"run_path": str(run_path), "profiles": profiles}
