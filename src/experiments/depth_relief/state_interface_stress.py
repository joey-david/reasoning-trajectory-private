"""Out-of-template stress tests for saved decimal and opaque state interfaces."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import write_jsonl

from .benchmark import apply_rule
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .state_handoff_continuation import evaluate_recursive_program_hf
from .state_handoff_data import build_test_programs
from .state_handoff_evaluation import _load_evaluation_model
from .state_interface_data import semantic_states_for_code
from .state_interface_evaluation import evaluate_interface_program_hf


STRESS_ROOT = Path("evaluation/stress")
STRESS_FAMILIES = ("structured", "iid", "shuffled", "cancellation", "repeated")


def stress_dir(run_path: Path, profile: str) -> Path:
    """Return the artifact owner for one stress-test profile."""
    if not profile or any(part in {"", ".", ".."} for part in Path(profile).parts):
        raise ValueError(f"Invalid stress profile: {profile!r}")
    return run_path / STRESS_ROOT / profile


def stress_config(run_path: Path) -> dict[str, Any]:
    """Read and validate the cross-adapter stress-test contract."""
    config = load_config(run_path).get("state_interface_stress", {})
    conditions = config.get("conditions", {})
    if not conditions:
        raise ValueError("Stress testing requires at least one source condition")
    for name, values in conditions.items():
        if values.get("kind") not in {"decimal", "interface"}:
            raise ValueError(f"Unknown stress condition kind for {name}")
        if not values.get("source_run") or not values.get("source_condition"):
            raise ValueError(f"Stress condition {name} lacks its source adapter")
    return config


def stress_profile(run_path: Path, profile: str) -> dict[str, Any]:
    """Resolve one deterministic stress profile."""
    configured = stress_config(run_path).get("profiles", {}).get(profile)
    if not configured:
        raise ValueError(f"Unknown stress profile: {profile!r}")
    values = dict(configured)
    values["horizons"] = [int(value) for value in values["horizons"]]
    values["families"] = [str(value) for value in values["families"]]
    unknown = sorted(set(values["families"]) - set(STRESS_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown stress history families: {unknown}")
    if any(horizon < 2 or horizon % 2 for horizon in values["horizons"]):
        raise ValueError("Stress horizons must be positive multiples of two")
    return values


def _rng(seed: int, *parts: Any) -> random.Random:
    payload = ":".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _endpoint(initial: int, values: list[int], modulus: int) -> int:
    return (initial + sum(values)) % modulus


def _stress_values(case: dict[str, Any], family: str, seed: int) -> list[int]:
    """Generate one held-out operation pattern with the saved endpoint."""
    modulus = 2 ** int(case["bits"])
    horizon = int(case["history_steps"])
    initial = int(case["initial_state"])
    target = int(case["current_state"])
    if family == "structured":
        return [int(rule["value"]) for rule in case["history"]]
    rng = _rng(seed, case["id"], family)
    if family == "iid":
        for _ in range(10_000):
            values = [rng.randrange(modulus) for _ in range(horizon)]
            if _endpoint(initial, values, modulus) == target:
                return values
        raise RuntimeError("IID rejection sampling failed to reach the endpoint")
    if family == "shuffled":
        values = [rng.randrange(modulus) for _ in range(horizon - 1)]
        values.append((target - _endpoint(initial, values, modulus)) % modulus)
        rng.shuffle(values)
        return values
    if family == "cancellation":
        values = []
        for _ in range(horizon // 2):
            value = rng.randrange(modulus)
            values.extend((value, (-value) % modulus))
        position = rng.randrange(horizon)
        values[position] = (
            values[position] + target - _endpoint(initial, values, modulus)
        ) % modulus
        rng.shuffle(values)
        return values
    if family == "repeated":
        value = rng.randrange(modulus)
        values = [value] * (horizon - 1)
        values.append((target - _endpoint(initial, values, modulus)) % modulus)
        shift = rng.randrange(horizon)
        return values[shift:] + values[:shift]
    raise ValueError(f"Unknown stress family: {family!r}")


def _with_stress_history(
    case: dict[str, Any], family: str, seed: int
) -> dict[str, Any]:
    modulus = 2 ** int(case["bits"])
    values = _stress_values(case, family, seed)
    history = [{"kind": "add", "value": value} for value in values]
    states = [int(case["initial_state"])]
    for rule in history:
        states.append(apply_rule(rule, states[-1], modulus))
    if states[-1] != int(case["current_state"]):
        raise AssertionError("Stress history changed its requested endpoint")
    semantic = {
        **case,
        "history": history,
        "state_path": states,
        "stress_family": family,
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    semantic["id"] = f"{case['id']}__{family}_{digest}"
    return semantic


def prepare_stress_profile(run_path: Path, profile: str) -> dict[str, Any]:
    """Write balanced structured and out-of-template histories."""
    values = stress_profile(run_path, profile)
    base = build_test_programs(
        horizons=tuple(values["horizons"]),
        context_count=int(values["context_count"]),
        paths_per_state=int(values["paths_per_state"]),
        width=int(values.get("bits", 3)),
        seed=int(values["seed"]),
        split=f"stress_{profile}",
    )
    cases = [
        _with_stress_history(case, family, int(values["seed"]))
        for family in values["families"]
        for case in base
    ]
    output = stress_dir(run_path, profile)
    programs_path = output / "programs.jsonl"
    write_jsonl(programs_path, cases)
    manifest = {
        "schema_version": 1,
        "profile": profile,
        "case_count": len(cases),
        "base_case_count": len(base),
        "program_context_count": len({case["program_context"] for case in cases}),
        "families": values["families"],
        "horizons": values["horizons"],
        "paths_per_state": int(values["paths_per_state"]),
        "sha256": hashlib.sha256(programs_path.read_bytes()).hexdigest(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def read_stress_programs(run_path: Path, profile: str) -> list[dict[str, Any]]:
    """Read one prepared stress bank and reject duplicate IDs."""
    path = stress_dir(run_path, profile) / "programs.jsonl"
    if not path.exists():
        prepare_stress_profile(run_path, profile)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate stress program IDs in {path}")
    return rows


def stress_condition_dir(run_path: Path, profile: str, condition: str) -> Path:
    """Return one stress condition's append-only artifact directory."""
    return stress_dir(run_path, profile) / condition


def read_stress_cases(
    run_path: Path, profile: str, condition: str
) -> list[dict[str, Any]]:
    """Read saved stress rows and reject duplicate cases."""
    path = stress_condition_dir(run_path, profile, condition) / "cases.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate stress evaluation IDs in {path}")
    return rows


def _interface_semantic_correct(
    *, result: dict[str, Any], case: dict[str, Any], condition: str, config: dict[str, Any]
) -> bool:
    code = result["predicted_code"]
    if code is None:
        return False
    return int(case["current_state"]) in semantic_states_for_code(
        condition=condition,
        case=case,
        code_index=int(code),
        interface_config=config,
    )


def _interface_local_semantic(
    *, step: dict[str, Any], case: dict[str, Any], condition: str, config: dict[str, Any]
) -> bool:
    predicted = step.get("unconstrained_prediction")
    if predicted is None:
        return False
    expected = int(step["local_expected_code"])
    return semantic_states_for_code(
        condition=condition,
        case=case,
        code_index=int(predicted),
        interface_config=config,
    ) == semantic_states_for_code(
        condition=condition,
        case=case,
        code_index=expected,
        interface_config=config,
    )


def _standard_row(
    *,
    case: dict[str, Any],
    source: dict[str, Any],
    result: dict[str, Any],
    interface_config: dict[str, Any],
) -> dict[str, Any]:
    if source["kind"] == "decimal":
        local = [bool(step["locally_correct"]) for step in result["predicted_steps"]]
        predicted = result["predicted_state"]
        semantic_set = (
            [int(predicted)] if predicted is not None else []
        )
        semantic_correct = bool(result["state_correct"])
        answer_correct = bool(result["final"]["is_expected_unconstrained"])
    else:
        condition = str(source["source_condition"])
        local = [
            _interface_local_semantic(
                step=step,
                case=case,
                condition=condition,
                config=interface_config,
            )
            for step in result["steps"]
        ]
        predicted = result["predicted_code"]
        semantic_set = (
            list(
                semantic_states_for_code(
                    condition=condition,
                    case=case,
                    code_index=int(predicted),
                    interface_config=interface_config,
                )
            )
            if predicted is not None
            else []
        )
        semantic_correct = _interface_semantic_correct(
            result=result,
            case=case,
            condition=condition,
            config=interface_config,
        )
        answer_correct = bool(
            result["predicted_final"]
            and result["predicted_final"]["is_expected_unconstrained"]
        )
    return {
        "schema_version": 1,
        "id": str(case["id"]),
        "stress_family": str(case["stress_family"]),
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["program_context"]),
        "path_code": int(case["path_code"]),
        "current_state": int(case["current_state"]),
        "predicted_representation": predicted,
        "predicted_semantic_set": semantic_set,
        "semantic_state_correct": semantic_correct,
        "answer_correct": answer_correct,
        "local_semantic_correct": local,
        "raw_result": result,
    }


def summarize_stress_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report semantic closure and path agreement by family and horizon."""
    if not rows:
        raise ValueError("Cannot summarize an empty stress evaluation")
    cells = {}
    for family, horizon in sorted(
        {(str(row["stress_family"]), int(row["history_steps"])) for row in rows}
    ):
        selected = [
            row
            for row in rows
            if row["stress_family"] == family
            and int(row["history_steps"]) == horizon
        ]
        clusters = [str(row["program_context"]) for row in selected]
        local = [value for row in selected for value in row["local_semantic_correct"]]
        groups: defaultdict[tuple[str, int], list[Any]] = defaultdict(list)
        semantic_groups: defaultdict[tuple[str, int], list[tuple[int, ...]]] = (
            defaultdict(list)
        )
        for row in selected:
            key = (str(row["program_context"]), int(row["current_state"]))
            groups[key].append(row["predicted_representation"])
            semantic_groups[key].append(tuple(row["predicted_semantic_set"]))
        agreement = []
        for values in groups.values():
            for index, left in enumerate(values):
                agreement.extend(
                    left is not None and left == right for right in values[index + 1 :]
                )
        quotient_agreement = []
        for values in semantic_groups.values():
            for index, left in enumerate(values):
                quotient_agreement.extend(
                    bool(left) and left == right for right in values[index + 1 :]
                )
        cells[f"{family}_h{horizon}"] = {
            "case_count": len(selected),
            "semantic_state_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["semantic_state_correct"]) for row in selected],
                clusters,
                seed=73_000 + horizon,
            ),
            "answer_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["answer_correct"]) for row in selected],
                clusters,
                seed=73_100 + horizon,
            ),
            "local_semantic_closure": bootstrap_mean_ci(
                local, seed=73_200 + horizon
            ),
            "exact_representation_agreement": bootstrap_mean_ci(
                agreement, seed=73_300 + horizon
            ),
            "quotient_representation_agreement": bootstrap_mean_ci(
                quotient_agreement, seed=73_400 + horizon
            ),
        }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "program_context_count": len({row["program_context"] for row in rows}),
        "by_cell": cells,
    }


def evaluate_stress_condition(
    run_path: Path,
    profile: str,
    condition: str,
    *,
    max_cases: int | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate one saved adapter on every prepared history family."""
    config = stress_config(run_path)
    source = config["conditions"].get(condition)
    if source is None:
        raise ValueError(f"Unknown stress condition: {condition!r}")
    source_run = Path(str(source["source_run"]))
    source_condition = str(source["source_condition"])
    model, tokenizer = _load_evaluation_model(source_run, source_condition)
    source_experiment = load_config(source_run).get("state_handoff_training", {})
    prompt_config = source_experiment.get("prompt", {})
    interface_config = source_experiment.get("interfaces", {})
    cases = read_stress_programs(run_path, profile)
    completed = {
        row["id"] for row in read_stress_cases(run_path, profile, condition)
    }
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    output = stress_condition_dir(run_path, profile, condition) / "cases.jsonl"
    for index, case in enumerate(pending, 1):
        if source["kind"] == "decimal":
            result = evaluate_recursive_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=prompt_config,
                block_size=2,
            )
        else:
            result = evaluate_interface_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=prompt_config,
                condition=source_condition,
                interface_config=interface_config,
                block_size=2,
            )
        append_jsonl(
            output,
            _standard_row(
                case=case,
                source=source,
                result=result,
                interface_config=interface_config,
            ),
        )
        if on_progress and (index == 1 or index == len(pending) or index % 10 == 0):
            on_progress(f"stress {condition} {index}/{len(pending)}")
    rows = read_stress_cases(run_path, profile, condition)
    summary = summarize_stress_rows(rows)
    summary.update(
        condition=condition,
        profile=profile,
        expected_case_count=len(cases),
        complete=len(rows) == len(cases),
    )
    write_json(stress_condition_dir(run_path, profile, condition) / "summary.json", summary)
    return summary


def compare_stress_conditions(run_path: Path, profile: str) -> dict[str, Any]:
    """Collect complete stress summaries without rerunning model inference."""
    conditions = stress_config(run_path)["conditions"]
    summaries = {}
    for condition in conditions:
        path = stress_condition_dir(run_path, profile, condition) / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing stress summary for {condition}")
        summary = json.loads(path.read_text())
        if not summary.get("complete"):
            raise RuntimeError(f"Stress evaluation is incomplete for {condition}")
        summaries[condition] = summary
    result = {
        "schema_version": 1,
        "profile": profile,
        "conditions": list(conditions),
        "summaries": summaries,
    }
    write_json(stress_dir(run_path, profile) / "comparison_summary.json", result)
    return result
