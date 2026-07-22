"""Explicit one-token state handoff evaluation from saved factorization cases."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples

from .benchmark import apply_rule, state_symbols
from .factorization import render_factorization_update_prompt
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .pipeline import read_factorization_results
from .qualification import evaluate_prompt_conditions_hf


RESULT_PATH = Path("depth_relief/explicit_handoff/cases.jsonl")
SUMMARY_PATH = Path("depth_relief/explicit_handoff/summary.json")
CONDITIONS = (
    "one_pass_compose",
    "synthesize",
    "oracle_executor_handoff",
    "lm_self_handoff",
    "gold_handoff",
    "stepwise_explicit",
)
DEFAULT_GATE = {
    "min_gold_accuracy": 0.90,
    "min_h2_self_improvement": 0.20,
    "min_h4_stepwise_accuracy": 0.90,
}


def discrete_capacity_bits(*, slots: int, codebook_size: int) -> int:
    """Return exact capacity for a power-of-two discrete codebook."""
    if slots < 1 or codebook_size < 2 or codebook_size & (codebook_size - 1):
        raise ValueError("Discrete capacity needs positive slots and a power-of-two codebook")
    return int(slots) * (int(codebook_size).bit_length() - 1)


def explicit_handoff_output_path(run_path: Path) -> Path:
    """Return the append-only explicit-handoff artifact path."""
    return run_path / RESULT_PATH


def _actual_prediction(condition: dict[str, Any]) -> int | None:
    value = condition.get("unconstrained_prediction")
    return int(value) if value is not None else None


def _invalid_condition(*, expected: int, reason: str) -> dict[str, Any]:
    return {
        "prediction": None,
        "unconstrained_prediction": None,
        "unconstrained_token_id": None,
        "candidate_probability_mass": None,
        "expected_next_state": int(expected),
        "is_expected": False,
        "is_expected_unconstrained": False,
        "skipped_reason": reason,
    }


def artifact_handoff_record(
    case: dict[str, Any], factorization_row: dict[str, Any]
) -> dict[str, Any]:
    """Derive Compose, Synthesize, and Python-executor results without inference."""
    if str(case["id"]) != str(factorization_row["id"]):
        raise ValueError("Factorization row and benchmark case IDs do not match")
    if discrete_capacity_bits(slots=1, codebook_size=len(state_symbols(case))) != 3:
        raise ValueError("Explicit handoff requires an exact three-bit state code")
    synthesize = dict(factorization_row["conditions"]["synthesize"])
    compose = dict(factorization_row["conditions"]["compose"])
    predicted_state = _actual_prediction(synthesize)
    if predicted_state is None:
        oracle = _invalid_condition(
            expected=int(case["next_state"]),
            reason="synthesize_output_is_not_a_state",
        )
    else:
        prediction = apply_rule(
            case["final_rule"], predicted_state, 2 ** int(case["bits"])
        )
        oracle = {
            "prediction": prediction,
            "unconstrained_prediction": prediction,
            "unconstrained_token_id": None,
            "candidate_probability_mass": None,
            "source_state_prediction": predicted_state,
            "expected_next_state": int(case["next_state"]),
            "is_expected": prediction == int(case["next_state"]),
            "is_expected_unconstrained": prediction == int(case["next_state"]),
        }
    return {
        "schema_version": 1,
        "id": str(case["id"]),
        "phase": "artifact",
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["abstraction_group"]),
        "program_context_split": str(case["abstraction_split"]),
        "current_state": int(case["current_state"]),
        "next_state": int(case["next_state"]),
        "conditions": {
            "one_pass_compose": compose,
            "synthesize": synthesize,
            "oracle_executor_handoff": oracle,
        },
    }


def _score_prompt(
    *,
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    return evaluate_prompt_conditions_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        candidate_symbols=state_symbols(case),
    )[str(prompt["name"])]


def _end_to_end_record(
    record: dict[str, Any], *, true_target: int, provided_target: int
) -> dict[str, Any]:
    result = dict(record)
    result.update(
        provided_state_expected_next_state=int(provided_target),
        is_provided_state_expected=(
            result.get("prediction") == int(provided_target)
        ),
        is_provided_state_expected_unconstrained=(
            _actual_prediction(result) == int(provided_target)
        ),
        expected_next_state=int(true_target),
        is_expected=result.get("prediction") == int(true_target),
        is_expected_unconstrained=(
            _actual_prediction(result) == int(true_target)
        ),
    )
    return result


def evaluate_explicit_handoff_case_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    factorization_row: dict[str, Any],
    config: dict[str, Any],
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run history-free self, gold, and sequential explicit-state calls."""
    if str(case["id"]) != str(factorization_row["id"]):
        raise ValueError("Factorization row and benchmark case IDs do not match")
    synthesize = factorization_row["conditions"]["synthesize"]
    predicted_state = _actual_prediction(synthesize)
    true_target = int(case["next_state"])
    modulus = 2 ** int(case["bits"])
    capacity = discrete_capacity_bits(
        slots=1, codebook_size=len(state_symbols(case))
    )
    if capacity != 3:
        raise ValueError("Explicit handoff requires an exact three-bit state code")

    def score(stage: str, prompt: dict[str, Any]) -> dict[str, Any]:
        if on_progress is not None:
            on_progress(stage)
        return _score_prompt(model=model, tokenizer=tokenizer, prompt=prompt, case=case)

    gold_prompt = render_factorization_update_prompt(
        tokenizer=tokenizer,
        case=case,
        config=config,
        state=int(case["current_state"]),
        rule=case["final_rule"],
        name="gold_handoff",
        label="FINAL",
    )
    gold = score("gold handoff", gold_prompt)

    if predicted_state is None:
        self_handoff = _invalid_condition(
            expected=true_target,
            reason="synthesize_output_is_not_a_state",
        )
    else:
        self_prompt = render_factorization_update_prompt(
            tokenizer=tokenizer,
            case=case,
            config=config,
            state=predicted_state,
            rule=case["final_rule"],
            name="lm_self_handoff",
            label="FINAL",
        )
        self_raw = score("self handoff", self_prompt)
        self_handoff = _end_to_end_record(
            self_raw,
            true_target=true_target,
            provided_target=apply_rule(
                case["final_rule"], predicted_state, modulus
            ),
        )

    step_records: list[dict[str, Any]] = []
    current: int | None = int(case["initial_state"])
    for index, rule in enumerate(case["history"], 1):
        if current is None:
            break
        prompt = render_factorization_update_prompt(
            tokenizer=tokenizer,
            case=case,
            config=config,
            state=current,
            rule=rule,
            name=f"history_step_{index}",
            label="Operation",
        )
        step = score(f"stepwise history {index}/{len(case['history'])}", prompt)
        step["gold_expected_next_state"] = int(case["state_path"][index])
        step["is_gold_expected_unconstrained"] = (
            _actual_prediction(step) == int(case["state_path"][index])
        )
        step_records.append(step)
        current = _actual_prediction(step)

    if current is None:
        stepwise = _invalid_condition(
            expected=true_target,
            reason="stepwise_output_is_not_a_state",
        )
    else:
        final_prompt = render_factorization_update_prompt(
            tokenizer=tokenizer,
            case=case,
            config=config,
            state=current,
            rule=case["final_rule"],
            name="stepwise_final",
            label="FINAL",
        )
        final_raw = score("stepwise final", final_prompt)
        stepwise = _end_to_end_record(
            final_raw,
            true_target=true_target,
            provided_target=apply_rule(case["final_rule"], current, modulus),
        )
    stepwise["history_calls"] = step_records
    stepwise["completed_history_calls"] = len(step_records)

    return {
        "schema_version": 1,
        "id": str(case["id"]),
        "phase": "inference",
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["abstraction_group"]),
        "program_context_split": str(case["abstraction_split"]),
        "prompt_contract": {
            "state_slots": 1,
            "codebook_size": len(state_symbols(case)),
            "capacity_bits": capacity,
            "second_call_inputs": ["state", "final_rule"],
            "original_history_deleted": True,
        },
        "conditions": {
            "lm_self_handoff": self_handoff,
            "gold_handoff": gold,
            "stepwise_explicit": stepwise,
        },
    }


def read_explicit_handoff_records(run_path: Path) -> list[dict[str, Any]]:
    """Read handoff events and reject duplicate case-phase identities."""
    path = explicit_handoff_output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [(str(row["id"]), str(row["phase"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate explicit-handoff case phases in {path}")
    if any(phase not in {"artifact", "inference"} for _, phase in keys):
        raise ValueError(f"Unknown explicit-handoff phase in {path}")
    return rows


def materialize_artifact_handoffs(run_path: Path) -> dict[str, int]:
    """Append missing Python-only handoff rows from completed source artifacts."""
    cases = {str(case["id"]): case for case in load_samples(run_path / "dataset.jsonl")}
    factorization = {
        str(row["id"]): row for row in read_factorization_results(run_path)
    }
    if set(cases) != set(factorization):
        raise ValueError("Explicit handoff needs one factorization row per source case")
    existing = {
        (str(row["id"]), str(row["phase"]))
        for row in read_explicit_handoff_records(run_path)
    }
    added = 0
    for case_id in cases:
        if (case_id, "artifact") in existing:
            continue
        append_jsonl(
            explicit_handoff_output_path(run_path),
            artifact_handoff_record(cases[case_id], factorization[case_id]),
        )
        added += 1
    return {"source_cases": len(cases), "artifact_rows_added": added}


def merge_explicit_handoff_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge append-only artifact and inference events into deterministic cases."""
    merged: dict[str, dict[str, Any]] = {}
    for row in records:
        case_id = str(row["id"])
        target = merged.setdefault(
            case_id,
            {
                "id": case_id,
                "history_steps": int(row["history_steps"]),
                "program_context": str(row["program_context"]),
                "program_context_split": str(row["program_context_split"]),
                "conditions": {},
                "phases": [],
            },
        )
        for field in ("history_steps", "program_context", "program_context_split"):
            if target[field] != row[field]:
                raise ValueError(f"Explicit-handoff metadata changed for {case_id}")
        overlap = set(target["conditions"]) & set(row["conditions"])
        if overlap:
            raise ValueError(
                f"Explicit-handoff conditions repeated for {case_id}: {sorted(overlap)}"
            )
        target["conditions"].update(row["conditions"])
        target["phases"].append(str(row["phase"]))
    return [merged[case_id] for case_id in sorted(merged)]


def _correct(row: dict[str, Any], condition: str) -> int:
    return int(row["conditions"][condition]["is_expected_unconstrained"])


def _condition_summary(
    rows: list[dict[str, Any]], condition: str, seed: int
) -> dict[str, Any]:
    available = [row for row in rows if condition in row["conditions"]]
    masses = [
        float(row["conditions"][condition]["candidate_probability_mass"])
        for row in available
        if row["conditions"][condition].get("candidate_probability_mass") is not None
    ]
    return {
        "accuracy": bootstrap_mean_ci(
            [_correct(row, condition) for row in available], seed=seed
        ),
        "invalid_output_rate": bootstrap_mean_ci(
            [
                _actual_prediction(row["conditions"][condition]) is None
                for row in available
            ],
            seed=seed + 1,
        ),
        "candidate_probability_mass": bootstrap_mean_ci(masses, seed=seed + 2),
    }


def _paired_vs_compose(
    rows: list[dict[str, Any]], condition: str, seed: int
) -> dict[str, Any]:
    paired = [
        row
        for row in rows
        if condition in row["conditions"]
        and "one_pass_compose" in row["conditions"]
    ]
    return cluster_bootstrap_mean_ci(
        [
            _correct(row, condition) - _correct(row, "one_pass_compose")
            for row in paired
        ],
        [str(row["program_context"]) for row in paired],
        seed=seed,
    )


def _group_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    return {
        "case_count": len(rows),
        "program_context_count": len(
            {str(row["program_context"]) for row in rows}
        ),
        "conditions": {
            condition: _condition_summary(rows, condition, seed + 10 * index)
            for index, condition in enumerate(CONDITIONS)
        },
        "paired_difference_vs_compose": {
            condition: _paired_vs_compose(rows, condition, seed + 100 + index)
            for index, condition in enumerate(CONDITIONS[1:])
        },
    }


def summarize_explicit_handoffs(
    rows: list[dict[str, Any]], gate_config: dict[str, Any]
) -> dict[str, Any]:
    """Aggregate handoff accuracy and apply the prespecified Phase 1 gate."""
    if not rows:
        raise ValueError("Cannot summarize empty explicit-handoff results")
    gate = {**DEFAULT_GATE, **gate_config}
    by_horizon = {
        str(horizon): _group_summary(
            [row for row in rows if int(row["history_steps"]) == horizon],
            1000 + horizon,
        )
        for horizon in sorted({int(row["history_steps"]) for row in rows})
    }
    by_split = {
        split: _group_summary(
            [row for row in rows if row["program_context_split"] == split],
            2000 + index,
        )
        for index, split in enumerate(
            sorted({str(row["program_context_split"]) for row in rows})
        )
    }
    by_horizon_split = {
        f"h{horizon}_{split}": _group_summary(
            [
                row
                for row in rows
                if int(row["history_steps"]) == horizon
                and row["program_context_split"] == split
            ],
            3000 + 10 * horizon + index,
        )
        for horizon in sorted({int(row["history_steps"]) for row in rows})
        for index, split in enumerate(sorted(by_split))
    }
    overall = _group_summary(rows, 500)
    phase_counts = Counter(phase for row in rows for phase in row["phases"])
    inference_complete = phase_counts["inference"] == len(rows)
    required_horizons = {"2", "4"}
    gate_ready = inference_complete and required_horizons.issubset(by_horizon)
    checks: dict[str, bool | None]
    if gate_ready:
        gold = overall["conditions"]["gold_handoff"]["accuracy"]
        h2_delta = by_horizon["2"]["paired_difference_vs_compose"][
            "lm_self_handoff"
        ]
        h4_stepwise = by_horizon["4"]["conditions"]["stepwise_explicit"][
            "accuracy"
        ]
        checks = {
            "gold_handoff_accuracy": float(gold["mean"])
            >= float(gate["min_gold_accuracy"]),
            "h2_self_improvement": float(h2_delta["mean"])
            >= float(gate["min_h2_self_improvement"]),
            "h2_program_context_ci_excludes_zero": float(h2_delta["ci95"][0]) > 0,
            "h4_stepwise_accuracy": float(h4_stepwise["mean"])
            >= float(gate["min_h4_stepwise_accuracy"]),
        }
    else:
        checks = {
            "gold_handoff_accuracy": None,
            "h2_self_improvement": None,
            "h2_program_context_ci_excludes_zero": None,
            "h4_stepwise_accuracy": None,
        }
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "program_context_count": len(
            {str(row["program_context"]) for row in rows}
        ),
        "phase_counts": dict(sorted(phase_counts.items())),
        "overall": overall,
        "by_horizon": by_horizon,
        "by_program_context_split": by_split,
        "by_horizon_and_program_context_split": by_horizon_split,
        "phase1_gate": {
            "status": (
                "pending_inference"
                if not gate_ready
                else "passed" if all(checks.values()) else "failed"
            ),
            "thresholds": gate,
            "checks": checks,
        },
    }


def analyze_explicit_handoff(run_path: Path) -> dict[str, Any]:
    """Materialize Python-only rows and write a deterministic saved-artifact summary."""
    materialize_artifact_handoffs(run_path)
    config = load_config(run_path).get("explicit_handoff", {})
    rows = merge_explicit_handoff_records(read_explicit_handoff_records(run_path))
    summary = summarize_explicit_handoffs(rows, config.get("gate", {}))
    write_json(run_path / SUMMARY_PATH, summary)
    return summary


def explicit_handoff_status(run_path: Path) -> dict[str, Any]:
    """Report artifact and inference completion without loading a model."""
    total = len(load_samples(run_path / "dataset.jsonl"))
    records = read_explicit_handoff_records(run_path)
    counts = Counter(str(row["phase"]) for row in records)
    summary_path = run_path / SUMMARY_PATH
    gate_status = None
    if summary_path.exists():
        gate_status = json.loads(summary_path.read_text())["phase1_gate"]["status"]
    return {
        "run_path": str(run_path),
        "total_cases": total,
        "artifact_cases": counts["artifact"],
        "inference_cases": counts["inference"],
        "remaining_inference_cases": total - counts["inference"],
        "summary_exists": summary_path.exists(),
        "phase1_gate_status": gate_status,
    }
