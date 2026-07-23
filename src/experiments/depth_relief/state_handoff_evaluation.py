"""Saved-artifact evaluation and comparison for trained state-handoff adapters."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Callable

from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config

from .benchmark import answer_symbols, apply_rule, state_symbols
from .factorization import (
    render_factorization_prompts,
    render_factorization_update_prompt,
)
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .qualification import evaluate_prompt_conditions_hf
from .state_handoff_data import COMPUTE_MANIFEST_PATH, TEST_PATH, read_programs
from .state_handoff_reporting import (
    _frozen_screen_summary,
    _write_comparison_plots,
)
from .state_handoff_training import condition_training_dir


EVALUATION_CONDITIONS = ("outcome_only", "explicit_handoff")
DEFAULT_PILOT_GATE = {
    "min_ood_improvement": 0.10,
    "min_same_state_agreement": 0.90,
    "min_gold_gap_closed": 0.50,
}


def condition_evaluation_dir(run_path: Path, condition: str) -> Path:
    """Return the saved evaluation owner for one training condition."""
    if condition not in EVALUATION_CONDITIONS:
        raise ValueError(f"Unknown state-handoff evaluation condition: {condition!r}")
    return run_path / "evaluation" / condition


def read_evaluation_cases(run_path: Path, condition: str) -> list[dict[str, Any]]:
    """Read append-only evaluation rows and reject duplicate case IDs."""
    path = condition_evaluation_dir(run_path, condition) / "cases.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate state-handoff evaluation IDs in {path}")
    return rows


def _prediction(condition: dict[str, Any]) -> int | None:
    value = condition.get("unconstrained_prediction")
    return int(value) if value is not None else None


def _score(
    *, model: Any, tokenizer: Any, prompt: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    return evaluate_prompt_conditions_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        candidate_symbols=(
            answer_symbols(case)
            if prompt.get("output_kind") == "answer"
            else state_symbols(case)
        ),
    )[str(prompt["name"])]


def _invalid(expected: int, reason: str) -> dict[str, Any]:
    return {
        "prediction": None,
        "unconstrained_prediction": None,
        "unconstrained_token_id": None,
        "candidate_probability_mass": None,
        "expected_next_state": expected,
        "is_expected": False,
        "is_expected_unconstrained": False,
        "skipped_reason": reason,
    }


def evaluate_program_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
) -> dict[str, Any]:
    """Evaluate one saved test program without any training-data dependency."""
    prompts = {
        str(prompt["name"]): prompt
        for prompt in render_factorization_prompts(
            tokenizer=tokenizer, case=case, config=prompt_config
        )
    }
    if condition == "outcome_only":
        conditions = {
            "one_pass_compose": _score(
                model=model,
                tokenizer=tokenizer,
                prompt=prompts["compose"],
                case=case,
            )
        }
    elif condition == "explicit_handoff":
        state = _score(
            model=model,
            tokenizer=tokenizer,
            prompt=prompts["synthesize"],
            case=case,
        )
        gold_prompt = render_factorization_update_prompt(
            tokenizer=tokenizer,
            case=case,
            config=prompt_config,
            state=int(case["current_state"]),
            rule=case["final_rule"],
            name="gold_handoff",
            label="FINAL",
        )
        gold = _score(
            model=model, tokenizer=tokenizer, prompt=gold_prompt, case=case
        )
        predicted_state = _prediction(state)
        if predicted_state is None:
            predicted = _invalid(
                int(case["next_state"]), "predicted_state_is_not_in_codebook"
            )
        else:
            predicted_prompt = render_factorization_update_prompt(
                tokenizer=tokenizer,
                case=case,
                config=prompt_config,
                state=predicted_state,
                rule=case["final_rule"],
                name="predicted_handoff",
                label="FINAL",
            )
            predicted = _score(
                model=model,
                tokenizer=tokenizer,
                prompt=predicted_prompt,
                case=case,
            )
            provided_target = apply_rule(
                case["final_rule"], predicted_state, 2 ** int(case["bits"])
            )
            predicted["provided_state_expected_next_state"] = provided_target
            predicted["is_provided_state_expected_unconstrained"] = (
                _prediction(predicted) == provided_target
            )
            predicted["expected_next_state"] = int(case["next_state"])
            predicted["is_expected"] = (
                predicted.get("prediction") == int(case["next_state"])
            )
            predicted["is_expected_unconstrained"] = (
                _prediction(predicted) == int(case["next_state"])
            )
        conditions = {
            "state": state,
            "gold_handoff": gold,
            "predicted_handoff": predicted,
        }
    else:
        raise ValueError(f"Unknown state-handoff evaluation condition: {condition!r}")
    return {
        "schema_version": 1,
        "id": str(case["id"]),
        "condition": condition,
        "history_steps": int(case["history_steps"]),
        "bits": int(case["bits"]),
        "program_context": str(case["program_context"]),
        "program_context_split": str(case["program_context_split"]),
        "domain": str(case.get("domain", "addition")),
        "composition_split": str(case.get("composition_split", "seen")),
        "proof_composition_active": bool(
            case.get("proof_composition_active", False)
        ),
        "history_family": str(case["history_family"]),
        "path_code": int(case["path_code"]),
        "current_state": int(case["current_state"]),
        "next_state": int(case["next_state"]),
        "conditions": conditions,
    }


def _load_evaluation_model(run_path: Path, condition: str) -> tuple[Any, Any]:
    from peft import PeftModel

    config = load_config(run_path)
    manifest_path = condition_training_dir(run_path, condition) / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    preference = config.get("state_handoff_training", {}).get(
        "evaluation_adapter", "best"
    )
    if preference == "best" and manifest.get("best_checkpoint"):
        adapter = Path(manifest["best_checkpoint"]) / "adapter"
    else:
        adapter = Path(manifest["final_adapter"])
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    return PeftModel.from_pretrained(model, adapter).eval(), tokenizer


def evaluate_state_handoff_condition(
    run_path: Path,
    condition: str,
    *,
    max_cases: int | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resume test inference for one adapter and write its summary."""
    if model is None or tokenizer is None:
        model, tokenizer = _load_evaluation_model(run_path, condition)
    config = load_config(run_path).get("state_handoff_training", {})
    cases = read_programs(run_path / TEST_PATH)
    completed = {str(row["id"]) for row in read_evaluation_cases(run_path, condition)}
    pending = [case for case in cases if str(case["id"]) not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    output = condition_evaluation_dir(run_path, condition) / "cases.jsonl"
    for index, case in enumerate(pending, 1):
        append_jsonl(
            output,
            evaluate_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=config.get("prompt", {}),
                condition=condition,
            ),
        )
        if on_progress is not None and (
            index == 1 or index == len(pending) or index % 25 == 0
        ):
            on_progress(
                f"state handoff evaluation {condition} {index}/{len(pending)} cases"
            )
    rows = read_evaluation_cases(run_path, condition)
    summary = summarize_evaluation_rows(rows, condition)
    summary["expected_case_count"] = len(cases)
    summary["complete"] = len(rows) == len(cases)
    write_json(condition_evaluation_dir(run_path, condition) / "summary.json", summary)
    return summary


def _accuracy(
    rows: list[dict[str, Any]], name: str, seed: int
) -> dict[str, Any]:
    available = [row for row in rows if name in row["conditions"]]
    return cluster_bootstrap_mean_ci(
        [row["conditions"][name]["is_expected_unconstrained"] for row in available],
        [str(row["program_context"]) for row in available],
        seed=seed,
    )


def _candidate_mass(rows: list[dict[str, Any]], name: str, seed: int) -> dict[str, Any]:
    values = [
        float(row["conditions"][name]["candidate_probability_mass"])
        for row in rows
        if name in row["conditions"]
        and row["conditions"][name].get("candidate_probability_mass") is not None
    ]
    return bootstrap_mean_ci(values, seed=seed)


def _pairwise_code_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same_values = []
    same_groups: defaultdict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    different_groups: defaultdict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        same_groups[
            (int(row["history_steps"]), str(row["program_context"]), int(row["current_state"]))
        ].append(row)
        different_groups[
            (int(row["history_steps"]), str(row["program_context"]), int(row["path_code"]))
        ].append(row)
    for group in same_groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_code = _prediction(left["conditions"]["state"])
                right_code = _prediction(right["conditions"]["state"])
                same_values.append(
                    left_code is not None and left_code == right_code
                )
    different_values = []
    for group in different_groups.values():
        for left_index, left in enumerate(group):
            for right in group[left_index + 1 :]:
                left_code = _prediction(left["conditions"]["state"])
                right_code = _prediction(right["conditions"]["state"])
                different_values.append(
                    left_code is not None
                    and right_code is not None
                    and left_code != right_code
                )
    return {
        "same_state_code_agreement": bootstrap_mean_ci(same_values, seed=4100),
        "different_state_code_separation": bootstrap_mean_ci(
            different_values, seed=4101
        ),
    }


def summarize_evaluation_rows(
    rows: list[dict[str, Any]], condition: str
) -> dict[str, Any]:
    """Summarize ID/OOD accuracy, code agreement, and codebook use."""
    if not rows:
        raise ValueError("Cannot summarize empty state-handoff evaluation")
    primary = "one_pass_compose" if condition == "outcome_only" else "predicted_handoff"
    horizons = sorted({int(row["history_steps"]) for row in rows})
    by_horizon = {}
    for index, horizon in enumerate(horizons):
        selected = [row for row in rows if int(row["history_steps"]) == horizon]
        summary = {
            "case_count": len(selected),
            "program_context_count": len(
                {str(row["program_context"]) for row in selected}
            ),
            "accuracy": _accuracy(selected, primary, 3000 + index),
            "candidate_probability_mass": _candidate_mass(
                selected, primary, 3050 + index
            ),
        }
        if condition == "explicit_handoff":
            summary.update(
                state_accuracy=_accuracy(selected, "state", 3100 + index),
                gold_code_continuation_accuracy=_accuracy(
                    selected, "gold_handoff", 3150 + index
                ),
                predicted_code_continuation_accuracy=summary["accuracy"],
                **_pairwise_code_metrics(selected),
            )
        by_horizon[str(horizon)] = summary
    result = {
        "schema_version": 1,
        "condition": condition,
        "case_count": len(rows),
        "program_context_count": len({str(row["program_context"]) for row in rows}),
        "in_distribution_horizon": 2,
        "ood_horizons": [4, 8],
        "unseen_program_context_accuracy": _accuracy(rows, primary, 3200),
        "by_horizon": by_horizon,
    }
    if condition == "explicit_handoff":
        predictions = [_prediction(row["conditions"]["state"]) for row in rows]
        valid = [value for value in predictions if value is not None]
        counts = Counter(valid)
        result.update(
            _pairwise_code_metrics(rows),
            codebook_utilization={
                "used_codes": sorted(counts),
                "used_code_count": len(counts),
                "utilization_rate": len(counts) / 8,
                "invalid_count": len(predictions) - len(valid),
                "max_code_frequency": max(counts.values(), default=0) / max(len(valid), 1),
                "code_counts": {str(code): counts[code] for code in range(8)},
            },
        )
    return result


def build_code_donors(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose deterministic same-state and different-state donor programs."""
    indexed = {str(case["id"]): case for case in cases}
    donors = []
    for case_id in sorted(indexed):
        case = indexed[case_id]
        compatible = [
            candidate
            for candidate in cases
            if candidate["program_context"] == case["program_context"]
            and int(candidate["history_steps"]) == int(case["history_steps"])
        ]
        same = next(
            candidate
            for candidate in compatible
            if int(candidate["current_state"]) == int(case["current_state"])
            and int(candidate["path_code"]) != int(case["path_code"])
        )
        different = next(
            candidate
            for candidate in compatible
            if int(candidate["path_code"]) == int(case["path_code"])
            and int(candidate["current_state"])
            == (int(case["current_state"]) + 1) % 8
        )
        donors.append(
            {
                "recipient_id": case_id,
                "same_state_donor_id": str(same["id"]),
                "different_state_donor_id": str(different["id"]),
                "recipient_state": int(case["current_state"]),
                "different_state": int(different["current_state"]),
            }
        )
    return donors


def compare_state_handoff_conditions(run_path: Path) -> dict[str, Any]:
    """Compare matched pilot cases, apply the gate, and render final pilot plots."""
    for condition in EVALUATION_CONDITIONS:
        path = condition_evaluation_dir(run_path, condition) / "summary.json"
        if not path.exists() or not json.loads(path.read_text()).get("complete"):
            raise RuntimeError(f"Evaluation is incomplete for {condition}")
    outcome = {str(row["id"]): row for row in read_evaluation_cases(run_path, "outcome_only")}
    explicit = {str(row["id"]): row for row in read_evaluation_cases(run_path, "explicit_handoff")}
    shared = sorted(set(outcome) & set(explicit))
    if not shared:
        raise ValueError("Training conditions have no shared evaluation cases")
    config = load_config(run_path).get("state_handoff_training", {})
    gate = {**DEFAULT_PILOT_GATE, **config.get("pilot_gate", {})}
    differences = {}
    accuracies = {}
    gap_closed = {}
    for horizon in (2, 4, 8):
        ids = [case_id for case_id in shared if int(outcome[case_id]["history_steps"]) == horizon]
        clusters = [str(outcome[case_id]["program_context"]) for case_id in ids]
        outcome_values = [
            int(outcome[case_id]["conditions"]["one_pass_compose"]["is_expected_unconstrained"])
            for case_id in ids
        ]
        predicted_values = [
            int(explicit[case_id]["conditions"]["predicted_handoff"]["is_expected_unconstrained"])
            for case_id in ids
        ]
        gold_values = [
            int(explicit[case_id]["conditions"]["gold_handoff"]["is_expected_unconstrained"])
            for case_id in ids
        ]
        differences[str(horizon)] = cluster_bootstrap_mean_ci(
            [right - left for left, right in zip(outcome_values, predicted_values)],
            clusters,
            seed=5000 + horizon,
        )
        accuracies[str(horizon)] = {
            "outcome_only": sum(outcome_values) / len(outcome_values),
            "explicit_handoff_predicted": sum(predicted_values) / len(predicted_values),
            "explicit_handoff_gold": sum(gold_values) / len(gold_values),
        }
        denominator = accuracies[str(horizon)]["explicit_handoff_gold"] - accuracies[str(horizon)]["outcome_only"]
        gap_closed[str(horizon)] = (
            (accuracies[str(horizon)]["explicit_handoff_predicted"] - accuracies[str(horizon)]["outcome_only"])
            / denominator
            if denominator > 0
            else None
        )
    explicit_summary = summarize_evaluation_rows(list(explicit.values()), "explicit_handoff")
    compute = json.loads((run_path / COMPUTE_MANIFEST_PATH).read_text())
    ood_pass = any(
        float(differences[str(horizon)]["mean"]) >= float(gate["min_ood_improvement"])
        and float(differences[str(horizon)]["ci95"][0]) > 0
        for horizon in (4, 8)
    )
    checks = {
        "ood_improvement_and_positive_ci": ood_pass,
        "predicted_approaches_gold": any(
            gap_closed[str(horizon)] is not None
            and gap_closed[str(horizon)] >= float(gate["min_gold_gap_closed"])
            for horizon in (4, 8)
        ),
        "same_state_agreement": float(
            explicit_summary["same_state_code_agreement"]["mean"]
        )
        >= float(gate["min_same_state_agreement"]),
        "matched_compute": bool(
            compute["matched_forward_passes_and_tokens"]
        ),
    }
    summary = {
        "schema_version": 1,
        "shared_case_count": len(shared),
        "frozen_screen": _frozen_screen_summary(run_path),
        "horizon_accuracy": accuracies,
        "explicit_minus_outcome": differences,
        "gold_gap_closed": gap_closed,
        "pilot_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "thresholds": gate,
            "checks": checks,
        },
    }
    write_json(run_path / "evaluation/comparison_summary.json", summary)
    _write_comparison_plots(run_path, summary)
    return summary
