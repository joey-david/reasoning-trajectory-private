"""Recursive evaluation for rate-controlled opaque state interfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Callable

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config

from .benchmark import apply_rule
from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .qualification import evaluate_prompt_conditions_hf
from .state_handoff_data import INTERFACE_CONDITIONS, TEST_PATH, read_programs
from .state_handoff_evaluation import _load_evaluation_model
from .state_interface_data import (
    CODEBOOK_SIZES,
    interface_code_index,
    interface_code_symbols,
    render_interface_consumer_prompt,
    render_interface_encoder_prompt,
    render_interface_transition_prompt,
    semantic_states_for_code,
)


INTERFACE_EVALUATION_ROOT = Path("evaluation/interfaces")


def interface_evaluation_dir(run_path: Path, condition: str) -> Path:
    """Return the saved evaluation owner for one code condition."""
    if condition not in INTERFACE_CONDITIONS:
        raise ValueError(f"Unknown state-interface condition: {condition!r}")
    return run_path / INTERFACE_EVALUATION_ROOT / condition


def read_interface_evaluation_cases(
    run_path: Path, condition: str
) -> list[dict[str, Any]]:
    """Read append-only interface rows and reject duplicate case IDs."""
    path = interface_evaluation_dir(run_path, condition) / "cases.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate state-interface evaluation IDs in {path}")
    return rows


def _score(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    candidates: tuple[str, ...],
    expected: int,
    name: str,
) -> dict[str, Any]:
    prompt = {"name": name, "text": text, "expected_next_state": expected}
    return evaluate_prompt_conditions_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        candidate_symbols=candidates,
    )[name]


def _prediction(row: dict[str, Any]) -> int | None:
    value = row.get("unconstrained_prediction")
    return int(value) if value is not None else None


def _block(case: dict[str, Any], start: int, block_size: int) -> dict[str, Any]:
    history = list(case["history"][start : start + block_size])
    return {
        **case,
        "history": history,
        "history_steps": len(history),
        "id": f"{case['id']}__block{start // block_size}",
    }


def _advance_states(
    states: tuple[int, ...], history: list[dict[str, Any]]
) -> tuple[int, ...]:
    result = []
    for state in states:
        for rule in history:
            state = apply_rule(rule, state, 8)
        result.append(state)
    return tuple(sorted(set(result)))


def evaluate_interface_program_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
    interface_config: dict[str, Any],
    block_size: int = 2,
) -> dict[str, Any]:
    """Encode once, recur through opaque codes, then consume the final code."""
    if int(case["history_steps"]) % block_size:
        raise ValueError("Interface evaluation requires complete blocks")
    symbols = interface_code_symbols(condition, interface_config)
    variant = int(case["path_code"]) % 2
    predicted_code = None
    steps = []
    for start in range(0, int(case["history_steps"]), block_size):
        local = _block(case, start, block_size)
        true_state = int(case["state_path"][start + block_size])
        global_expected = interface_code_index(
            condition=condition,
            case=case,
            state=true_state,
            interface_config=interface_config,
            variant=variant,
        )
        if start == 0:
            text = render_interface_encoder_prompt(
                tokenizer=tokenizer,
                case={
                    **local,
                    "initial_state": int(case["initial_state"]),
                },
                prompt_config=prompt_config,
                condition=condition,
            )
            local_expected = global_expected
            compatible_inputs = (int(case["initial_state"]),)
        elif predicted_code is None:
            break
        else:
            compatible_inputs = semantic_states_for_code(
                condition=condition,
                case=case,
                code_index=predicted_code,
                interface_config=interface_config,
            )
            compatible_outputs = _advance_states(compatible_inputs, local["history"])
            possible_codes = {
                interface_code_index(
                    condition=condition,
                    case=case,
                    state=state,
                    interface_config=interface_config,
                    variant=variant,
                )
                for state in compatible_outputs
            }
            if len(possible_codes) != 1:
                raise ValueError("A compressed transition has no unique local code")
            local_expected = possible_codes.pop()
            text = render_interface_transition_prompt(
                tokenizer=tokenizer,
                case=local,
                prompt_config=prompt_config,
                condition=condition,
                input_code=symbols[predicted_code],
            )
        result = _score(
            model=model,
            tokenizer=tokenizer,
            text=text,
            candidates=symbols,
            expected=local_expected,
            name=f"block_{start // block_size}",
        )
        predicted_code = _prediction(result)
        result.update(
            supplied_code=(None if start == 0 else steps[-1]["prediction"]),
            compatible_input_states=list(compatible_inputs),
            local_expected_code=local_expected,
            global_expected_code=global_expected,
            locally_correct=predicted_code == local_expected,
            globally_correct=predicted_code == global_expected,
        )
        steps.append(result)

    true_code = interface_code_index(
        condition=condition,
        case=case,
        state=int(case["current_state"]),
        interface_config=interface_config,
        variant=variant,
    )
    answer_symbols = tuple(str(value) for value in range(8))
    predicted_final = None
    if predicted_code is not None:
        predicted_final = _score(
            model=model,
            tokenizer=tokenizer,
            text=render_interface_consumer_prompt(
                tokenizer=tokenizer,
                case=case,
                prompt_config=prompt_config,
                condition=condition,
                code=symbols[predicted_code],
            ),
            candidates=answer_symbols,
            expected=int(case["next_state"]),
            name="predicted_final",
        )
    gold_final = _score(
        model=model,
        tokenizer=tokenizer,
        text=render_interface_consumer_prompt(
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt_config,
            condition=condition,
            code=symbols[true_code],
        ),
        candidates=answer_symbols,
        expected=int(case["next_state"]),
        name="gold_final",
    )
    return {
        "schema_version": 1,
        "id": str(case["id"]),
        "condition": condition,
        "history_steps": int(case["history_steps"]),
        "block_size": block_size,
        "program_context": str(case["program_context"]),
        "path_code": int(case["path_code"]),
        "current_state": int(case["current_state"]),
        "true_code": true_code,
        "predicted_code": predicted_code,
        "state_correct": predicted_code == true_code,
        "steps": steps,
        "predicted_final": predicted_final,
        "gold_final": gold_final,
    }


def summarize_interface_rows(
    rows: list[dict[str, Any]], condition: str
) -> dict[str, Any]:
    """Summarize recursive code accuracy, closure, and utilization."""
    if not rows:
        raise ValueError("Cannot summarize empty interface evaluation")
    by_horizon = {}
    for horizon in sorted({int(row["history_steps"]) for row in rows}):
        selected = [row for row in rows if int(row["history_steps"]) == horizon]
        clusters = [str(row["program_context"]) for row in selected]
        steps = [step for row in selected for step in row["steps"]]
        same_groups: defaultdict[tuple[str, int], list[int | None]] = defaultdict(list)
        for row in selected:
            same_groups[(row["program_context"], int(row["current_state"]))].append(
                row["predicted_code"]
            )
        agreement = []
        for codes in same_groups.values():
            for index, left in enumerate(codes):
                agreement.extend(
                    left is not None and left == right for right in codes[index + 1 :]
                )
        by_horizon[str(horizon)] = {
            "case_count": len(selected),
            "state_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["state_correct"]) for row in selected], clusters, seed=9100 + horizon
            ),
            "predicted_answer_accuracy": cluster_bootstrap_mean_ci(
                [
                    bool(row["predicted_final"] and row["predicted_final"]["is_expected_unconstrained"])
                    for row in selected
                ],
                clusters,
                seed=9200 + horizon,
            ),
            "gold_code_answer_accuracy": cluster_bootstrap_mean_ci(
                [bool(row["gold_final"]["is_expected_unconstrained"]) for row in selected],
                clusters,
                seed=9300 + horizon,
            ),
            "local_closure_accuracy": bootstrap_mean_ci(
                [bool(step["locally_correct"]) for step in steps], seed=9400 + horizon
            ),
            "same_state_code_agreement": bootstrap_mean_ci(
                agreement, seed=9500 + horizon
            ),
        }
    valid = [int(row["predicted_code"]) for row in rows if row["predicted_code"] is not None]
    counts = Counter(valid)
    from .state_handoff_information import (
        conditional_entropy,
        conditional_mutual_information,
        discrete_entropy,
        mutual_information,
    )

    true_contract = [
        (int(row["current_state"]), int(row["true_code"]), str(row["program_context"]))
        for row in rows
    ]
    predicted_contract = [
        (int(row["current_state"]), int(row["predicted_code"]), str(row["program_context"]))
        for row in rows
        if row["predicted_code"] is not None
    ]

    def information(values: list[tuple[int, int, str]]) -> dict[str, Any]:
        state_code = [(state, code) for state, code, _ in values]
        return {
            "state_information_bits": mutual_information(state_code),
            "state_information_given_context_bits": conditional_mutual_information(values),
            "state_given_code_bits": conditional_entropy(state_code),
            "code_given_state_bits": conditional_entropy(
                (code, state) for state, code, _ in values
            ),
            "code_given_state_and_context_bits": conditional_entropy(
                (code, (state, context)) for state, code, context in values
            ),
            "code_entropy_bits": discrete_entropy(code for _, code, _ in values),
        }
    return {
        "schema_version": 1,
        "condition": condition,
        "case_count": len(rows),
        "program_context_count": len({row["program_context"] for row in rows}),
        "by_horizon": by_horizon,
        "code_counts": {str(key): value for key, value in sorted(counts.items())},
        "invalid_code_count": len(rows) - len(valid),
        "declared_codebook_size": CODEBOOK_SIZES[condition],
        "declared_capacity_bits": math.log2(CODEBOOK_SIZES[condition]),
        "true_code_information": information(true_contract),
        "predicted_code_information": information(predicted_contract),
    }


def evaluate_state_interface_condition(
    run_path: Path,
    condition: str,
    *,
    max_cases: int | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Resume recursive evaluation for one trained code condition."""
    if model is None or tokenizer is None:
        model, tokenizer = _load_evaluation_model(run_path, condition)
    experiment = load_config(run_path).get("state_handoff_training", {})
    cases = read_programs(run_path / TEST_PATH)
    completed = {row["id"] for row in read_interface_evaluation_cases(run_path, condition)}
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    output = interface_evaluation_dir(run_path, condition) / "cases.jsonl"
    for index, case in enumerate(pending, 1):
        append_jsonl(
            output,
            evaluate_interface_program_hf(
                model=model,
                tokenizer=tokenizer,
                case=case,
                prompt_config=experiment.get("prompt", {}),
                condition=condition,
                interface_config=experiment.get("interfaces", {}),
            ),
        )
        if on_progress is not None and (index == 1 or index == len(pending) or index % 10 == 0):
            on_progress(f"state interface {condition} {index}/{len(pending)} cases")
    rows = read_interface_evaluation_cases(run_path, condition)
    summary = summarize_interface_rows(rows, condition)
    summary.update(expected_case_count=len(cases), complete=len(rows) == len(cases))
    write_json(interface_evaluation_dir(run_path, condition) / "summary.json", summary)
    return summary


def compare_state_interface_conditions(run_path: Path) -> dict[str, Any]:
    """Compare equal-compute code contracts and apply the continuation gate."""
    config = load_config(run_path).get("state_handoff_training", {})
    conditions = tuple(str(value) for value in config.get("conditions", ()))
    summaries = {}
    for condition in conditions:
        path = interface_evaluation_dir(run_path, condition) / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing interface evaluation summary: {condition}")
        summary = json.loads(path.read_text())
        if not summary.get("complete"):
            raise RuntimeError(f"Interface evaluation is incomplete: {condition}")
        summaries[condition] = summary
    if "canonical_opaque" not in summaries:
        raise ValueError("Interface comparison requires canonical_opaque")
    accuracy = {
        condition: {
            horizon: values["predicted_answer_accuracy"]["mean"]
            for horizon, values in summary["by_horizon"].items()
        }
        for condition, summary in summaries.items()
    }
    gate_config = config.get("interface_gate", {})
    min_h8 = float(gate_config.get("min_canonical_h8", 0.90))
    min_h16 = float(gate_config.get("min_canonical_h16", 0.80))
    min_context_gap = float(gate_config.get("min_context_gap", 0.20))
    checks = {
        "canonical_h8": accuracy["canonical_opaque"].get("8", 0.0) >= min_h8,
        "canonical_h16": accuracy["canonical_opaque"].get("16", 0.0) >= min_h16,
        "canonical_gold_consumer": summaries["canonical_opaque"]["by_horizon"]["8"]
        ["gold_code_answer_accuracy"]["mean"]
        >= 0.95,
    }
    if "context_bound" in accuracy:
        checks["canonical_beats_context_bound"] = (
            accuracy["canonical_opaque"].get("8", 0.0)
            - accuracy["context_bound"].get("8", 0.0)
            >= min_context_gap
        )
    result = {
        "schema_version": 1,
        "conditions": list(conditions),
        "horizon_accuracy": accuracy,
        "summaries": summaries,
        "interface_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "thresholds": {
                "min_canonical_h8": min_h8,
                "min_canonical_h16": min_h16,
                "min_context_gap": min_context_gap,
            },
            "checks": checks,
        },
    }
    from .state_interface_interchange import analyze_interface_interchange

    result["interchange"] = {
        condition: analyze_interface_interchange(run_path, condition)
        for condition in conditions
    }
    from .state_interface_equivalence import analyze_predicted_code_equivalence

    result["predicted_equivalence"] = {
        condition: analyze_predicted_code_equivalence(run_path, condition)
        for condition in conditions
    }
    write_json(run_path / INTERFACE_EVALUATION_ROOT / "comparison_summary.json", result)
    _write_interface_plot(run_path, result)
    return result


def _write_interface_plot(run_path: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    horizons = [2, 4, 8, 16]
    figure, axis = plt.subplots(figsize=(7, 4))
    for condition, values in summary["horizon_accuracy"].items():
        axis.plot(
            horizons,
            [values.get(str(horizon), float("nan")) for horizon in horizons],
            marker="o",
            label=condition.replace("_", " "),
        )
    axis.axhline(0.125, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(horizons)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("History horizon")
    axis.set_ylabel("Recursive answer accuracy")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        run_path / INTERFACE_EVALUATION_ROOT / "interface_accuracy.png", dpi=160
    )
    plt.close(figure)
