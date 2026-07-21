from __future__ import annotations

import json

import pytest

from src.experiments.depth_relief.abstraction import (
    build_state_abstraction_benchmark,
)
from src.experiments.depth_relief.explicit_handoff import (
    artifact_handoff_record,
    discrete_capacity_bits,
    evaluate_explicit_handoff_case_hf,
    merge_explicit_handoff_records,
    read_explicit_handoff_records,
    summarize_explicit_handoffs,
)
from src.experiments.depth_relief.factorization import (
    render_factorization_update_prompt,
)


def _case() -> dict:
    return build_state_abstraction_benchmark(
        {
            "bits": 3,
            "history_steps": [2],
            "groups_per_horizon": 3,
            "paths_per_state": 8,
            "formats": ["prose"],
            "seed": 19,
        }
    )[0]


def _condition(prediction: int | None, expected: int) -> dict:
    return {
        "prediction": prediction,
        "unconstrained_prediction": prediction,
        "unconstrained_token_id": prediction,
        "candidate_probability_mass": 1.0,
        "expected_next_state": expected,
        "is_expected": prediction == expected,
        "is_expected_unconstrained": prediction == expected,
    }


def _factorization_row(case: dict, synthesized: int | None = None) -> dict:
    synthesized = int(case["current_state"]) if synthesized is None else synthesized
    return {
        "id": case["id"],
        "conditions": {
            "synthesize": _condition(synthesized, int(case["current_state"])),
            "compose": _condition(0, int(case["next_state"])),
        },
    }


def test_arbitrary_state_update_prompt_uses_supplied_state() -> None:
    case = _case()
    prompt = render_factorization_update_prompt(
        tokenizer=None,
        case=case,
        config={"prompt": {"mode": "plain"}},
        state=7,
        rule={"kind": "add", "value": 3},
        name="arbitrary",
        label="Operation",
    )
    assert "Current state: 7." in prompt["text"]
    assert "Start state:" not in prompt["text"]
    assert prompt["expected_next_state"] == 2


def test_horizon_one_matched_histories_reach_each_state() -> None:
    cases = build_state_abstraction_benchmark(
        {
            "bits": 3,
            "history_steps": [1],
            "groups_per_horizon": 3,
            "paths_per_state": 8,
            "formats": ["prose"],
            "seed": 23,
        }
    )
    group = [case for case in cases if case["abstraction_group"] == "h1_g0"]
    assert len(group) == 64
    assert all(len(case["history"]) == 1 for case in group)
    assert all(case["state_path"][-1] == case["current_state"] for case in group)
    assert {
        case["history"][0]["value"]
        for case in group
        if case["current_state"] == 0
    } == {(-int(group[0]["initial_state"])) % 8 + 8 * path for path in range(8)}


def test_oracle_handoff_applies_final_to_recorded_synthesis() -> None:
    case = _case()
    predicted_state = (int(case["current_state"]) + 1) % 8
    record = artifact_handoff_record(
        case, _factorization_row(case, synthesized=predicted_state)
    )
    oracle = record["conditions"]["oracle_executor_handoff"]
    assert oracle["source_state_prediction"] == predicted_state
    assert oracle["prediction"] == case["final_rule"]["mapping"][predicted_state]
    assert oracle["is_expected_unconstrained"] == (
        oracle["prediction"] == case["next_state"]
    )


def test_all_handoff_calls_delete_original_history(monkeypatch: pytest.MonkeyPatch) -> None:
    case = _case()
    prompts: list[str] = []

    def score_prompt(*, prompt: dict, **_kwargs: object) -> dict:
        prompts.append(str(prompt["text"]))
        expected = int(prompt["expected_next_state"])
        return _condition(expected, expected)

    monkeypatch.setattr(
        "src.experiments.depth_relief.explicit_handoff._score_prompt",
        score_prompt,
    )
    record = evaluate_explicit_handoff_case_hf(
        model=None,
        tokenizer=None,
        case=case,
        factorization_row=_factorization_row(case),
        config={"prompt": {"mode": "plain"}},
    )
    assert record["prompt_contract"]["original_history_deleted"] is True
    assert len(prompts) == int(case["history_steps"]) + 3
    assert all("Start state:" not in prompt for prompt in prompts)
    assert all("Step 1:" not in prompt for prompt in prompts)
    assert record["conditions"]["stepwise_explicit"][
        "is_expected_unconstrained"
    ]


def test_discrete_capacity_is_exactly_three_bits() -> None:
    assert discrete_capacity_bits(slots=1, codebook_size=8) == 3
    with pytest.raises(ValueError, match="power-of-two"):
        discrete_capacity_bits(slots=1, codebook_size=7)


def test_duplicate_explicit_handoff_rows_are_rejected(tmp_path) -> None:
    path = tmp_path / "depth_relief/explicit_handoff/cases.jsonl"
    path.parent.mkdir(parents=True)
    row = {
        "id": "case",
        "phase": "artifact",
        "history_steps": 2,
        "program_context": "h2_g0",
        "program_context_split": "test",
        "conditions": {},
    }
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate explicit-handoff"):
        read_explicit_handoff_records(tmp_path)


def test_saved_artifact_summary_is_deterministic() -> None:
    case = _case()
    rows = merge_explicit_handoff_records(
        [artifact_handoff_record(case, _factorization_row(case))]
    )
    first = summarize_explicit_handoffs(rows, {})
    second = summarize_explicit_handoffs(rows, {})
    assert first == second
    assert first["phase1_gate"]["status"] == "pending_inference"
