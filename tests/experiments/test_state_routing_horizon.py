from __future__ import annotations

from src.experiments.state_routing_horizon import (
    build_cases,
    parse_answer,
    summarize,
    validate_cases,
    validate_token_contract,
)


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def test_cases_are_matched_and_mark_real_source_tokens() -> None:
    rows = build_cases(count=12, seed=7)

    manifest = validate_cases(rows, expected=12)

    assert all(manifest["checks"].values())
    for row in rows:
        assert row["clean_answer"] != row["corrupt_answer"]
        for site, (start, end) in row["clean"]["spans"].items():
            assert site in {"problem_source", "cot_write"}
            assert row["clean"]["text"][start:end].isdigit()
            assert row["corrupt"]["text"][start:end].isdigit()


def test_token_contract_maps_spans_and_answer_candidates() -> None:
    row = build_cases(count=1, seed=11)[0]

    contract = validate_token_contract(CharacterTokenizer(), row)

    assert contract["changed_token_count"] >= 2
    assert contract["positions"].keys() == {"problem_source", "cot_write"}
    assert all(len(value) == 1 for value in contract["positions"].values())
    assert len(set(contract["candidate_ids"])) == 10


def test_answer_parser_accepts_observed_model_forms() -> None:
    assert parse_answer("Final score is Answer=7.") == 7
    assert parse_answer(r"Therefore, Ada's final score is \(\boxed{8}\).") == 8
    assert parse_answer("The final score is 4.") == 4


def _result(index: int, cot: float, problem: float, correct: bool) -> dict:
    def cells(value: float) -> list[dict]:
        return [{"layer": 0, "normalized_indirect_effect": value}]

    return {
        "id": f"case_{index}",
        "difficulty": "easy" if index % 2 == 0 else "hard",
        "clean_answer": 7,
        "clean_prefers_clean": True,
        "corrupt_prefers_corrupt": True,
        "effects": {
            "problem_source": cells(problem),
            "cot_write": cells(cot),
            "read_site": cells(1.0),
        },
        "generation": {"parsed": True, "text": f"Answer={'7' if correct else '2'}"},
        "generation_correct": correct,
    }


def test_summary_identifies_cot_source_only_with_enough_cases() -> None:
    rows = [_result(index, 0.8, 0.2, index % 3 != 0) for index in range(160)]

    report = summarize(rows)

    assert report["decision"] == {
        "causal_source": "cot_write",
        "identified": True,
        "continue_large_run": True,
    }
    assert report["cot_minus_problem_effect"]["mean"] > 0.5
