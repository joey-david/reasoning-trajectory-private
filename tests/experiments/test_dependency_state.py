"""Exact interpreter checks and paired-control tests for dependency-state pilots."""

import ast

import pytest

from src.experiments.dependency_state import (
    load_cases,
    parse_result,
    program_facts,
    revision_cases,
    score_case,
    summarize,
)


def test_schedule_pairs_share_graph_operations_literals_and_answer():
    rows = load_cases(count=24, seed=71)
    assert rows == load_cases(count=24, seed=71)
    for low, high in zip(rows[::2], rows[1::2]):
        assert low["family_id"] == high["family_id"]
        assert low["graph"] == high["graph"]
        assert low["expected"] == high["expected"]
        assert sorted(low["code"].splitlines()) == sorted(high["code"].splitlines())
        assert low["operations"] == high["operations"]
        assert low["mean_live"] < high["mean_live"]
        if low["width"] > 2:
            assert low["peak_live"] < high["peak_live"]


def test_interpreter_reference_and_rejected_code():
    facts = program_facts("x = 12\ny = x * 3\nz = y - x\nresult = (x, z, y)")
    assert facts["values"]["result"] == (12, 24, 36)
    assert facts["dependencies"]["z"] == ["x", "y"]
    for code in (
        "import os",
        "x = print(3)",
        "x = y + 1",
        "x = 1\nx = 2",
        "x = True\nresult = (x,)",
        "x = 1.5\nresult = (x,)",
    ):
        with pytest.raises(ValueError):
            program_facts(code)


def test_revision_oracle_and_selective_deletion():
    rows = revision_cases(count=12, seed=72)
    for start in range(0, len(rows), 5):
        intact, affected, independent, root, restart = rows[start : start + 5]
        assert len({tuple(row["expected"]) for row in rows[start : start + 5]}) == 1
        assert (
            affected["removed_lines"] == independent["removed_lines"] == intact["depth"]
        )
        assert root["removed_lines"] == 1
        assert "a00 =" in affected["retained_work"]
        assert "a01 =" not in affected["retained_work"]
        assert "b01 =" in affected["retained_work"]
        assert "a01 =" in independent["retained_work"]
        assert "b01 =" not in independent["retained_work"]
        assert not restart["retained_work"]
        old = program_facts(intact["original_code"])["values"]
        new = program_facts(intact["code"])["values"]
        assert all(new[k] == v for k, v in old.items() if k.startswith("b"))
        if intact["changed"]:
            assert new["a00"] == old["a00"] + 1
            assert intact["expected"][1] != intact["stale_descendant_final"]
        else:
            assert new == old


def test_strict_final_answer_parser():
    assert parse_result("Calculation: 3 * 4 = 12\nAnswer: [12, -3, 0].") == [12, -3, 0]
    assert parse_result("Answer: (12,)") == [12]
    assert parse_result("So, the final answer is: [13, -11, 84]") == [13, -11, 84]
    assert parse_result("So, the final result is: `[14, -2, 84]`") == [14, -2, 84]
    for text in (
        "The result is 12",
        "Answer: [12.0]",
        "Answer: [True]",
        "Answer: [3 * 4]",
        "Answer: [12]\nActually, no",
        "Answer: [__import__('os')]",
    ):
        assert parse_result(text) is None


def test_stale_score_requires_acceptance_and_independent_preservation():
    case = next(r for r in revision_cases(count=1, seed=73) if r["changed"])
    prediction = list(case["expected"])
    prediction[1] = case["stale_descendant_final"]
    scores = score_case(case, f"Answer: {prediction}")
    assert scores["stale_with_correct_root"] and not scores["correct"]
    prediction[2] += 1
    assert not score_case(case, f"Answer: {prediction}")["stale_with_correct_root"]
    assert not score_case(case, "Answer: []")["parsed"]


def test_summary_pairs_and_missing_outputs():
    cases = load_cases(count=2, seed=74)
    generations = [
        {
            "sample_id": row["id"],
            "produced_text": f"Answer: {row['expected']}",
            "generated_token_ids": [1, 2],
        }
        for row in cases[:2]
    ]
    summary = summarize(cases, generations, max_new_tokens=2)
    assert summary["completed"] == 2 and len(summary["missing"]) == 2
    assert summary["paired"]["local-breadth:changed=na"]["mean_difference"] == 0
    assert all(r["capped"] for r in summary["measurements"])
    with pytest.raises(ValueError):
        summarize(cases, generations * 2, max_new_tokens=4)


def test_complete_programs_execute_in_cpython():
    # Independent execution of every rendered program catches rendering/oracle drift.
    for row in load_cases(count=12, seed=75) + revision_cases(count=6, seed=76):
        namespace = {}
        exec(
            compile(ast.parse(row["code"]), "<test>", "exec"),
            {"__builtins__": {}},
            namespace,
        )
        assert list(namespace["result"]) == row["expected"]


def test_observed_qwen_arithmetic_error_is_not_stale_state():
    # Job 1774974, changed-input intact arm: it recomputed a02=76 correctly,
    # then wrote -8 * 2 + 14 = -12. This is not the stale-descendant signature.
    case = next(
        r
        for r in revision_cases(count=1, seed=26090501)
        if r["changed"] and r["arm"] == "intact"
    )
    score = score_case(case, "final = -8 * 2 + 14 = -12\nAnswer: [14, -12, 84]")
    assert score["root_correct"] and score["independent_correct"]
    assert not score["downstream_correct"] and not score["stale_with_correct_root"]
