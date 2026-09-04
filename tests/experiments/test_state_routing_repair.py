from __future__ import annotations

from src.experiments.state_routing_heads import build_head_cases, parse_generated_writes
from src.experiments.state_routing_repair import (
    intermediate_read_sites,
    steering_eligible,
    summarize_intermediate_steering,
    summarize_steering,
)


def test_steering_requires_two_literal_target_writes_before_answer() -> None:
    row = {"target": "Ada"}
    measurement = {"generation": {"text": "1. Ada = 2\nAnswer=3\n2. Ada = 4"}}
    free = {
        "eligible_routing": True,
        "parsed_writes": [
            {"name": "Ada", "literal_digit": True, "char_span": [9, 10]},
            {"name": "Ada", "literal_digit": True, "char_span": [27, 28]},
        ],
    }
    assert not steering_eligible(row, measurement, free)
    free["parsed_writes"][1]["char_span"] = [8, 9]
    assert steering_eligible(row, measurement, free)


def _conditions(answer: int, baseline: int, valid: dict[float, int]) -> dict:
    conditions = {
        "baseline": {"prediction": baseline, "correct_margin": 0.0},
    }
    for alpha in (0.1, 0.25):
        for name, prediction in (
            ("valid", valid[alpha]),
            ("stale", baseline),
            ("control", baseline),
        ):
            conditions[f"{name}_{alpha:g}"] = {
                "prediction": prediction,
                "correct_margin": float(prediction == answer),
            }
    return conditions


def test_summary_selects_safe_development_strength_then_scores_test() -> None:
    rows = []
    for split in ("development", "test"):
        rows.extend(
            [
                {
                    "split": split,
                    "target_was_correct": False,
                    "saved_prediction": 2,
                    "answer": 1,
                    "conditions": _conditions(1, 2, {0.1: 1, 0.25: 1}),
                },
                {
                    "split": split,
                    "target_was_correct": True,
                    "saved_prediction": 1,
                    "answer": 1,
                    "conditions": _conditions(1, 1, {0.1: 1, 0.25: 2}),
                },
            ]
        )
    result = summarize_steering(rows, [0.1, 0.25])
    assert result["selected_alpha"] == 0.1
    assert result["test"]["valid_repair"] == 1.0
    assert result["test"]["preservation"] == 1.0
    assert all(result["gate"].values())


def test_intermediate_sites_parse_canonical_model_trace() -> None:
    sites = []
    for seed in range(10):
        [row] = build_head_cases(
            screen_count=1, development_count=0, test_count=0, seed=seed
        )
        generated = row["clean"]["text"].split("Solution:\n", 1)[1]
        names = [write["name"] for write in row["clean"]["writes"]]
        sites.extend(
            intermediate_read_sites(
                row,
                {"generation": {"text": generated}},
                {"parsed_writes": parse_generated_writes(generated, names)},
            )
        )
    assert sites
    assert all(site["saved_operand"] == site["answer"] for site in sites)


def test_intermediate_summary_requires_read_and_next_write() -> None:
    def condition(operand: int, result: int) -> dict:
        return {
            "operand": {"prediction": operand},
            "result": {"prediction": result},
        }

    rows = [
        {
            "split": "test",
            "saved_operand": 2,
            "answer": 1,
            "result_answer": 4,
            "conditions": {
                "baseline": condition(2, 5),
                "valid": condition(1, 4),
                "wrong_source": condition(3, 4),
                "control_heads": condition(2, 4),
            },
        },
        {
            "split": "test",
            "saved_operand": 1,
            "answer": 1,
            "result_answer": 4,
            "conditions": {
                "baseline": condition(1, 4),
                "valid": condition(1, 4),
                "wrong_source": condition(3, 4),
                "control_heads": condition(1, 4),
            },
        },
    ]
    result = summarize_intermediate_steering(rows)
    assert result["failed_read_repair"]["valid"] == {
        "read": 1.0,
        "next_write": 1.0,
    }
    assert result["failed_read_repair"]["wrong_source"]["next_write"] == 0.0
    assert result["correct_read_preservation"] == {
        "read": 1.0,
        "next_write": 1.0,
    }
