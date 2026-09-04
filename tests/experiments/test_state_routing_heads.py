from __future__ import annotations

import numpy as np

from src.experiments.state_routing_heads import (
    build_head_cases,
    parse_generated_writes,
    select_heads,
    summarize_measurements,
    validate_head_cases,
)


def test_generated_write_parser_keeps_result_not_modulus_or_comment() -> None:
    text = (
        "1. Bela: 5 - 7 = 8 (Bela after subtracting 7)\n"
        "2. Ada: \\(8 + 3 = 11 \\equiv 1 \\pmod{10}\\)\n"
        "3. Bela: 8 mod 10 + 3 = 11 \\equiv 1 \\pmod{10}\n"
    )
    rows = parse_generated_writes(text, ["Bela", "Ada", "Bela"])
    assert [row["value"] for row in rows] == [8, 1, 1]
    assert [text[s:e] for s, e in (row["char_span"] for row in rows)] == [
        "8",
        "1",
        "1",
    ]


def test_generated_write_parser_rejects_repeated_digit_as_literal() -> None:
    [row] = parse_generated_writes("1. Ada: 0 + 0 = 00\n", ["Ada"])
    assert row["value"] == 0
    assert not row["literal_digit"]


def test_head_cases_lock_short_fit_and_long_test() -> None:
    rows = build_head_cases(screen_count=4, development_count=6, test_count=6, seed=7)
    report = validate_head_cases(rows, {"screen": 4, "development": 6, "test": 6})
    assert report["checks"]["long_test"]
    assert all(row["clean_answer"] != row["corrupt_answer"] for row in rows)


def test_head_selection_has_layer_matched_controls() -> None:
    rows = []
    for index in range(3):
        scores = np.zeros((2, 4))
        scores[0, 1] = 4 + index
        scores[1, 2] = 3 + index
        rows.append(
            {
                "scores": scores.tolist(),
                "eligible": True,
                "clean_logit_difference": 1.0,
                "original_logit_difference": -1.0,
            }
        )
    result = select_heads(rows, count=2, seed=3)
    assert [(row["layer"], row["head"]) for row in result["selected"]] == [
        (0, 1),
        (1, 2),
    ]
    assert [row["layer"] for row in result["selected"]] == [
        row["layer"] for row in result["layer_matched_controls"]
    ]


def test_summary_fits_short_and_scores_long() -> None:
    rows = []
    for split, count in (("development", 20), ("test", 20)):
        for index in range(count):
            correct = index % 2 == 0
            rows.append(
                {
                    "split": split,
                    "event_count": 4 if split == "development" else 12,
                    "generation": {"parsed": True},
                    "generation_correct": correct,
                    "answer_logit_margin": 1.0 if correct else -1.0,
                    "routing": {
                        "routing_margin": 2.0 if correct else -2.0,
                        "valid_support_fraction": 0.2 if correct else 0.01,
                        "attention_entropy": 1.0,
                    },
                    "control_routing": {
                        "routing_margin": 0.0,
                        "valid_support_fraction": 0.1,
                        "attention_entropy": 1.0,
                    },
                }
            )
    result = summarize_measurements(rows)
    assert result["routing_predictor"]["roc_auc"] == 1.0
    assert result["accuracy_by_split"] == {"development": 0.5, "test": 0.5}
