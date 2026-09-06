from src.experiments.natural_state_routing import (
    build_natural_cases,
    summarize_natural_measurements,
)


def test_build_natural_cases_keeps_one_verified_unambiguous_reuse() -> None:
    text = "Work:\n2 + 3 = 5\n4 + 4 = 8\n5 * 2 = 10\nAnswer: 10"
    rows = build_natural_cases(
        [
            {
                "sample_id": "case",
                "produced_text": text,
                "generated_token_ids": list(range(30)),
            }
        ],
        [
            {
                "id": "case",
                "question": "Start with 2, 3, and 4.",
                "gold_answer": "#### 10",
            }
        ],
        produced_answer_regex=r"Answer:\s*(\d+)",
        gold_answer_regex=r"####\s*(\d+)",
    )

    assert len(rows) == 1
    assert rows[0]["final_correct"] is True
    assert rows[0]["source_text"] == "5"
    assert rows[0]["wrong_source_text"] == "8"
    assert text[slice(*rows[0]["target_span"])] == "5"


def test_build_natural_cases_rejects_unverified_source() -> None:
    rows = build_natural_cases(
        [
            {
                "sample_id": "case",
                "produced_text": "2 + 3 = 8\n4 + 6 = 10\n8 * 2 = 16\nAnswer: 16",
                "generated_token_ids": list(range(30)),
            }
        ],
        [{"id": "case", "question": "Use 2, 3, 4, 6.", "gold_answer": "#### 16"}],
        produced_answer_regex=r"Answer:\s*(\d+)",
        gold_answer_regex=r"####\s*(\d+)",
    )

    assert rows == []


def test_summarize_natural_measurements_uses_aligned_rows_only() -> None:
    def row(baseline: int, wrong: int, control: int) -> dict:
        def condition(prediction: int, margin: float) -> dict:
            return {
                "prediction_id": prediction,
                "target_id": 1,
                "wrong_id": 2,
                "target_minus_wrong": margin,
            }

        return {
            "final_correct": True,
            "source_equation_index": 0,
            "target_equation_index": 2,
            "tokens": [
                {
                    "conditions": {
                        "baseline": condition(baseline, 4.0),
                        "valid_source": condition(1, 5.0),
                        "wrong_source": condition(wrong, -2.0),
                        "matched_heads": condition(control, 3.0),
                    }
                }
            ],
        }

    summary = summarize_natural_measurements([row(1, 2, 1), row(9, 2, 2)])

    assert summary["baseline_aligned"] == 1
    assert summary["wrong_source_exact"] == 1.0
    assert summary["matched_head_wrong_source_exact"] == 0.0
    assert summary["mean_margin_shift_to_wrong_source"] == 6.0
    assert summary["gate"]["exact_advantage_at_least_010"] is True
