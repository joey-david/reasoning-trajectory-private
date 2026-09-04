from __future__ import annotations

from src.experiments.state_routing_repair import select_donors, summarize_repairs


def test_donors_match_variable_and_differ_in_answer() -> None:
    dataset = {
        "target": {"id": "target", "split": "test", "target": "Ada", "clean_answer": 1},
        "good": {
            "id": "good",
            "split": "development",
            "target": "Ada",
            "clean_answer": 2,
        },
        "bad": {
            "id": "bad",
            "split": "development",
            "target": "Ada",
            "clean_answer": 3,
        },
    }
    free = [
        {"id": "good", "eligible_routing": True, "generation_correct": True},
        {"id": "bad", "eligible_routing": True, "generation_correct": False},
    ]
    assert select_donors({"id": "target"}, free, dataset) == {
        "successful": "good",
        "failed": "bad",
    }


def test_repair_summary_uses_aligned_failed_reads() -> None:
    conditions = {
        "baseline": {"prediction": 2, "correct_margin": -2.0},
        "successful_q": {"prediction": 2, "correct_margin": -1.0},
        "successful_k": {"prediction": 1, "correct_margin": 1.0},
        "successful_qk": {"prediction": 1, "correct_margin": 2.0},
        "control_heads_qk": {"prediction": 2, "correct_margin": -1.5},
        "failed_donor_qk": {"prediction": 2, "correct_margin": -1.0},
    }
    rows = [
        {
            "saved_prediction": 2,
            "answer": 1,
            "target_was_correct": False,
            "conditions": conditions,
        }
        for _ in range(5)
    ]
    result = summarize_repairs(rows)
    assert result["failed_reads"] == 5
    assert result["repair_accuracy"]["successful_qk"] == 1.0
    assert result["repair_accuracy"]["control_heads_qk"] == 0.0
