"""Check exact arithmetic, matched controls, and real transformer interventions."""

import re

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

from src.experiments.computation_interference import (
    block_edges,
    build_cases,
    freeze_heads,
    score_continuation,
    summarize,
)
from src.experiments.symbolic import safe_arithmetic_eval


def test_histories_are_true_and_value_matched():
    rows = build_cases(screen_count=8, test_count=16, seed=26090611)
    assert rows == build_cases(screen_count=8, test_count=16, seed=26090611)
    expressions = set()
    for row in rows:
        prefix = row["arms"]["repeat"]["prefix"]
        current = prefix.splitlines()[-1].removesuffix(" = ")
        assert current not in expressions
        expressions.add(current)
        assert safe_arithmetic_eval(current) == row["answer"]
        for arm, content in row["arms"].items():
            equations = re.findall(r"(?m)^(\d+ [+-] \d+) = (\d+)$", content["prefix"])
            assert len(equations) == (9 if row["split"] == "screen" else 25)
            assert all(safe_arithmetic_eval(lhs) == int(rhs) for lhs, rhs in equations)
            start, end = content["source_span"]
            assert int(content["prefix"][start:end]) == (
                row["answer"] if arm == "repeat" else row["lure"]
            )
            assert len(content["prefix"][slice(*content["control_span"])]) == 3


@pytest.mark.parametrize("kind", ["llama", "qwen"])
def test_edge_mask_is_head_batch_and_query_specific_and_restored(kind):
    torch.manual_seed(4)
    config_cls, model_cls = (
        (LlamaConfig, LlamaForCausalLM)
        if kind == "llama"
        else (Qwen2Config, Qwen2ForCausalLM)
    )
    model = model_cls(
        config_cls(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            attn_implementation="eager",
        )
    ).eval()
    ids = torch.tensor([[1, 7, 12, 2, 8], [1, 7, 12, 2, 8]])
    with torch.inference_mode():
        base = model(input_ids=ids, use_cache=False, output_attentions=True)
        with block_edges(model, [{"layer": 1, "head": 2, "batch_index": 0}], [1, 2], 3):
            changed = model(input_ids=ids, use_cache=False, output_attentions=True)
        restored = model(input_ids=ids, use_cache=False, output_attentions=True)
    torch.testing.assert_close(base.logits, restored.logits, rtol=0, atol=0)
    torch.testing.assert_close(base.logits[1], changed.logits[1], rtol=0, atol=0)
    torch.testing.assert_close(
        base.logits[:, :3], changed.logits[:, :3], rtol=0, atol=0
    )
    attn = changed.attentions[1]
    assert torch.count_nonzero(attn[0, 2, 3:, 1:3]) == 0
    torch.testing.assert_close(attn.sum(-1), torch.ones_like(attn.sum(-1)))
    assert torch.count_nonzero(torch.triu(attn, diagonal=1)) == 0
    for head in (0, 1, 3):
        torch.testing.assert_close(
            attn[0, head], base.attentions[1][0, head], rtol=0, atol=0
        )
    assert not torch.equal(base.logits[0, -1], changed.logits[0, -1])
    with pytest.raises(RuntimeError, match="intentional"):
        with block_edges(model, [{"layer": 1, "head": 2}], [1], 3):
            raise RuntimeError("intentional")
    assert not model.model.layers[1].self_attn._forward_pre_hooks


def test_score_uses_whole_numbers_and_the_final_answer():
    row = {"answer": 46, "lure": 66, "final_answer": 99}
    assert score_continuation("46.\nTwice this is 92; add 7.\nAnswer: 99", row)[
        "final_correct"
    ]
    assert score_continuation("66\nAnswer: 139", row)["lure_followed"]
    assert not score_continuation("46\nAnswer: 139", row)["final_correct"]
    assert score_continuation("460\nAnswer: 99", row)["subtotal"] == 460
    assert score_continuation("46.5\nAnswer: 99", row)["subtotal"] is None
    assert score_continuation("46\nAnswer: 99\nActually unsure", row)["answer"] is None


def test_real_llama_failure_is_not_a_completed_numeric_subtotal():
    # dependency_revision_pilot, revision-26090502-0004-1:restart, after "12 * 2 + 22 =".
    continuation = " 44 + 22 = 66`\n13. `result = (a00, final, b04) = (22, 66, 285)`\n\nSo, the final result is: `[22, 66, 285]`"
    scores = score_continuation(
        continuation, {"answer": 46, "lure": 66, "final_answer": 99}
    )
    assert (
        scores["subtotal"] is None
    )  # Do not mistake the first operand for the result.
    assert scores["answer"] is None  # A list is not this experiment's scalar answer.


def test_selection_does_not_filter_correct_screen_cases():
    heads = freeze_heads(
        [
            {"original_logit_difference": 2, "scores": [[0, 3, 0, 0]]},
            {"original_logit_difference": -2, "scores": [[0, 1, 0, 0]]},
        ],
        count=1,
        seed=1,
    )
    assert heads["eligible_screen_cases"] == 2
    assert heads["selected"][0] == {"layer": 0, "head": 1, "mean_effect": 2.0}
    assert heads["layer_matched_controls"][0]["head"] != 1


def test_summary_counts_damage_and_rejects_duplicates():
    rows = []
    for i, base, repaired in ((0, False, True), (1, True, False), (2, True, True)):
        for condition, correct in (("baseline", base), ("source", repaired)):
            rows.append(
                {
                    "id": str(i),
                    "arm": "operator_lure",
                    "condition": condition,
                    "subtotal_correct": correct,
                    "final_correct": correct,
                    "lure_followed": not correct,
                    "subtotal_parsed": True,
                    "final_parsed": True,
                    "capped": False,
                }
            )
    paired = summarize(rows)["paired"]["operator_lure:source-baseline"]
    assert (paired["wins"], paired["losses"], paired["accuracy_difference"]) == (
        1,
        1,
        0,
    )
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1])
