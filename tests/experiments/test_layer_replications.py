from __future__ import annotations

import random
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, Qwen2Config, Qwen2ForCausalLM

from src.experiments.layer_replications.robustness import evaluate_chunk
from src.experiments.layer_replications import rl_analysis
from src.experiments.layer_replications.single_layer_rl import (
    _load_evaluation_rows,
    build_grpo_config,
    configure_trainable_layers,
)
from src.experiments.layer_replications.symbolic import (
    _build_pair,
    causal_mediation,
)
from src.experiments.layer_replications.symbolic_analysis import fwer_threshold


class ExactTokenizer:
    """Minimal tokenizer for deterministic identity-rule protocol tests."""

    def __init__(self) -> None:
        self.words = {word: index + 10 for index, word in enumerate("abcdefghijkl")}
        self.separators = {"^": 1, "\n": 2}

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, object]:
        del add_special_tokens
        ids = []
        current = ""
        for character in text:
            if character in self.separators:
                if current:
                    ids.append(self.words[current])
                    current = ""
                ids.append(self.separators[character])
            else:
                current += character
        if current:
            ids.append(self.words[current])
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class FixedPromptTokenizer:
    """Map two symbolic prompts to matched fixed token sequences."""

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        ids = [3, 5, 7, 9] if text == "donor" else [4, 6, 8, 10]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class LayerReplicationTests(unittest.TestCase):
    def test_symbolic_contexts_have_the_paper_causal_targets(self):
        tokenizer = ExactTokenizer()
        words = list(tokenizer.words)
        abstract = _build_pair(
            tokenizer,
            words,
            shots=2,
            context_type="abstract",
            base_rule="ABA",
            rng=random.Random(2),
        )
        token = _build_pair(
            tokenizer,
            words,
            shots=2,
            context_type="token",
            base_rule="ABA",
            rng=random.Random(2),
        )

        self.assertIsNotNone(abstract)
        self.assertIsNotNone(token)
        assert abstract is not None and token is not None
        self.assertNotEqual(abstract["causal_answer"], abstract["donor_answer"])
        self.assertEqual(token["causal_answer"], token["donor_answer"])
        self.assertNotEqual(token["causal_answer"], token["target_answer"])

    def test_robustness_metrics_cover_drop_and_swap(self):
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=16,
                n_embd=16,
                n_layer=2,
                n_head=2,
            )
        )
        blocks = [{"id": "tiny", "input_ids": [1, 2, 3, 4], "scored_tokens": 3}]

        dropped = evaluate_chunk(
            model,
            blocks,
            kind="drop",
            layer=0,
            chunk=0,
            blocks_per_task=1,
        )
        swapped = evaluate_chunk(
            model,
            blocks,
            kind="swap",
            layer=0,
            chunk=0,
            blocks_per_task=1,
        )

        self.assertEqual(dropped["token_count"], 3)
        self.assertEqual(swapped["token_count"], 3)
        self.assertGreaterEqual(dropped["kl_sum"], 0.0)
        self.assertGreaterEqual(swapped["kl_sum"], 0.0)

    def test_qwen_head_patching_returns_every_layer_and_head(self):
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=16,
            )
        )
        row = {
            "id": "tiny",
            "context_type": "token",
            "base_rule": "ABA",
            "donor_prompt": "donor",
            "target_prompt": "target",
            "target_answer_id": 11,
            "causal_answer_id": 12,
            "patch_positions": [3],
            "token_count": 4,
        }

        result = causal_mediation(
            model,
            FixedPromptTokenizer(),
            row,
            mechanism="retrieval",
            head_batch_size=2,
        )

        self.assertEqual(len(result["scores"]), 2)
        self.assertTrue(all(len(layer) == 4 for layer in result["scores"]))

    def test_rl_freezes_everything_except_requested_decoder_scope(self):
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
            )
        )

        single = configure_trainable_layers(model, layer=1)
        self.assertEqual(single["selected_layers"], [1])
        self.assertTrue(single["embeddings_frozen"])
        self.assertTrue(single["lm_head_frozen"])
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.model.layers[0].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.model.layers[1].parameters()
            )
        )

        full = configure_trainable_layers(model, full=True)
        self.assertEqual(full["selected_layers"], [0, 1])
        self.assertTrue(full["embeddings_frozen"])
        self.assertTrue(full["lm_head_frozen"])

    def test_rl_batch_translation_matches_the_paper(self):
        args = build_grpo_config(
            {
                "revision": "pinned",
                "attn_implementation": "sdpa",
            },
            {
                "learning_rate": 5e-6,
                "train_batch_size": 512,
                "ppo_mini_batch_size": 128,
                "micro_batch_size": 8,
                "group_size": 4,
                "max_response_length": 3072,
                "kl_coefficient": 0.001,
                "clip_range": 0.2,
                "epochs": 4,
                "seed": 7,
                "use_transformers_continuous_batching": True,
            },
            Path("/tmp/layer-rl-test"),
            world_size=1,
        )

        self.assertEqual(args.generation_batch_size, 2048)
        self.assertEqual(args.per_device_train_batch_size, 8)
        self.assertEqual(args.gradient_accumulation_steps, 16)
        self.assertEqual(args.num_generations, 4)
        self.assertEqual(args.loss_type, "grpo")

    def test_rl_extracts_gsm8k_final_answer_before_scoring(self):
        with patch(
            "datasets.load_dataset",
            return_value=[{"question": "How many?", "answer": "work\n#### 18"}],
        ):
            rows = _load_evaluation_rows(
                {
                    "key": "gsm8k",
                    "path": "openai/gsm8k",
                    "problem_field": "question",
                    "answer_field": "answer",
                    "answer_regex": r"####\s*(?P<answer>.+?)\s*$",
                }
            )

        self.assertEqual(rows, [{"problem": "How many?", "solutions": ["18"]}])

    def test_symbolic_permutation_threshold_is_seed_deterministic(self):
        matrices = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4) / 10

        first = fwer_threshold(matrices, trials=100, seed=7)
        second = fwer_threshold(matrices, trials=100, seed=7)

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_rl_analysis_accepts_the_published_anchor_subset(self):
        with TemporaryDirectory() as temporary:
            run = Path(temporary)
            output = run / "layer_replications/zhang_single_layer_rl"
            evaluations = output / "evaluations"
            evaluations.mkdir(parents=True)
            for setting, score in {
                "base": 0.4,
                "full": 0.5,
                "layer-01": 0.45,
                "layer-10": 0.52,
            }.items():
                (evaluations / f"{setting}.json").write_text(
                    json.dumps({"math_average": score}), encoding="utf-8"
                )

            config = {
                "model": {"layer_count": 28},
                "single_layer_rl": {"core_scan_layers": [1, 10]},
            }
            with (
                patch.object(rl_analysis, "load_config", return_value=config),
                patch.object(rl_analysis, "_plot_curve"),
            ):
                report = rl_analysis.analyze(run)

            self.assertFalse(report["scan_complete"])
            self.assertEqual(report["scanned_layers"], [1, 10])
            self.assertEqual(report["best_layer"], 10)
            self.assertAlmostEqual(report["best_contribution"], 1.2)


if __name__ == "__main__":
    unittest.main()
