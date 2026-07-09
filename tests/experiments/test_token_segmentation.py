"""Tests for token-boundary scoring and window labeling."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from src.experiments.solution_object_extraction.labeling import (
    EXAMPLE_INPUT,
    EXAMPLE_OUTPUT,
    label_messages,
    locate_unique_quote,
    parse_json_object,
    resolve_window_label,
    window_chars,
)
from src.experiments.token_segmentation.evaluation import (
    boundary_f1,
    normalized_utility,
)
from src.experiments.token_segmentation.semantic_evaluation import (
    sentence_relation_summary,
)
from src.experiments.token_segmentation.semantic_labels import (
    SemanticSpan,
    SemanticTrace,
    load_semantic_traces,
)
from src.experiments.token_segmentation.signals import (
    local_mean_shift,
    top_boundaries,
)
from src.experiments.token_segmentation.text_boundary import (
    paired_auc_difference,
    transition_contexts,
)
from src.models.vllm_backend import generate_vllm


class TokenSegmentationTests(unittest.TestCase):
    def test_normalized_utility_anchors_random_expectation_and_oracle(self) -> None:
        scores = np.asarray([0.0, 1.0, 4.0, 0.0])
        oracle = top_boundaries(scores, 1, min_segment_tokens=1)

        self.assertEqual(oracle.tolist(), [2])
        self.assertAlmostEqual(normalized_utility(scores, oracle, oracle), 1.0)

    def test_local_mean_shift_finds_piecewise_constant_transition(self) -> None:
        values = np.asarray([[0.0], [0.0], [4.0], [4.0]])

        self.assertEqual(int(np.argmax(local_mean_shift(values, 2))), 1)

    def test_constrained_selection_is_globally_optimal(self) -> None:
        scores = np.asarray([0.0, 6.0, 10.0, 6.0, 0.0, 0.0, 0.0])

        selected = top_boundaries(scores, 2, min_segment_tokens=2)

        self.assertEqual(selected.tolist(), [1, 3])

    def test_boundary_f1_matches_each_target_once(self) -> None:
        score = boundary_f1(
            np.asarray([2, 3]),
            np.asarray([3]),
            tolerance=1,
        )

        self.assertAlmostEqual(score, 2.0 / 3.0)

    def test_window_chars_ignores_empty_token_pieces(self) -> None:
        spans = [(0, 2), None, (2, 5), (5, 8)]

        self.assertEqual(window_chars(spans, 1, 4), (2, 8))

    def test_window_label_requires_exact_span_text(self) -> None:
        row = {"text": "Compute 2 + 3 = 5."}
        label = {
            "spans": [
                {
                    "text": "wrong",
                    "label": "derive_value",
                    "confidence": 0.9,
                }
            ]
        }

        _, errors = resolve_window_label(row, label)
        self.assertIn(
            "span 0: quoted text does not occur in the excerpt; copy it verbatim",
            errors,
        )

    def test_quote_must_discriminate_one_interval(self) -> None:
        match, error = locate_unique_quote("check x, then check x again", "check x")

        self.assertIsNone(match)
        self.assertIn("occurs 2 times", error)

    def test_in_context_example_resolves_without_offsets(self) -> None:
        text = EXAMPLE_INPUT["excerpt"]
        resolved, errors = resolve_window_label(
            {
                "text": text,
                "token_start": 10,
                "token_char_spans": [[index, index + 1] for index in range(len(text))],
            },
            EXAMPLE_OUTPUT,
        )

        self.assertEqual(errors, [])
        self.assertEqual(resolved["spans"][1]["start_char"], 65)
        self.assertEqual(resolved["spans"][1]["end_char"], 112)
        self.assertEqual(resolved["spans"][1]["token_start"], 75)
        self.assertEqual(resolved["spans"][1]["token_end"], 121)

    def test_local_label_parser_accepts_fenced_json(self) -> None:
        label = parse_json_object('```json\n{"spans":[]}\n```')

        self.assertEqual(label, {"spans": []})

    def test_retry_prompt_contains_the_alignment_failure(self) -> None:
        messages = label_messages(
            {"question": "Q", "text": "excerpt", "bronze_edits": []},
            previous_output='{"spans":[]}',
            feedback="quoted text occurs twice",
        )

        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertIn("quoted text occurs twice", messages[-1]["content"])

    def test_vllm_adapter_returns_text_and_token_count(self) -> None:
        fake_vllm = ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kwargs: kwargs
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *_args, **_kwargs: "rendered prompt"
        )
        engine = SimpleNamespace(
            generate=lambda *_args, **_kwargs: [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text='{"spans":[]}', token_ids=[1, 2])]
                )
            ]
        )

        with patch.dict(sys.modules, {"vllm": fake_vllm}):
            text, count = generate_vllm(
                engine,
                tokenizer,
                [{"role": "user", "content": "label"}],
                max_tokens=20,
            )

        self.assertEqual(text, '{"spans":[]}')
        self.assertEqual(count, 2)

    def test_semantic_reconciliation_filters_edges_and_votes_duplicates(self) -> None:
        windows = [
            {
                "record_id": "q::1::0-14",
                "sample_id": "q",
                "seed": 1,
                "token_start": 0,
                "token_end": 14,
            },
            {
                "record_id": "q::1::6-20",
                "sample_id": "q",
                "seed": 1,
                "token_start": 6,
                "token_end": 20,
            },
        ]
        labels = [
            {
                "record_id": "q::1::0-14",
                "accepted": True,
                "silver_label": {
                    "spans": [
                        {
                            "token_start": 9,
                            "token_end": 10,
                            "label": "derive_value",
                            "confidence": 0.9,
                            "text": "derive",
                        },
                        {
                            "token_start": 12,
                            "token_end": 13,
                            "label": "plan",
                            "confidence": 0.9,
                            "text": "edge",
                        },
                    ]
                },
            },
            {
                "record_id": "q::1::6-20",
                "accepted": True,
                "silver_label": {
                    "spans": [
                        {
                            "token_start": 9,
                            "token_end": 10,
                            "label": "verify",
                            "confidence": 0.8,
                            "text": "derive",
                        },
                        {
                            "token_start": 17,
                            "token_end": 19,
                            "label": "extract_answer",
                            "confidence": 0.9,
                            "text": "answer",
                        },
                    ]
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, rows in (("windows.jsonl", windows), ("labels.jsonl", labels)):
                (root / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            traces, audit = load_semantic_traces(
                root / "windows.jsonl", root / "labels.jsonl"
            )

        spans = traces[("q", 1)].spans
        self.assertEqual([span.label for span in spans], ["derive_value", "extract_answer"])
        self.assertEqual(audit["edge_spans_dropped"], 1)
        self.assertEqual(audit["duplicate_interval_groups"], 1)
        self.assertEqual(audit["duplicate_interval_label_agreement"], 0.0)

    def test_sentence_relation_counts_non_sentence_boundaries(self) -> None:
        spans = [
            SemanticSpan("q", 1, 0, 9, "derive_value", 0.9, "text", "record")
        ]
        semantic = {
            ("q", 1): SemanticTrace(
                "q", 1, 15, spans, np.asarray([4, 9], dtype=np.int32)
            )
        }
        traces = {
            ("q", 1): SimpleNamespace(
                train=False,
                sentence_boundaries=np.asarray([5, 12], dtype=np.int32),
            )
        }

        summary = sentence_relation_summary(traces, semantic, tolerance=1)

        self.assertEqual(summary["held_out"]["semantic_boundary_sentence_aligned"], 0.5)
        self.assertEqual(summary["held_out"]["spans_crossing_sentence_boundaries"], 1.0)

    def test_text_boundary_context_marks_the_exact_transition(self) -> None:
        trace = SimpleNamespace(
            text="abc def",
            token_count=2,
            token_char_ends=np.asarray([3, 7]),
        )

        contexts, indices = transition_contexts(trace, context_chars=3)

        self.assertEqual(indices.tolist(), [0])
        self.assertEqual(contexts, ["abc <<<CUT>>>  de"])

    def test_paired_boundary_auc_uses_shared_questions(self) -> None:
        result = paired_auc_difference(
            {"a": 0.9, "b": 0.8, "latent_only": 1.0},
            {"a": 0.7, "b": 0.6, "text_only": 0.0},
        )

        self.assertAlmostEqual(result["latent_minus_text"], 0.2)
        self.assertEqual(result["questions"], 2)


if __name__ == "__main__":
    unittest.main()
