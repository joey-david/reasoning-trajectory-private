from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.experiments.causal_reasoning.datasets import (
    EXPERIMENTS,
    build_experiment_cases,
    validate_experiment_cases,
)
from src.experiments.causal_reasoning.evaluation import _patch_layers
from src.experiments.causal_reasoning.reporting import (
    _feature_probes,
    reduce_experiment,
)
from src.orchestration.jobs.causal_reasoning import pending_tasks
from src.runtime.data import write_jsonl


class CausalReasoningDatasetTests(unittest.TestCase):
    def test_all_datasets_are_deterministic_and_group_disjoint(self):
        for offset, experiment in enumerate(EXPERIMENTS):
            rows = build_experiment_cases(
                experiment, count=40, seed=729_000 + offset
            )
            repeated = build_experiment_cases(
                experiment, count=40, seed=729_000 + offset
            )
            manifest = validate_experiment_cases(
                rows, experiment=experiment, expected_count=40
            )
            repeated_manifest = validate_experiment_cases(
                repeated, experiment=experiment, expected_count=40
            )
            self.assertEqual(manifest["sha256"], repeated_manifest["sha256"])
            split_groups = {
                split: {row["group"] for row in rows if row["split"] == split}
                for split in ("train", "validation", "test")
            }
            self.assertFalse(split_groups["train"] & split_groups["validation"])
            self.assertFalse(split_groups["train"] & split_groups["test"])
            self.assertFalse(
                split_groups["validation"] & split_groups["test"]
            )

    def test_equivalent_state_donors_separate_state_from_answer(self):
        rows = build_experiment_cases(
            "equivalent_state", count=40, seed=729_101
        )
        for row in rows:
            labels = row["labels"]
            query = int(labels["query"])
            bit = 1 << query
            self.assertEqual(
                bool(int(labels["state"]) & bit),
                bool(int(labels["same_answer_state"]) & bit),
            )
            self.assertNotEqual(
                bool(int(labels["state"]) & bit),
                bool(int(labels["donor_state"]) & bit),
            )
            self.assertNotEqual(
                row["prompts"]["target"]["text"],
                row["prompts"]["same_state"]["text"],
            )

    def test_hysteresis_has_distinct_old_and_correct_outcomes(self):
        rows = build_experiment_cases(
            "reasoning_hysteresis", count=40, seed=729_103
        )
        self.assertTrue(
            all(
                row["labels"]["correct_answer"]
                != row["labels"]["old_plan_answer"]
                for row in rows
            )
        )

    def test_boundary_bandwidth_and_layer_modes_are_explicit(self):
        rows = build_experiment_cases(
            "boundary_handoff", count=8, seed=729_105
        )
        specifications = {
            item["name"]: item for item in rows[0]["evaluations"]
        }
        self.assertEqual(specifications["one_token_one_layer"]["token_width"], 1)
        self.assertEqual(
            specifications["three_tokens_one_layer"]["token_width"], 3
        )
        self.assertEqual(
            specifications["one_token_all_layers"]["layer_modes"], ["all"]
        )
        self.assertEqual(
            _patch_layers(
                mode="window3",
                center=7,
                probe_layers=[3, 7, 11],
                layer_count=12,
            ),
            [6, 7, 8],
        )

    def test_trace_alignment_crosses_surface_and_value(self):
        row = build_experiment_cases(
            "trace_alignment", count=8, seed=729_301
        )[0]
        target = row["prompts"]["target"]
        aligned = row["prompts"]["aligned_different"]
        misaligned = row["prompts"]["misaligned_different"]
        target_text = target["text"]
        aligned_text = aligned["text"]
        target_span = target_text[
            target["checkpoint_start"] : target["checkpoint_end"]
        ]
        aligned_span = aligned_text[
            aligned["checkpoint_start"] : aligned["checkpoint_end"]
        ]
        def masked(prompt):
            text = prompt["text"]
            return (
                text[: prompt["checkpoint_start"]]
                + "<VALUE>"
                + text[prompt["checkpoint_end"] :]
            )

        self.assertNotEqual(target_span, aligned_span)
        self.assertEqual(masked(target), masked(aligned))
        self.assertNotEqual(
            masked(target),
            masked(misaligned),
        )

    def test_second_wave_targets_actual_values_and_full_spans(self):
        utility = build_experiment_cases(
            "prospective_utility", count=8, seed=729_302
        )[0]
        relevant = utility["prompts"]["a_relevant"]
        self.assertTrue(
            relevant["text"][
                relevant["checkpoint_start"] : relevant["checkpoint_end"]
            ].isdigit()
        )
        correction = build_experiment_cases(
            "correction_hysteresis", count=8, seed=729_303
        )[0]
        corrected = correction["prompts"]["corrected"]
        self.assertTrue(
            corrected["text"][
                corrected["checkpoint_start"] : corrected["checkpoint_end"]
            ].isdigit()
        )
        boundary = build_experiment_cases(
            "boundary_bandwidth", count=8, seed=729_305
        )[0]
        full_span = next(
            item
            for item in boundary["evaluations"]
            if item["name"] == "history_full_all_layers"
        )
        self.assertEqual(full_span["token_width"], "all")


class CausalReasoningArtifactTests(unittest.TestCase):
    def test_feature_probes_include_text_only_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "features").mkdir()
            rows = []
            cases = []
            for index in range(20):
                split = (
                    "train"
                    if index < 10
                    else "validation"
                    if index < 15
                    else "test"
                )
                feature = Path("features") / f"{index}.npz"
                np.savez_compressed(
                    run / feature,
                    states=np.asarray(
                        [[index, index % 2], [index % 3, index]], dtype=np.float32
                    ),
                    layers=np.asarray([3, 7]),
                )
                rows.append(
                    {
                        "id": str(index),
                        "split": split,
                        "labels": {
                            "binary": index % 2,
                            "numeric": index,
                        },
                        "feature_file": feature.as_posix(),
                    }
                )
                cases.append(
                    {
                        "id": str(index),
                        "feature_prompt": "target",
                        "prompts": {
                            "target": {"text": f"case value {index}"}
                        },
                    }
                )
            result = _feature_probes(run, rows, cases)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertIn(
                "text_only_control", result["probes"]["binary"]
            )
            self.assertIn(
                "text_only_control", result["probes"]["numeric"]
            )

    def test_pending_tasks_reject_duplicate_completed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            suite = root / "suite"
            child.mkdir()
            suite.mkdir()
            rows = build_experiment_cases(
                "query_switch", count=4, seed=729_106
            )
            write_jsonl(child / "dataset.jsonl", rows)
            (suite / "config.yaml").write_text(
                "causal_reasoning_suite:\n"
                f"  runs:\n    - {child.as_posix()}\n",
                encoding="utf-8",
            )
            evaluation = child / "evaluation"
            evaluation.mkdir()
            duplicate = {"id": rows[0]["id"]}
            write_jsonl(evaluation / "cases.jsonl", [duplicate, duplicate])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                pending_tasks(suite)

    def test_partial_reduction_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "config.yaml").write_text(
                "causal_reasoning:\n"
                "  experiment: query_switch\n"
                "  hypothesis: test\n"
                "  probe_layers: [3]\n",
                encoding="utf-8",
            )
            rows = build_experiment_cases(
                "query_switch", count=5, seed=729_106
            )
            write_jsonl(run / "dataset.jsonl", rows)
            (run / "evaluation").mkdir()
            output = {
                "id": rows[0]["id"],
                "experiment": "query_switch",
                "group": rows[0]["group"],
                "split": rows[0]["split"],
                "labels": rows[0]["labels"],
                "feature_file": None,
                "results": [
                    {
                        "condition": "textual_reuse_from_query_a",
                        "layer_mode": "baseline",
                        "layer": None,
                        "token_width": 0,
                        "is_expected": True,
                        "is_expected_unconstrained": True,
                        "expected_probability": 0.8,
                        "candidate_probability_mass": 0.9,
                        "unconstrained_prediction": 0,
                    }
                ],
            }
            write_jsonl(run / "evaluation" / "cases.jsonl", [output])
            first = reduce_experiment(run)
            second = reduce_experiment(run)
            self.assertEqual(
                json.dumps(first, sort_keys=True),
                json.dumps(second, sort_keys=True),
            )
            self.assertEqual(first["status"], "partial")
            self.assertEqual(first["validation_selected_test"], [])


if __name__ == "__main__":
    unittest.main()
