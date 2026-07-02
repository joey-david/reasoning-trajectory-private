from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from collections import Counter

import numpy as np
import torch

from src.experiments.causal_patching import (
    ProjectionSubspace,
    completion_state_index,
    load_completed_patches,
    output_degeneration_reasons,
    select_control_donor,
)
from src.experiments.boundary_interventions import position_matched_boundaries
from src.experiments.common import interval_dynamics, robust_spike_indices
from src.experiments.correctness_prediction import prefix_representations
from src.experiments.gold_answers import capture_gold_answer
from src.experiments.thought_units import apply_gold_answer_scores
from src.experiments.patching_analysis import (
    analyze_causal_patching,
    validate_h3_smoke,
)
from src.experiments.process_isomers import (
    deduplicate_pair_candidates,
    sequence_edit_distance,
)
from src.experiments.sentence_lattice import (
    boundary_f1,
    object_update_costs,
    optimal_partition,
    partition_cost,
    squared_error_costs,
)
from src.experiments.symbolic import extract_symbolic_updates
from src.prompting.templates import build_prompt
from src.runtime.artifact_store import (
    load_component_states_npz,
    load_hidden_states_npz,
    save_hidden_states_npz,
)
from src.runtime.paths import REPO_ROOT, resolve_repo_path


class FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return "|".join(
            f"{message['role']}={message['content']}" for message in messages
        )


class ExperimentPrimitiveTests(unittest.TestCase):
    def test_symbolic_updates_verify_arithmetic_and_keep_final_extract(self):
        text = (
            "First 7 + 4 = 11. A mistaken 2 + 2 = 5 is ignored. "
            "The answer is 11. Final Answer: 11"
        )
        updates = extract_symbolic_updates(text, token_count=40)
        self.assertEqual(
            [update.operator for update in updates],
            ["OPERATE", "EXTRACT"],
        )
        self.assertEqual(updates[0].operation_signature, "ADD")
        self.assertEqual(updates[-1].value, 11)

    def test_robust_spikes_are_local_and_separated(self):
        magnitudes = np.ones(20, dtype=np.float32)
        magnitudes[[5, 6, 15]] = [10, 8, 12]
        self.assertEqual(
            robust_spike_indices(magnitudes, min_distance=3).tolist(),
            [5, 15],
        )

    def test_chat_demonstrations_precede_final_question(self):
        prompt = build_prompt(
            {"question": "Target?"},
            {
                "mode": "chat",
                "system": "System",
                "instruction": "Solve.",
                "demonstrations": [
                    {"user": "Example?", "assistant": "Example answer."}
                ],
            },
            FakeTokenizer(),
        )
        self.assertEqual(
            prompt,
            "system=System|user=Example?|assistant=Example answer.|"
            "user=Solve.\n\nTarget?",
        )

    def test_component_states_round_trip_with_residuals(self):
        values = np.arange(48, dtype=np.float32).reshape(3, 2, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "states.npz"
            save_hidden_states_npz(
                path=path,
                hidden_states=values,
                component_states={"mlp_output": values / 2},
                layer_indices=[9, -1],
                storage_dtype="float16",
            )
            residual, residual_layers = load_hidden_states_npz(path)
            mlp, mlp_layers = load_component_states_npz(path, "mlp_output")
        np.testing.assert_allclose(residual, values, atol=0.02)
        np.testing.assert_allclose(mlp, values / 2, atol=0.02)
        self.assertEqual(residual_layers, [9, -1])
        self.assertEqual(mlp_layers, [9, -1])

    def test_relative_paths_resolve_from_checkout_root(self):
        self.assertEqual(resolve_repo_path("runs"), REPO_ROOT / "runs")
        self.assertTrue((REPO_ROOT / "src").is_dir())

    def test_interval_dynamics_distinguish_wave_from_peak(self):
        wave = np.arange(5, dtype=np.float32)[:, None]
        peaked = np.asarray([0, 0, 0, 0, 4], dtype=np.float32)[:, None]
        wave_metrics = interval_dynamics(wave, 0, 4)
        peak_metrics = interval_dynamics(peaked, 0, 4)
        self.assertAlmostEqual(wave_metrics.path_length, 4.0)
        self.assertAlmostEqual(wave_metrics.effective_width_fraction, 1.0)
        self.assertAlmostEqual(wave_metrics.peak_share, 0.25)
        self.assertAlmostEqual(peak_metrics.effective_width_tokens, 1.0)
        self.assertAlmostEqual(peak_metrics.peak_share, 1.0)

    def test_sentence_partition_finds_piecewise_constant_change(self):
        values = np.asarray([0.0, 0.0, 4.0, 4.0])[:, None]
        costs = squared_error_costs(values)
        boundaries = optimal_partition(costs, segments=2)
        self.assertEqual(boundaries.tolist(), [1])
        self.assertAlmostEqual(partition_cost(costs, boundaries), 0.0)

    def test_object_partition_prefers_one_update_per_segment(self):
        costs = object_update_costs(np.asarray([0, 1, 0, 1]))
        boundaries = optimal_partition(costs, segments=2)
        self.assertEqual(partition_cost(costs, boundaries), 0.0)

    def test_boundary_f1_matches_each_expected_boundary_once(self):
        self.assertAlmostEqual(
            boundary_f1(
                np.asarray([2, 3]),
                np.asarray([3]),
                tolerance=1,
            ),
            2.0 / 3.0,
        )

    def test_position_matched_boundaries_are_distinct(self):
        selected = position_matched_boundaries(
            np.asarray([1, 4, 7, 9]),
            sentence_count=12,
            target_positions=[0.3, 0.7],
        )
        self.assertEqual(selected.tolist(), [4, 7])

    def test_gold_answer_capture_persists_aligned_manifest_and_states(self):
        class FakeTokenizer:
            bos_token_id = 1
            eos_token_id = 2

            def __call__(self, text, **_kwargs):
                self.text = text
                return {"input_ids": [7, 8]}

        captured = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6)
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            with patch(
                "src.experiments.gold_answers.capture_selected_activations",
                return_value=(captured, {}),
            ) as capture:
                record = capture_gold_answer(
                    run_path=run_path,
                    model=object(),
                    tokenizer=FakeTokenizer(),
                    sample={
                        "id": "sample",
                        "gold_answer": "worked solution",
                    },
                    layers=[-1],
                    storage_dtype="float16",
                    max_tokens=10,
                )
            states, layers = load_hidden_states_npz(
                run_path / record["hidden_states_file"]
            )
            manifest = json.loads(
                (run_path / "gold_answers" / "manifest.jsonl").read_text()
            )
        capture.assert_called_once()
        self.assertEqual(capture.call_args.kwargs["full_seq_ids"], [1, 7, 8])
        self.assertEqual(capture.call_args.kwargs["prompt_len"], 1)
        np.testing.assert_allclose(states, captured.numpy())
        self.assertEqual(layers, [-1])
        self.assertEqual(manifest["sample_id"], "sample")

    def test_gold_answer_scores_replace_cross_rollout_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_run = root / "gold"
            hidden_path = gold_run / "gold_answers" / "hidden_states" / "a.npz"
            save_hidden_states_npz(
                path=hidden_path,
                hidden_states=np.asarray(
                    [[[1.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]]
                ),
                layer_indices=[-1],
                storage_dtype="float32",
            )
            (gold_run / "gold_answers" / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "hidden_states_file": (
                            "gold_answers/hidden_states/a.npz"
                        ),
                    }
                )
                + "\n"
            )
            (gold_run / "gold_answers" / "metadata.json").write_text(
                json.dumps({"alignment": "test alignment"})
            )
            cache = {
                "offsets": np.asarray([0, 3]),
                "raw": np.asarray(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                ),
                "answer_score": np.asarray([0.0, 0.1, 0.2]),
                "records": [
                    {
                        "sample_id": "a",
                        "seed": 1,
                        "train": False,
                    }
                ],
            }
            updated, information = apply_gold_answer_scores(
                cache,
                gold_run,
                selected_indices=[0],
            )
        self.assertEqual(information["status"], "gold_solution_alignment")
        self.assertGreater(updated["answer_score"][0], updated["answer_score"][1])

    def test_completion_alignment_is_derived_from_token_end(self):
        point = {"token_end": 7, "state_index": 3}
        row = {"generated_token_ids": list(range(20))}
        self.assertEqual(completion_state_index(point, row), 8)

    def test_completion_alignment_rejects_uncaptured_final_state(self):
        point = {"token_end": 4}
        row = {"generated_token_ids": list(range(5))}
        with self.assertRaisesRegex(ValueError, "no captured completion state"):
            completion_state_index(point, row)

    def test_mismatched_control_uses_its_symbolic_completion_state(self):
        pair = {
            "pair_id": 1,
            "graph_signature": "target",
            "target": {"sample_id": "target", "seed": 0, "token_end": 2},
            "donor": {"sample_id": "equivalent", "seed": 0, "token_end": 2},
        }
        mismatch = {
            "pair_id": 2,
            "graph_signature": "different",
            "donor": {"sample_id": "mismatch", "seed": 0, "token_end": 5},
        }
        rows = {
            ("target", 0): {"generated_token_ids": list(range(10))},
            ("mismatch", 0): {"generated_token_ids": list(range(10))},
        }
        donor, state_index = select_control_donor(
            condition="mismatched",
            pair=pair,
            pairs=[pair, mismatch],
            rows=rows,
            target_row=rows[("target", 0)],
        )
        self.assertEqual(donor["sample_id"], "mismatch")
        self.assertEqual(state_index, 6)

    def test_random_control_uses_exact_position_from_unrelated_trace(self):
        pair = {
            "pair_id": 1,
            "graph_signature": "target",
            "target": {"sample_id": "target", "seed": 0, "token_end": 7},
            "donor": {"sample_id": "equivalent", "seed": 0, "token_end": 4},
        }
        alternative = {
            "pair_id": 2,
            "graph_signature": "different",
            "donor": {"sample_id": "target", "seed": 0, "token_end": 2},
            "target": {"sample_id": "long", "seed": 0, "token_end": 12},
        }
        rows = {
            ("target", 0): {"generated_token_ids": list(range(20))},
            ("long", 0): {"generated_token_ids": list(range(20))},
        }
        donor, state_index = select_control_donor(
            condition="position_random",
            pair=pair,
            pairs=[pair, alternative],
            rows=rows,
            target_row=rows[("target", 0)],
        )
        self.assertEqual(donor["sample_id"], "long")
        self.assertEqual(state_index, 8)

    def test_symbolic_prefix_features_use_only_completed_updates(self):
        states = np.arange(5, dtype=np.float32)[:, None]
        segments = [
            SimpleNamespace(token_start=0, token_end=1),
            SimpleNamespace(token_start=2, token_end=4),
        ]
        updates = [
            SimpleNamespace(token_start=0, token_end=1),
            SimpleNamespace(token_start=2, token_end=3),
        ]
        features = prefix_representations(states, segments, updates, checkpoint=3)
        np.testing.assert_allclose(
            features["sentence_mean_variance"],
            [0.5, 0.0],
        )
        np.testing.assert_allclose(
            features["symbolic_update_mean_variance"],
            [2.0, 0.0],
        )

    def test_subspace_swap_replaces_only_projection_row_space(self):
        weight = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        projection = ProjectionSubspace(
            path=Path("projection.pt"),
            weight=weight,
            pseudoinverse=torch.linalg.pinv(weight),
            rank=2,
            condition_number=1.0,
        )
        reconstructed, diagnostics = projection.swap(
            target=torch.tensor([1.0, 2.0, 3.0]),
            donor=torch.tensor([4.0, 6.0, 9.0]),
        )
        torch.testing.assert_close(reconstructed, torch.tensor([4.0, 6.0, 3.0]))
        self.assertLess(
            diagnostics["coordinate_reconstruction_relative_residual"],
            1e-6,
        )
        self.assertLess(
            diagnostics["orthogonal_leakage_relative_residual"],
            1e-6,
        )

    def test_completed_patch_keys_are_mode_specific(self):
        rows = [
            {
                "pair_id": 1,
                "patch_mode": "full",
                "condition": "equivalent",
                "continuation": 0,
            },
            {
                "pair_id": 1,
                "patch_mode": "subspace",
                "condition": "equivalent",
                "continuation": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "continuations.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            completed = load_completed_patches(path)
        self.assertEqual(
            completed,
            {
                (1, "full", "equivalent", 0),
                (1, "subspace", "equivalent", 0),
            },
        )

    def test_symbolic_history_distance_detects_reordered_derivations(self):
        self.assertEqual(
            sequence_edit_distance(("ADD:a", "MULTIPLY:b"), ("MULTIPLY:b", "ADD:a")),
            2,
        )
        self.assertEqual(
            sequence_edit_distance(("ADD:a",), ("ADD:a",)),
            0,
        )

    def test_process_isomer_deduplication_removes_revisited_pair_state(self):
        def point(seed, update_index, token_end, history):
            return {
                "sample_id": "sample",
                "seed": seed,
                "graph_signature": "state",
                "update_index": update_index,
                "token_end": token_end,
                "_structural_history": history,
            }

        candidates = [
            {
                "graph_signature": "state",
                "donor": point(1, donor_index, donor_end, ("path-a",)),
                "target": point(2, 3, 30, ("path-b",)),
            }
            for donor_index, donor_end in ((5, 50), (7, 70))
        ]
        rejections = Counter()
        deduplicated = deduplicate_pair_candidates(candidates, rejections)
        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0]["donor"]["update_index"], 5)
        self.assertEqual(rejections["duplicate_pair_history"], 1)

    def test_output_degeneration_is_conservative(self):
        self.assertEqual(output_degeneration_reasons([1, 2, 3], "normal"), [])
        self.assertIn(
            "repeated_token_run",
            output_degeneration_reasons([7] * 32, "repeated"),
        )

    def test_h3_smoke_rejects_different_duplicated_baselines(self):
        conditions = ("baseline", "equivalent", "position_random", "mismatched")
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            pair_path = run_path / "pairs.jsonl"
            pair_path.write_text(
                json.dumps({"pair_id": 1}) + "\n",
                encoding="utf-8",
            )
            config = {
                "patching": {
                    "pairs": str(pair_path),
                    "patch_modes": ["full", "subspace"],
                    "conditions": list(conditions),
                }
            }
            (run_path / "config.yaml").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            rows = []
            for mode in ("full", "subspace"):
                for condition in conditions:
                    row = {
                        "pair_id": 1,
                        "patch_mode": mode,
                        "condition": condition,
                        "continuation": 0,
                        "target_question": "question",
                        "generated_token_ids": [1 if mode == "full" else 2],
                        "has_valid_answer": True,
                        "degenerate_output": False,
                        "hit_token_limit": False,
                    }
                    if mode == "subspace":
                        row["reconstruction"] = {
                            "coordinate_reconstruction_relative_residual": 0.0,
                            "orthogonal_leakage_relative_residual": 0.0,
                        }
                    rows.append(row)
            continuation_path = run_path / "patching" / "continuations.jsonl"
            continuation_path.parent.mkdir()
            continuation_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "smoke gate failed"):
                validate_h3_smoke(run_path, pair_count=1)
            report = json.loads(
                (run_path / "preflight" / "smoke_report.json").read_text()
            )
            self.assertIn(
                "pair 1 duplicated baselines differ at continuation 0",
                report["errors"],
            )

    def test_h3_analysis_uses_actual_manifest_pair_count(self):
        conditions = ("baseline", "equivalent", "position_random", "mismatched")
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            pair_path = run_path / "pairs.jsonl"
            pair_path.write_text(
                json.dumps({"pair_id": 1}) + "\n",
                encoding="utf-8",
            )
            config = {
                "patching": {
                    "pairs": str(pair_path),
                    "component": "attention_output",
                    "layer": 18,
                    "projection_path": "projection.pt",
                    "patch_modes": ["full", "subspace"],
                    "conditions": list(conditions),
                    "max_pairs": 3,
                    "continuations_per_condition": 1,
                }
            }
            (run_path / "config.yaml").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            rows = [
                {
                    "pair_id": 1,
                    "patch_mode": mode,
                    "condition": condition,
                    "continuation": 0,
                    "target_question": "question",
                    "degenerate_output": False,
                    "has_valid_answer": True,
                    "matches_target_answer": True,
                    "matches_donor_answer": None,
                    "matches_neither_answer": False,
                    "hit_token_limit": False,
                }
                for mode in ("full", "subspace")
                for condition in conditions
            ]
            continuation_path = run_path / "patching" / "continuations.jsonl"
            continuation_path.parent.mkdir()
            continuation_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            report_path = analyze_causal_patching(run_path)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["design"]["expected_pairs"], 1)
            self.assertEqual(report["design"]["expected_total_continuations"], 8)
            self.assertEqual(report["design"]["completion_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
