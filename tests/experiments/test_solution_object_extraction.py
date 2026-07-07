from __future__ import annotations

import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.experiments.solution_object_extraction.anchors import anchor_token_range
from src.experiments.solution_object_extraction.decoders import (
    decode_representation,
)
from src.experiments.solution_object_extraction.iso_dataset import build_bank
from src.experiments.solution_object_extraction.mixed_trajectories import (
    unsupervised_consensus_scores,
)
from src.experiments.solution_object_extraction.patching import (
    ablate_subspace,
    parse_final_number,
    swap_subspace,
)
from src.experiments.solution_object_extraction.pipeline import mixed_source_status
from src.experiments.solution_object_extraction.projections import (
    fit_group_projection,
    project,
)
from src.experiments.solution_object_extraction.nonlinear import ObjectEncoder
from src.experiments.solution_object_extraction.sweeps import (
    causal_leakage_pareto,
    forward_with_layer_deltas,
    with_template_validation,
)
from src.experiments.solution_object_extraction.writer import (
    finite_masked_kl_divergence,
    split_writer_pairs,
    surface_js_divergence,
)
from src.experiments.solution_object_extraction.ablations import (
    select_ablation_grid_cells,
    trajectory_gate,
)
from src.experiments.solution_object_extraction.retrieval import (
    evaluate_retrieval,
)
from src.experiments.solution_object_extraction.schemas import make_graph
from src.experiments.solution_object_extraction.storage import (
    read_jsonl,
    write_jsonl,
    write_npz,
)


class SolutionObjectExtractionTests(unittest.TestCase):
    def test_graph_hash_and_storage_contracts_are_stable(self):
        graph = make_graph(
            graph_id="g",
            operation="ADD",
            operand_a=2,
            operand_b=3,
            result=5,
        ).to_record()
        self.assertTrue(graph["graph_hash"].startswith("sha256:"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "rows.jsonl", [graph])
            write_npz(root / "vectors.npz", values=np.eye(2))
            self.assertEqual(read_jsonl(root / "rows.jsonl")[0], graph)
            with np.load(root / "vectors.npz") as data:
                np.testing.assert_array_equal(data["values"], np.eye(2))

    def test_anchor_alignment_requires_one_exact_interval(self):
        class CharacterTokenizer:
            def __call__(self, text, **_kwargs):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

        self.assertEqual(
            anchor_token_range(CharacterTokenizer(), "prefix anchor", "anchor"),
            (7, 12),
        )
        with self.assertRaisesRegex(ValueError, "not unique"):
            anchor_token_range(CharacterTokenizer(), "x x", "x")

    def test_bank_has_disjoint_surfaces_and_train_prototypes(self):
        _graphs, records = build_bank(graph_count=8, include_corruptions=True)
        train_graphs = {
            row["canonical_graph_id"] for row in records if row["split"] == "train"
        }
        heldout = {
            row["canonical_graph_id"]
            for row in records
            if row["split"].startswith("heldout")
        }
        self.assertTrue(heldout.issubset(train_graphs))
        train_vocab = {
            row["surface"]["lexical_family"]
            for row in records
            if row["split"] == "train"
        }
        heldout_vocab = {
            row["surface"]["lexical_family"]
            for row in records
            if row["split"] == "heldout_vocab"
        }
        self.assertTrue(train_vocab.isdisjoint(heldout_vocab))

    def test_projection_and_interventions_preserve_orthogonal_coordinates(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(30, 8)).astype(np.float32)
        labels = np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        mean, basis = fit_group_projection(values, labels, max_dim=2)
        np.testing.assert_allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-5)
        target, donor = values[:2]
        swapped = swap_subspace(target, donor, basis)
        np.testing.assert_allclose(
            (swapped - donor) @ basis.T, np.zeros(len(basis)), atol=1e-5
        )
        ablated = ablate_subspace(target, mean, basis)
        np.testing.assert_allclose(
            (ablated - mean) @ basis.T, np.zeros(len(basis)), atol=1e-5
        )
        self.assertEqual(project(values, mean, basis).shape, (30, len(basis)))

    def test_supervised_projection_does_not_fill_with_unlabeled_pca(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(30, 20)).astype(np.float32)
        labels = np.asarray(["a"] * 10 + ["b"] * 10 + ["c"] * 10)
        _mean, basis = fit_group_projection(values, labels, max_dim=16)
        self.assertLessEqual(len(basis), 2)

    def test_residual_encoder_starts_at_the_linear_readout(self):
        model = ObjectEncoder(
            input_dim=6,
            hidden_dim=8,
            latent_dim=4,
            class_counts={"graph": 2},
            base_dim=2,
        )
        values = torch.tensor([[3.0, 4.0, 2.0, 1.0, 0.0, -1.0]])
        latent, _heads = model(values, adversarial_scale=0.0)
        torch.testing.assert_close(
            latent,
            torch.tensor([[0.6, 0.8, 0.0, 0.0]]),
        )

    def test_residual_encoder_can_truncate_a_larger_base(self):
        model = ObjectEncoder(
            input_dim=64,
            hidden_dim=8,
            latent_dim=32,
            class_counts={"graph": 2},
            base_dim=32,
        )
        values = torch.arange(1, 65, dtype=torch.float32)[None]
        latent, _heads = model(values, adversarial_scale=0.0)
        expected = torch.nn.functional.normalize(values[:, :32], dim=1)
        torch.testing.assert_close(latent, expected)

    def test_multilayer_deltas_accumulate_on_the_live_stream(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(2, 2)
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Identity(), torch.nn.Identity()]
                )

            def get_input_embeddings(self):
                return self.embedding

            def forward(self, input_ids, **_kwargs):
                hidden = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    2,
                    device=input_ids.device,
                )
                for layer in self.layers:
                    hidden = layer(hidden)
                return SimpleNamespace(logits=hidden)

        class ToyTokenizer:
            def __call__(self, _text, **_kwargs):
                return {"input_ids": torch.tensor([[0]])}

        model = ToyModel()
        deltas = {
            0: np.asarray([[1.0, 0.0]], dtype=np.float32),
            1: np.asarray([[2.0, 0.0]], dtype=np.float32),
        }
        with patch(
            "src.experiments.solution_object_extraction.sweeps.get_decoder_layers",
            return_value=model.layers,
        ):
            logits = forward_with_layer_deltas(
                model, ToyTokenizer(), "x", deltas
            )
        torch.testing.assert_close(logits, torch.tensor([3.0, 0.0]))

    def test_template_selection_and_writer_validation_are_disjoint(self):
        _graphs, records = build_bank(graph_count=8, include_corruptions=True)
        selected = with_template_validation(records)
        self.assertTrue(
            any(row["split"] == "template_validation" for row in selected)
        )
        train, validation, manifest = split_writer_pairs(
            records, train_limit=12, validation_limit=4
        )
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertEqual(manifest["overlap"], [])

    def test_surface_divergence_is_zero_for_identical_logits(self):
        logits = torch.tensor([0.1, -0.2, 0.3, 1.2])
        self.assertAlmostEqual(
            surface_js_divergence(
                logits, logits.clone(), excluded_tokens=(0, 1)
            ),
            0.0,
            places=7,
        )

    def test_writer_surface_kl_masks_answer_tokens_without_nan(self):
        baseline = torch.tensor([1000.0, -1000.0, 0.3, -0.1])
        patched = torch.tensor([-1000.0, 1000.0, 0.2, -0.2])
        value = finite_masked_kl_divergence(
            baseline,
            patched,
            excluded_tokens=(0, 1),
        )
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value), 0.0)

    def test_ablation_grid_selects_low_leakage_target_dimensions(self):
        rows = [
            {
                "layer": 29,
                "dimension": 32,
                "scope": "multi_layer",
                "patch_layers": [26, 29, 32],
                "view": "anchor_prefix",
                "lexical_probe_accuracy": 0.80,
                "object_minus_strongest_control_donor_delta": 0.47,
            },
            {
                "layer": 29,
                "dimension": 64,
                "scope": "multi_layer",
                "patch_layers": [26, 29, 32],
                "view": "anchor_prefix",
                "lexical_probe_accuracy": 0.97,
                "object_minus_strongest_control_donor_delta": 0.48,
            },
            {
                "layer": 32,
                "dimension": 16,
                "scope": "operation_interval",
                "patch_layers": [32],
                "view": "anchor_prefix",
                "lexical_probe_accuracy": 0.61,
                "object_minus_strongest_control_donor_delta": 0.34,
            },
        ]
        selected = select_ablation_grid_cells(
            rows,
            dimensions={16, 32},
            scopes={"operation_interval", "multi_layer"},
            max_lexical_probe=0.85,
            min_causal_strength=0.0,
            limit=4,
        )
        self.assertEqual([row["dimension"] for row in selected], [32, 16])
        self.assertTrue(
            all(row["lexical_probe_accuracy"] <= 0.85 for row in selected)
        )

    def test_trajectory_gate_accepts_ablation_grid_cell_schema(self):
        gate = trajectory_gate(
            retrieval={
                "retrieval": {
                    "heldout_vocab": {"top1": 0.8},
                    "heldout_template": {"top1": 0.6},
                }
            },
            nonlinear={"selected_epoch": 3},
            causal={
                "causal_strength": 0.2,
                "lexical_probe_accuracy": 0.8,
            },
            ablation_margin=0.1,
            max_lexical_probe=0.85,
            source="test",
        )
        self.assertEqual(gate["status"], "ready_for_real_trajectory")
        self.assertTrue(gate["checks"]["causal_patch_exceeds_surface_controls"])

    def test_causal_leakage_pareto_drops_dominated_cells(self):
        rows = [
            {
                "layer": 1,
                "dimension": 2,
                "scope": "final_token",
                "patch_layers": [1],
                "object_minus_strongest_control_donor_delta": 0.2,
                "lexical_probe_accuracy": 0.4,
            },
            {
                "layer": 2,
                "dimension": 2,
                "scope": "final_token",
                "patch_layers": [2],
                "object_minus_strongest_control_donor_delta": 0.1,
                "lexical_probe_accuracy": 0.5,
            },
        ]
        frontier = causal_leakage_pareto(rows)
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0]["layer"], 1)

    def test_retrieval_and_factorized_decoder_recover_separable_states(self):
        rng = np.random.default_rng(4)
        records = []
        vectors = []
        operations = ("ADD", "SUBTRACT")
        edits = ("BIND", "OPERATE")
        for repeat in range(20):
            for op_index, operation in enumerate(operations):
                for edit_index, edit in enumerate(edits):
                    vector = np.asarray(
                        [op_index * 4.0, edit_index * 4.0, repeat / 100],
                        dtype=np.float32,
                    )
                    vectors.append(vector + rng.normal(0, 0.01, 3))
                    records.append(
                        {
                            "canonical_graph_id": f"{operation}-{edit}",
                            "edit_type": edit,
                            "observed": {
                                "operation": operation,
                                "target": "result",
                                "operand_a": op_index,
                                "operand_b": edit_index,
                                "result": op_index + edit_index,
                            },
                        }
                    )
        x = np.stack(vectors)
        train = np.asarray(
            [index for index in range(len(x)) if index // 4 < 10]
        )
        test = np.asarray(
            [index for index in range(len(x)) if index // 4 >= 10]
        )
        train_labels = np.asarray(
            [records[index]["canonical_graph_id"] for index in train]
        )
        test_labels = np.asarray(
            [records[index]["canonical_graph_id"] for index in test]
        )
        retrieval, _, _ = evaluate_retrieval(
            x[train],
            x[test],
            train_labels,
            test_labels,
            [records[index] for index in test],
        )
        self.assertGreaterEqual(retrieval["top1"], 0.95)
        decoded = decode_representation(
            x[train],
            x[test],
            [records[index] for index in train],
            [records[index] for index in test],
        )
        self.assertGreaterEqual(decoded["operation"]["accuracy"], 0.95)

    def test_continuation_parser_and_consensus_scores(self):
        self.assertEqual(parse_final_number("3 + 4 = 7."), 7.0)
        vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
        scores = unsupervised_consensus_scores(
            vectors, np.asarray(["q", "q", "q"])
        )
        self.assertGreater(scores[0], scores[2])

    def test_missing_mixed_states_are_reported_without_loading_them(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            write_jsonl(
                source / "generation" / "generations.jsonl",
                [{"hidden_states_file": "generation/hidden_states/missing.npz"}],
            )
            status = mixed_source_status(source)
        self.assertFalse(status["available"])
        self.assertIn("missing 1 hidden-state", status["message"])


if __name__ == "__main__":
    unittest.main()
