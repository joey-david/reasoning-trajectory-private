"""Tests for objective-conditioned partitions and solution-object edits."""

from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from src.experiments.objective_segmentation import (
    append_objective_identity,
    boundary_budget,
    normalized_regret,
    objective_partition,
)
from src.experiments.sentence_lattice import squared_error_costs
from src.experiments.solution_object_edits import (
    admissible_bronze_update,
    typed_edit,
    validate_silver_label,
)
from src.experiments.thought_unit_probes import fit_conditioned_linear_model


class ObjectiveSegmentationTests(unittest.TestCase):
    """Cover fixed-budget OCS and deterministic object-edit contracts."""

    def test_objective_partition_and_regret_have_expected_anchors(self) -> None:
        """The exact oracle has zero regret and mean random defines one."""
        costs = squared_error_costs(np.asarray([0.0, 0.0, 10.0, 10.0]))
        oracle = objective_partition(costs, boundary_count=1)
        random_samples = [
            np.asarray([0], dtype=np.int32),
            np.asarray([2], dtype=np.int32),
        ]

        self.assertEqual(oracle.tolist(), [1])
        self.assertEqual(
            normalized_regret(costs, oracle, oracle, random_samples),
            0.0,
        )
        random_regrets = [
            normalized_regret(costs, sample, oracle, random_samples)
            for sample in random_samples
        ]
        self.assertAlmostEqual(float(np.mean(random_regrets)), 1.0)

    def test_boundary_budget_clamps_short_and_extreme_requests(self) -> None:
        """Boundary fractions always produce a valid exact budget."""
        self.assertEqual(boundary_budget(1, 0.2), 0)
        self.assertEqual(boundary_budget(5, 0.0), 1)
        self.assertEqual(boundary_budget(5, 2.0), 4)

    def test_objective_identity_is_one_hot_and_row_aligned(self) -> None:
        """Conditioning appends the same objective identity to every row."""
        features = np.ones((3, 2), dtype=np.float32)
        conditioned = append_objective_identity(features, 1, 4)

        self.assertEqual(conditioned.shape, (3, 6))
        np.testing.assert_array_equal(
            conditioned[:, 2:],
            np.asarray([[0, 1, 0, 0]] * 3),
        )

    def test_typed_bindings_preserve_variable_identity(self) -> None:
        """Equal values bound to different variables remain distinct edits."""
        first = {
            "operator": "BIND",
            "operation_signature": "BIND",
            "expression": "N=5",
            "value": 5,
            "lexical_items": ["N", "5"],
            "char_start": 0,
            "char_end": 3,
            "token_start": 0,
            "token_end": 2,
        }
        second = {**first, "expression": "cost=5", "char_end": 6}
        left = typed_edit(first, "")
        right = typed_edit(second, left.after_state)

        self.assertEqual(left.added_relations, ("BIND:n:5",))
        self.assertEqual(right.added_relations, ("BIND:cost:5",))

    def test_unit_bindings_are_not_bronze_object_edits(self) -> None:
        """Regex matches such as ``kg=75`` are rejected as unit syntax."""
        self.assertFalse(
            admissible_bronze_update({"operator": "BIND", "expression": "kg = 75"})
        )

    def test_silver_validation_rejects_ungrounded_numbers(self) -> None:
        """Silver labels cannot introduce quantities absent from the sentence."""
        errors = validate_silver_label(
            "There are 12 apples.",
            {
                "edit_type": "add_quantity",
                "entities": ["apples"],
                "quantities": [{"name": "count", "value": 13, "unit": "apples"}],
                "relations": [],
                "operation": None,
                "confidence": 0.8,
                "rationale": "Adds the count.",
            },
        )

        self.assertIn("ungrounded quantity: 13", errors)

    def test_conditioned_linear_model_learns_distinct_objective_heads(self) -> None:
        """One joint model can select different feature rules by objective."""
        rng = np.random.default_rng(7)
        features = rng.normal(size=(240, 2)).astype(np.float32)
        labels = np.column_stack([features[:, 0] > 0, features[:, 1] > 0]).astype(
            np.int8
        )
        model = fit_conditioned_linear_model(
            features,
            labels,
            epochs=6,
            batch_size=64,
        )

        for objective in range(2):
            probabilities = model.predict_proba(features, objective)
            self.assertGreater(
                roc_auc_score(labels[:, objective], probabilities),
                0.95,
            )


if __name__ == "__main__":
    unittest.main()
