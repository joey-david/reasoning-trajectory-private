from __future__ import annotations

import json
import unittest
from pathlib import Path
import tempfile

import numpy as np

from src.experiments.depth_relief.abstraction import (
    build_state_abstraction_benchmark,
    validate_abstraction_case,
)
from src.experiments.depth_relief.abstraction_information import (
    analyze_matched_history_information,
)
from src.experiments.depth_relief.abstraction_behavior import (
    summarize_abstraction_behavior,
)
from src.experiments.depth_relief.abstraction_interchange import (
    build_interchange_pairs,
    fit_implicit_state_subspaces,
    select_interchange_pairs,
    summarize_interchange,
)
from src.experiments.depth_relief.benchmark import (
    apply_rule,
    build_benchmark,
    build_transition_case,
    build_qualification_benchmark,
    condition_specs,
    qualification_condition_specs,
    render_prompt,
    render_qualification_direct_prompt,
    render_qualification_prompt,
)
from src.experiments.depth_relief.calibration import (
    build_calibration_benchmark,
    summarize_calibration_rows,
)
from src.experiments.depth_relief.decoding import (
    conditional_label_entropy,
    label_entropy,
)
from src.experiments.depth_relief.metrics import settling_depth, summarize_rows
from src.experiments.depth_relief.factorization import (
    build_factorization_benchmark,
    render_factorization_prompts,
    summarize_factorization_rows,
)
from src.experiments.depth_relief.handoff import (
    analyze_state_localization,
    summarize_handoff,
)
from src.experiments.depth_relief.qualification import summarize_qualification_rows
from src.experiments.depth_relief.routing import (
    render_routing_prompts,
    select_routing_cases,
    summarize_routing_rows,
    validate_routing_case,
)
from src.experiments.depth_relief.transfer import (
    build_transfer_split,
    fit_state_subspaces,
    render_transfer_prompts,
    summarize_transfer,
    validate_transfer_case,
)
from src.experiments.depth_relief.transfer_pipeline import handoff_eligibility


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        **_kwargs,
    ):
        result = {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class DepthReliefTests(unittest.TestCase):
    def test_empirical_label_entropies_do_not_assume_uniform_classes(self):
        self.assertAlmostEqual(
            label_entropy(np.asarray([0, 0, 0, 1])),
            0.8112781244591328,
        )
        self.assertEqual(
            conditional_label_entropy(
                np.asarray([0, 0, 1, 1]),
                np.asarray([0, 0, 1, 1]),
            ),
            0.0,
        )

    @staticmethod
    def _transfer_cases(count=80):
        cases = []
        for index in range(count):
            state = index % 8
            group = index // 8
            cases.append(
                {
                    "id": f"case_{index:03d}",
                    "format": "prose" if group % 2 else "assignments",
                    "bits": 3,
                    "initial_state": (state + 3) % 8,
                    "history": [{"kind": "add", "value": 1}],
                    "state_path": [(state + 3) % 8, state],
                    "history_steps": 1,
                    "history_family": "add",
                    "final_rule": {"kind": "pointer", "mapping": list(range(8))},
                    "current_state": state,
                    "next_state": group % 8,
                    "counterfactual_state": (state + 1) % 8,
                    "counterfactual_next_state": (group + 2) % 8,
                }
            )
        return cases

    def test_transfer_split_prevents_cross_case_answer_leakage(self):
        cases = self._transfer_cases()
        indexed = {case["id"]: case for case in cases}
        split = build_transfer_split(cases, seed=47)
        self.assertGreaterEqual(len(split["validation"]), 15)
        self.assertGreaterEqual(len(split["test"]), 15)
        for recipient_id, branches in split["donors"].items():
            recipient = indexed[recipient_id]
            for branch, state_key, target_key in (
                ("gold", "current_state", "next_state"),
                ("counterfactual", "counterfactual_state", "counterfactual_next_state"),
            ):
                donor_spec = branches[branch]
                donor = indexed[donor_spec["case_id"]]
                donor_state_key = "current_state" if donor_spec["condition"] == "materialized" else "counterfactual_state"
                donor_target_key = "next_state" if donor_spec["condition"] == "materialized" else "counterfactual_next_state"
                self.assertEqual(donor[donor_state_key], recipient[state_key])
                self.assertNotEqual(donor[donor_target_key], recipient[target_key])
                self.assertEqual(donor["format"], recipient["format"])

    def test_abstraction_benchmark_forms_balanced_history_equivalence_classes(self):
        cases = build_state_abstraction_benchmark(
            {
                "bits": 3,
                "history_steps": [2, 4],
                "groups_per_horizon": 3,
                "paths_per_state": 8,
                "formats": ["prose"],
                "seed": 31,
            }
        )
        self.assertEqual(len(cases), 384)
        self.assertEqual(
            {
                split: sum(case["abstraction_split"] == split for case in cases)
                for split in ("train", "validation", "test")
            },
            {"train": 128, "validation": 128, "test": 128},
        )
        for group in {case["abstraction_group"] for case in cases}:
            group_cases = [case for case in cases if case["abstraction_group"] == group]
            self.assertEqual(
                {
                    (case["current_state"], case["path_code"])
                    for case in group_cases
                },
                {(state, path) for state in range(8) for path in range(8)},
            )
            for path in range(8):
                path_cases = [case for case in group_cases if case["path_code"] == path]
                prefixes = {json.dumps(case["history"][:-1]) for case in path_cases}
                self.assertEqual(len(prefixes), 1)
            history_steps = int(group_cases[0]["history_steps"])
            for step in range(1, history_steps):
                counts = {
                    state: sum(case["state_path"][step] == state for case in group_cases)
                    for state in range(8)
                }
                self.assertEqual(len(set(counts.values())), 1)

    def test_abstraction_prompt_and_capture_positions_validate(self):
        case = build_state_abstraction_benchmark(
            {
                "bits": 3,
                "history_steps": [2],
                "groups_per_horizon": 3,
                "paths_per_state": 8,
                "formats": ["prose"],
                "seed": 31,
            }
        )[0]
        record = validate_abstraction_case(
            tokenizer=CharacterTokenizer(), case=case, config={}
        )
        self.assertEqual(
            [row["name"] for row in record["compose_positions"]],
            ["start", "history_step_1", "history_step_2", "final_rule", "answer"],
        )
        self.assertEqual(
            [row["name"] for row in record["update_positions"]],
            ["state", "answer"],
        )

    def test_matched_pairs_and_implicit_projection_preserve_contract(self):
        cases = build_state_abstraction_benchmark(
            {
                "bits": 3,
                "history_steps": [2],
                "groups_per_horizon": 3,
                "paths_per_state": 8,
                "formats": ["prose"],
                "seed": 31,
            }
        )
        rows = []
        activations = {}
        for case in cases:
            conditions = {
                name: {"is_expected_unconstrained": name != "compose"}
                for name in (
                    "read",
                    "update",
                    "synthesize",
                    "compose",
                    "history_step_1",
                    "history_step_2",
                )
            }
            conditions["compose"]["unconstrained_prediction"] = case[
                "diagnostic_targets"
            ]["final_on_start"]
            positions = [
                {"name": "start"},
                {"name": "history_step_1"},
                {"name": "history_step_2"},
                {"name": "final_rule"},
                {"name": "answer"},
            ]
            rows.append(
                {
                    "id": case["id"],
                    "history_steps": 2,
                    "next_state": case["next_state"],
                    "diagnostic_targets": case["diagnostic_targets"],
                    "conditions": conditions,
                    "compose_positions": positions,
                }
            )
            trace = np.zeros((5, 3, 16), dtype=np.float32)
            trace[2, :, int(case["current_state"])] = 1
            activations[case["id"]] = {"compose_trace": trace}
        pairs = build_interchange_pairs(cases, rows)
        self.assertEqual(len(pairs), len(cases))
        behavior = summarize_abstraction_behavior(
            {case["id"]: case for case in cases},
            {row["id"]: row for row in rows},
        )
        self.assertEqual(
            behavior["overall"]["causal_qualification"]["mean"], 1.0
        )
        self.assertEqual(
            behavior["overall"]["compose_given_causal_qualification"]["mean"],
            0.0,
        )
        selected = select_interchange_pairs(pairs, max_per_group=8, seed=73)
        self.assertEqual(len(selected), 24)
        self.assertEqual(
            {
                (pair["group"], pair["recipient_state"])
                for pair in selected
            },
            {
                (group, state)
                for group in {case["abstraction_group"] for case in cases}
                for state in range(8)
            },
        )
        indexed = {case["id"]: case for case in cases}
        for pair in pairs:
            recipient = indexed[pair["recipient_id"]]
            different = indexed[pair["different_state_source_id"]]
            same = indexed[pair["same_state_source_id"]]
            self.assertEqual(recipient["history"][:-1], different["history"][:-1])
            self.assertNotEqual(recipient["current_state"], different["current_state"])
            self.assertEqual(recipient["current_state"], same["current_state"])
        projection = fit_implicit_state_subspaces(
            cases=indexed,
            captures={row["id"]: row for row in rows},
            activations=activations,
            rank=7,
            seed=73,
        )
        self.assertEqual(projection["state_basis"].shape, (3, 16, 7))
        overlaps = np.einsum(
            "lhr,lhs->lrs",
            projection["state_basis"],
            projection["random_basis"],
        )
        self.assertLess(float(np.abs(overlaps).max()), 1e-5)

    def test_information_map_separates_state_and_path(self):
        cases_list = build_state_abstraction_benchmark(
            {
                "bits": 3,
                "history_steps": [2],
                "groups_per_horizon": 3,
                "paths_per_state": 8,
                "formats": ["prose"],
                "seed": 31,
            }
        )
        cases = {case["id"]: case for case in cases_list}
        captures = {}
        activations = {}
        for case in cases_list:
            positions = [
                {"name": "start"},
                {"name": "history_step_1"},
                {"name": "history_step_2"},
                {"name": "final_rule"},
                {"name": "answer"},
            ]
            captures[case["id"]] = {
                "compose_positions": positions,
                "update_positions": [{"name": "state"}, {"name": "answer"}],
            }
            trace = np.zeros((5, 3, 16), dtype=np.float32)
            state_labels = [
                int(case["initial_state"]),
                int(case["state_path"][1]),
                int(case["current_state"]),
                int(case["current_state"]),
                int(case["next_state"]),
            ]
            for position, state in enumerate(state_labels):
                trace[position, :, state] = 5
                trace[position, :, 8 + int(case["path_code"])] = 2
            synthesis = np.zeros((1, 3, 16), dtype=np.float32)
            synthesis[0, :, int(case["current_state"])] = 5
            update = np.zeros((2, 3, 16), dtype=np.float32)
            update[0, :, int(case["current_state"])] = 5
            activations[case["id"]] = {
                "compose_trace": trace,
                "synthesize_trace": synthesis,
                "update_trace": update,
            }
        report = analyze_matched_history_information(
            cases=cases,
            captures=captures,
            activations=activations,
            rank=7,
            seed=73,
        )
        implicit = report["sources"]["implicit_history"]["selected_test"]
        self.assertEqual(implicit["state"]["accuracy"]["mean"], 1.0)
        self.assertGreater(
            implicit["path_given_state"]["information_lower_bound_bits"]["mean"],
            0,
        )

    def test_interchange_selects_on_validation_and_clusters_test_groups(self):
        def distribution(target: int, probability: float) -> list[float]:
            values = [(1 - probability) / 7] * 8
            values[target] = probability
            return values

        def record(
            target: int, probability: float, expected: bool = False
        ) -> dict[str, object]:
            probabilities = distribution(target, probability)
            return {
                "final_candidate_probabilities": probabilities,
                "final_candidate_logprobabilities": np.log(probabilities).tolist(),
                "candidate_probability_mass": 1.0,
                "is_expected_unconstrained": expected,
            }

        pairs = []
        captures = {}
        patches = []
        for split in ("validation", "test"):
            for group_index in range(2):
                case_id = f"{split}_{group_index}"
                pairs.append(
                    {
                        "id": case_id,
                        "split": split,
                        "group": f"{split}_g{group_index}",
                        "recipient_target": 0,
                        "different_target": 1,
                    }
                )
                captures[case_id] = {
                    "conditions": {"compose": record(0, 0.125)}
                }
                for layer, state_probability in ((23, 0.75), (31, 0.25)):
                    conditions = {
                        "state_different": record(
                            1, state_probability, expected=layer == 23
                        ),
                        "full_different": record(1, 0.8, expected=True),
                        "random_different": record(1, 0.125),
                        "state_same": record(0, 0.125),
                        "full_same": record(0, 0.125),
                    }
                    patches.append(
                        {"id": case_id, "layer": layer, "conditions": conditions}
                    )
        report = summarize_interchange(
            captures=captures,
            pairs=pairs,
            patches=patches,
            gate={},
        )
        self.assertEqual(report["layer_selection"]["selected"], 23)
        self.assertEqual(
            report["interpretation"], "causal_history_quotient_at_the_endpoint"
        )
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(
            report["heldout"]["metrics"]["state_different"]
            ["target_probability_shift"]["cluster_n"],
            2,
        )

    def test_handoff_requires_both_prespecified_gates(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            output = run_path / "depth_relief" / "state_transfer"
            output.mkdir(parents=True)
            (output / "localization_summary.json").write_text(
                json.dumps(
                    {
                        "gate": {"passed": False},
                        "layer_selection": {"selected": 45},
                    }
                ),
                encoding="utf-8",
            )
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "gate": {"passed": True},
                        "layer_selection": {"selected": 47},
                    }
                ),
                encoding="utf-8",
            )

            report = handoff_eligibility(run_path)

        self.assertFalse(report["eligible"])
        self.assertFalse(report["localization_gate_passed"])
        self.assertTrue(report["transfer_gate_passed"])

    def test_transfer_prompt_and_state_subspace_contracts(self):
        cases = self._transfer_cases()
        prompts = render_transfer_prompts(
            tokenizer=CharacterTokenizer(), case=cases[0], config={}
        )
        self.assertEqual([prompt["name"] for prompt in prompts], ["compose", "materialized", "counterfactual"])
        self.assertEqual(
            validate_transfer_case(
                tokenizer=CharacterTokenizer(), case=cases[0], config={}
            )["condition_count"],
            3,
        )
        split = build_transfer_split(cases, seed=47)
        indexed = {case["id"]: case for case in cases}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case_id in split["train"]:
                case = indexed[case_id]
                compose = np.zeros((3, 16), dtype=np.float16)
                materialized = compose.copy()
                counterfactual = compose.copy()
                materialized[:, int(case["current_state"])] = 1
                counterfactual[:, int(case["counterfactual_state"])] = 1
                np.savez_compressed(
                    root / f"{case_id}.npz",
                    compose=compose,
                    materialized=materialized,
                    counterfactual=counterfactual,
                )
            projection = fit_state_subspaces(
                cases=indexed,
                split=split,
                activation_dir=root,
                rank=8,
                seed=47,
            )
        self.assertEqual(projection["state_basis"].shape, (3, 16, 8))
        gram = projection["state_basis"][0].T @ projection["state_basis"][0]
        np.testing.assert_allclose(gram, np.eye(8), atol=1e-5)

    def test_transfer_report_selects_on_validation_and_scores_heldout(self):
        cases = self._transfer_cases()
        indexed = {case["id"]: case for case in cases}
        split = build_transfer_split(cases, seed=47)
        captures = {}
        patches = []
        for case_id in [*split["validation"], *split["test"]]:
            case = indexed[case_id]
            captures[case_id] = {
                "conditions": {
                    "compose": {"final_candidate_probabilities": [0.125] * 8}
                }
            }
            conditions = {}
            for mode in (
                "state_gold",
                "state_counterfactual",
                "full_gold",
                "random_gold",
                "random_counterfactual",
            ):
                target = (
                    case["counterfactual_next_state"]
                    if "counterfactual" in mode
                    else case["next_state"]
                )
                probabilities = [0.0] * 8
                probabilities[target] = 0.9 if mode.startswith(("state", "full")) else 0.125
                conditions[mode] = {
                    "final_candidate_probabilities": probabilities,
                    "is_expected_unconstrained": mode.startswith(("state", "full")),
                }
            patches.append({"id": case_id, "layer": 3, "conditions": conditions})
        report = summarize_transfer(
            cases=indexed,
            split=split,
            captures=captures,
            patches=patches,
            gate={},
        )
        self.assertEqual(report["layer_selection"]["selected"], 3)
        self.assertTrue(report["gate"]["passed"])

    def test_routing_confirmation_selects_synthesized_cases_and_matches_state_token(self):
        case = build_factorization_benchmark(
            {
                "history_families": ["add"],
                "bits": [3],
                "history_steps": [1],
                "formats": ["prose"],
                "examples_per_cell": 1,
                "seed": 37,
            }
        )[0]
        passed = {"is_expected_unconstrained": True}
        failed = {"is_expected_unconstrained": False}
        row = {
            "id": case["id"],
            "conditions": {"read": passed, "update": passed, "synthesize": passed},
        }
        self.assertEqual(select_routing_cases([case], [row]), [case])
        row["conditions"]["synthesize"] = failed
        self.assertEqual(select_routing_cases([case], [row]), [])
        prompts = render_routing_prompts(
            tokenizer=CharacterTokenizer(), case=case, config={}
        )
        self.assertEqual([prompt["name"] for prompt in prompts], ["materialized", "counterfactual"])
        validation = validate_routing_case(
            tokenizer=CharacterTokenizer(), case=case, config={}
        )
        self.assertEqual(validation["condition_count"], 2)

    def test_routing_confirmation_requires_factual_and_counterfactual_control(self):
        def condition(correct):
            return {
                "is_expected_unconstrained": correct,
                "candidate_probability_mass": 0.99,
            }

        rows = [
            {
                "format": "prose" if index % 2 else "assignments",
                "conditions": {
                    "materialized": condition(True),
                    "counterfactual": condition(True),
                },
            }
            for index in range(60)
        ]
        report = summarize_routing_rows(rows, {})
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["joint_correct"]["n"], 60)

    def test_factorization_reuses_semantics_and_separates_diagnostic_targets(self):
        config = {
            "history_families": ["add"],
            "final_family": "pointer",
            "bits": [3],
            "history_steps": [1],
            "formats": ["prose", "assignments"],
            "examples_per_cell": 2,
            "seed": 29,
        }
        cases = build_factorization_benchmark(config)
        self.assertEqual(cases, build_factorization_benchmark(config))
        self.assertEqual(len(cases), 4)
        for case in cases:
            self.assertEqual(len(set(case["diagnostic_targets"].values())), 4)
            self.assertEqual(case["state_path"][-1], case["current_state"])
        for left, right in zip(cases[:2], cases[2:]):
            self.assertEqual(left["history"], right["history"])
            self.assertEqual(left["final_rule"], right["final_rule"])
            self.assertNotEqual(left["format"], right["format"])

    def test_factorization_renders_four_assays_and_each_history_control(self):
        case = build_factorization_benchmark(
            {
                "history_families": ["xor"],
                "bits": [3],
                "history_steps": [2],
                "formats": ["assignments"],
                "examples_per_cell": 1,
                "seed": 31,
            }
        )[0]
        prompts = render_factorization_prompts(
            tokenizer=CharacterTokenizer(), case=case, config={}
        )
        self.assertEqual(
            [prompt["name"] for prompt in prompts],
            ["read", "update", "synthesize", "compose", "history_step_1", "history_step_2"],
        )
        self.assertTrue(all(prompt["text"].endswith("Answer=") for prompt in prompts))

    def test_symbolic_factorization_uses_one_surface_alphabet_end_to_end(self):
        cases = build_factorization_benchmark(
            {
                "history_families": ["pointer"],
                "final_family": "pointer",
                "bits": [3],
                "history_steps": [1],
                "formats": ["prose"],
                "examples_per_cell": 2,
                "state_representation": "symbols",
                "state_symbols": list("ABCDEFGH"),
                "balance_current_states": True,
                "seed": 41,
            }
        )
        self.assertEqual(len(cases), 2)
        self.assertEqual(len({case["current_state"] for case in cases}), 2)
        self.assertTrue(all(case["id"].startswith("pointer_to_pointer_h1_b3_symbols") for case in cases))
        prompts = render_factorization_prompts(
            tokenizer=CharacterTokenizer(), case=cases[0], config={}
        )
        self.assertIn("The state is one of [A, B, C, D, E, F, G, H]", prompts[0]["text"])
        self.assertIn("replace the current symbol according to", prompts[1]["text"])
        self.assertIn(
            f"Current state: {cases[0]['state_symbols'][cases[0]['current_state']]}",
            prompts[0]["text"],
        )

    def test_state_localization_selects_heldout_history_signal(self):
        cases = self._transfer_cases()
        indexed = {case["id"]: case for case in cases}
        split = build_transfer_split(cases, seed=47)
        captures = {}
        activations = {}
        for case in cases:
            case_id = case["id"]
            positions = [
                {"name": "start", "token_index": 2},
                {"name": "history_step_1", "token_index": 4},
                {"name": "final_rule", "token_index": 6},
                {"name": "answer", "token_index": 8},
            ]
            captures[case_id] = {"compose_positions": positions}
            trace = np.zeros((4, 3, 16), dtype=np.float32)
            for layer in range(3):
                trace[0, layer, int(case["initial_state"])] = 1
                trace[1, layer, int(case["current_state"])] = 1
                trace[2, layer, int(case["current_state"])] = 1
                trace[3, layer, int(case["initial_state"])] = 1
            materialized = np.zeros((3, 16), dtype=np.float32)
            counterfactual = np.zeros((3, 16), dtype=np.float32)
            materialized[:, int(case["current_state"])] = 1
            counterfactual[:, int(case["counterfactual_state"])] = 1
            activations[case_id] = {
                "compose": trace[-1],
                "materialized": materialized,
                "counterfactual": counterfactual,
                "compose_trace": trace,
            }
        report = analyze_state_localization(
            cases=indexed,
            split=split,
            captures=captures,
            activations=activations,
            rank=7,
            seed=53,
            gate={},
        )
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(
            report["interpretation"],
            "state_replaced_by_final_on_start_shortcut",
        )
        self.assertEqual(
            report["heldout"]["by_position"]["history_end"]["current_state"]["mean"],
            1.0,
        )

    def test_self_handoff_summary_requires_correct_state_and_shortcut_suppression(self):
        cases = self._transfer_cases(40)
        for case in cases:
            mapping = list(range(8))
            mapping[int(case["initial_state"])] = (int(case["next_state"]) + 1) % 8
            case["final_rule"] = {"kind": "pointer", "mapping": mapping}
        indexed = {case["id"]: case for case in cases}
        captures = {}
        rows = []
        for case in cases:
            correct = int(case["next_state"])
            shortcut = apply_rule(case["final_rule"], int(case["initial_state"]), 8)
            baseline = [0.0] * 8
            baseline[correct] = 0.1
            baseline[shortcut] = 0.8
            captures[case["id"]] = {
                "conditions": {
                    "compose": {"final_candidate_probabilities": baseline}
                }
            }
            conditions = {}
            for mode in ("self_state", "random_self", "full_self"):
                probabilities = list(baseline)
                if mode != "random_self":
                    probabilities[correct] = 0.8
                    probabilities[shortcut] = 0.1
                conditions[mode] = {
                    "final_candidate_probabilities": probabilities,
                    "is_expected_unconstrained": mode != "random_self",
                }
            rows.append({"id": case["id"], "layer": 2, "conditions": conditions})
        report = summarize_handoff(
            cases=indexed,
            captures=captures,
            rows=rows,
            gate={},
        )
        self.assertTrue(report["gate"]["passed"])

    def test_factorization_decision_requires_controls_and_replication(self):
        def condition(prediction, expected):
            return {
                "unconstrained_prediction": prediction,
                "is_expected_unconstrained": prediction == expected,
                "candidate_probability_mass": 0.99,
                "final_candidate_logprobabilities": [-4.0] * 8,
            }

        rows = []
        for index in range(60):
            rows.append(
                {
                    "format": "prose" if index < 30 else "assignments",
                    "history_family": "add",
                    "history_steps": 1,
                    "diagnostic_targets": {
                        "correct_composition": 6,
                        "history_only": 4,
                        "final_on_start": 2,
                        "identity": 0,
                    },
                    "conditions": {
                        "read": condition(4, 4),
                        "update": condition(6, 6),
                        "synthesize": condition(0, 4),
                        "compose": condition(2, 6),
                        "history_step_1": condition(4, 4),
                    },
                }
            )
        report = summarize_factorization_rows(rows, {})
        self.assertTrue(report["decision"]["state_synthesis_bottleneck"]["supported"])
        self.assertEqual(report["competence_admission"]["count"], 60)
        self.assertEqual(
            report["compose_diagnostics_strict_admission"]["prediction_counts"]["final_on_start"],
            60,
        )

    def test_mixed_transition_calibration_is_deterministic(self):
        config = {
            "history_families": ["add", "xor"],
            "final_family": "pointer",
            "bits": [2],
            "history_steps": [1, 3],
            "examples_per_cell": 2,
            "seed": 23,
        }
        cases = build_calibration_benchmark(config)
        self.assertEqual(cases, build_calibration_benchmark(config))
        self.assertEqual(len(cases), 8)
        self.assertEqual({case["final_rule"]["kind"] for case in cases}, {"pointer"})
        self.assertEqual(
            {rule["kind"] for case in cases for rule in case["history"]},
            {"add", "xor"},
        )
        self.assertEqual(
            cases[0],
            build_transition_case(
                history_family="add",
                final_family="pointer",
                width=2,
                example_index=0,
                seed=23,
                history_steps=1,
            )
            | {"id": "add_to_pointer_h1_b2_0000"},
        )

    def test_benchmark_is_deterministic_bijective_and_condition_matched(self):
        config = {
            "families": ["pointer", "affine", "register"],
            "bits": [2, 3],
            "examples_per_cell": 2,
            "history_steps": 2,
            "seed": 17,
        }
        first = build_benchmark(config)
        second = build_benchmark(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        for case in first:
            modulus = 2 ** case["bits"]
            outputs = {
                apply_rule(case["final_rule"], state, modulus)
                for state in range(modulus)
            }
            self.assertEqual(len(outputs), modulus)
            self.assertNotEqual(case["next_state"], case["counterfactual_next_state"])
            prompts = [
                render_prompt(case, spec)
                for spec in condition_specs(case, self_state=case["current_state"])
            ]
            self.assertEqual(len({len(prompt.text) for prompt in prompts}), 1)
            self.assertEqual(
                len({(prompt.checkpoint_start, prompt.checkpoint_end) for prompt in prompts}),
                1,
            )

    def test_settling_depth_uses_dtr_running_minimum(self):
        divergences = np.asarray([0.9, 0.4, 0.8, 0.1])
        self.assertEqual(settling_depth(divergences, threshold=0.5), 1)
        self.assertEqual(settling_depth(np.asarray([0.9, 0.8, 0.0]), threshold=0.5), 2)

    def test_qualification_program_is_deterministic_and_register_matched(self):
        config = {
            "bits": [2],
            "history_steps": [1, 4],
            "examples_per_cell": 2,
            "seed": 19,
        }
        first = build_qualification_benchmark(config)
        self.assertEqual(first, build_qualification_benchmark(config))
        self.assertEqual(len(first), 4)
        self.assertEqual(
            {case["history_steps"] for case in first},
            {1, 4},
        )
        for case in first:
            specs = qualification_condition_specs(case)
            direct = render_qualification_direct_prompt(case)
            self.assertTrue(direct.endswith("Answer="))
            prompts = [
                render_qualification_prompt(case, spec)
                for spec in specs
                if spec["name"] != "direct"
            ]
            self.assertEqual(len({len(prompt.text) for prompt in prompts}), 1)
            self.assertEqual(
                len(
                    {
                        (prompt.checkpoint_start, prompt.checkpoint_end)
                        for prompt in prompts
                    }
                ),
                1,
            )

    def test_qualification_gate_requires_behavior_before_depth(self):
        def condition():
            return {
                "unconstrained_prediction": 2,
                "is_expected_unconstrained": True,
                "candidate_probability_mass": 0.99,
                "final_candidate_probabilities": [0.01, 0.01, 0.97, 0.01],
            }

        rows = [
            {
                "bits": 2,
                "history_steps": 1 + index % 2,
                "next_state": 2,
                "counterfactual_next_state": 2,
                "conditions": {
                    name: condition()
                    for name in (
                        "direct",
                        "none",
                        "gold",
                        "counterfactual",
                        "invalid",
                    )
                },
            }
            for index in range(60)
        ]
        report = summarize_qualification_rows(rows, {})
        self.assertTrue(report["gate"]["passed"])
        self.assertEqual(report["joint_none_gold_unconstrained"]["n"], 60)

    def test_invalid_invariance_is_not_conditioned_on_accuracy(self):
        def condition(prediction, expected):
            return {
                "unconstrained_prediction": prediction,
                "is_expected_unconstrained": prediction == expected,
                "candidate_probability_mass": 0.99,
                "final_candidate_probabilities": [0.7, 0.1, 0.1, 0.1],
            }

        rows = [
            {
                "bits": 2,
                "history_steps": 1,
                "next_state": 2,
                "counterfactual_next_state": 3,
                "conditions": {
                    "direct": condition(2, 2),
                    "none": condition(0, 2),
                    "gold": condition(2, 2),
                    "counterfactual": condition(3, 3),
                    "invalid": condition(0, 2),
                },
            }
            for _ in range(8)
        ]

        report = summarize_qualification_rows(rows, {})
        self.assertEqual(
            report["invalid_register"]["prediction_invariance"]["mean"],
            1.0,
        )
        self.assertEqual(
            report["invalid_register"]["joint_correct_prediction_invariance"][
                "mean"
            ],
            0.0,
        )

    def test_frontier_selection_requires_behavior_but_never_authorizes_depth(self):
        def condition(prediction, expected):
            return {
                "unconstrained_prediction": prediction,
                "is_expected_unconstrained": prediction == expected,
                "candidate_probability_mass": 0.99,
            }

        rows = [
            {
                "history_family": "add",
                "final_family": "pointer",
                "bits": 2,
                "history_steps": 1,
                "conditions": {
                    "direct": condition(2, 2),
                    "none": condition(2, 2),
                    "none_alt": condition(2, 2),
                    "gold": condition(2, 2),
                    "counterfactual": condition(3, 3),
                },
            }
            for _ in range(12)
        ]
        report = summarize_calibration_rows(rows, {})
        self.assertEqual(report["eligible_cell_count"], 1)
        self.assertEqual(report["next_stage"], "held_out_confirmation")
        self.assertFalse(report["depth_capture_authorized"])

    def test_summary_keeps_matched_accuracy_and_counterfactual_controls_separate(self):
        def condition(depth, prediction, expected, revealed=None, area=None):
            return {
                "settling_depth": depth,
                "settling_depth_by_threshold": {"0.5": depth},
                "dtr_jsd_auc": area if area is not None else depth / 10,
                "final_candidate_probabilities": [0.1, 0.2, 0.6, 0.1],
                "prediction": prediction,
                "is_expected_unconstrained": prediction == expected,
                "expected_next_state": expected,
                "is_expected": prediction == expected,
                "revealed_bits": revealed,
            }

        row = {
            "family": "affine",
            "next_state": 2,
            "writer": {
                "correct_top1_at_any_layer": True,
                "is_correct": True,
                "is_correct_unconstrained": True,
            },
            "conditions": {
                "none": condition(8, 2, 2, 0, 0.8),
                "partial_1": condition(7, 2, 2, 1, 0.6),
                "gold": condition(5, 2, 2, 2, 0.4),
                "self": condition(5, 2, 2, 2),
                "counterfactual": condition(4, 3, 3),
            },
            "causal": None,
        }
        report = summarize_rows([row])
        self.assertEqual(report["depth_relief"]["matched_accuracy"]["mean"], 3.0)
        self.assertEqual(report["depth_relief"]["by_threshold"]["0.5"]["all"]["mean"], 3.0)
        self.assertEqual(report["counterfactual_rule_consistent_rate"], 1.0)
        self.assertLess(report["dose_response"]["linear_slope"], 0.0)
        self.assertLess(report["dose_response"]["jsd_curve_area_linear_slope"], 0.0)
        self.assertEqual(report["validity"]["matched_none_gold_unconstrained"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
