from __future__ import annotations

import json

import pytest

from src.experiments.depth_relief.state_handoff_data import (
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_PATH,
    matched_compute_manifest,
    prepare_state_handoff_datasets,
    read_programs,
    training_sequence_pair,
)
from src.experiments.depth_relief.state_handoff_continuation import (
    block_case,
    continuation_case_id,
    prepare_continuation_programs,
    read_continuation_programs,
    summarize_continuation_rows,
)
from src.experiments.depth_relief.state_handoff_evaluation import (
    _frozen_screen_summary,
    build_code_donors,
)
from src.experiments.depth_relief.state_handoff_training import (
    _flush_checkpoint_metrics,
    read_training_metrics,
)
from src.experiments.depth_relief.state_handoff_information import (
    conditional_entropy,
    conditional_mutual_information,
    discrete_entropy,
    mutual_information,
    rate_capacity_table,
    summarize_code_information,
)
from src.experiments.depth_relief.state_interface_data import (
    CODEBOOK_SIZES,
    build_interface_training_pairs,
    interface_code_index,
    interface_training_sequence_pair,
    matched_interface_compute_manifest,
    semantic_states_for_code,
)
from src.experiments.depth_relief.state_handoff_smoke import (
    run_tiny_interface_smoke,
    run_tiny_smoke,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def _write_config(run_path) -> None:
    run_path.mkdir(parents=True)
    (run_path / "config.yaml").write_text(
        """
state_handoff_training:
  dataset:
    bits: 3
    seed: 101
    train_examples: 128
    validation_examples: 64
    train_horizons: [1, 2]
    validation_horizons: [1, 2]
    test_horizons: [2, 4, 8]
    train_program_contexts: 2
    validation_program_contexts: 2
    test_program_contexts: 3
    test_paths_per_state: 2
""".strip()
        + "\n"
    )


def test_training_splits_are_group_disjoint_and_deterministic(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    first = prepare_state_handoff_datasets(run_path)
    second = prepare_state_handoff_datasets(run_path)
    assert first == second
    contexts = {
        "train": {row["program_context"] for row in read_programs(run_path / TRAIN_PATH)},
        "validation": {
            row["program_context"] for row in read_programs(run_path / VALIDATION_PATH)
        },
        "test": {row["program_context"] for row in read_programs(run_path / TEST_PATH)},
    }
    assert contexts["train"].isdisjoint(contexts["validation"])
    assert contexts["train"].isdisjoint(contexts["test"])
    assert contexts["validation"].isdisjoint(contexts["test"])


def test_training_sequences_mask_every_prompt_token(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    case = read_programs(run_path / TRAIN_PATH)[0]
    pair = training_sequence_pair(
        tokenizer=CharacterTokenizer(),
        case=case,
        prompt_config={"prompt": {"mode": "plain"}},
        condition="explicit_handoff",
        max_length=1000,
    )
    assert [row["mapping"] for row in pair] == ["state", "answer"]
    assert all(sum(label != -100 for label in row["labels"]) == 1 for row in pair)
    assert all(
        next(label for label in row["labels"] if label != -100)
        == row["target_token_id"]
        for row in pair
    )


def test_outcome_and_handoff_training_compute_is_matched(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    cases = read_programs(run_path / TRAIN_PATH)[:8]
    manifest = matched_compute_manifest(
        tokenizer=CharacterTokenizer(),
        cases=cases,
        prompt_config={"prompt": {"mode": "plain"}},
        max_length=1000,
    )
    assert manifest["matched_forward_passes_and_tokens"] is True
    outcome = manifest["conditions"]["outcome_only"]
    handoff = manifest["conditions"]["explicit_handoff"]
    assert outcome["forward_passes"] == handoff["forward_passes"] == 16
    assert outcome["fixed_padding_compute_tokens"] == handoff[
        "fixed_padding_compute_tokens"
    ]


def test_same_and_different_state_donors_obey_contract(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    cases = read_programs(run_path / TEST_PATH)
    indexed = {row["id"]: row for row in cases}
    for donor in build_code_donors(cases)[:20]:
        recipient = indexed[donor["recipient_id"]]
        same = indexed[donor["same_state_donor_id"]]
        different = indexed[donor["different_state_donor_id"]]
        assert same["current_state"] == recipient["current_state"]
        assert same["path_code"] != recipient["path_code"]
        assert different["path_code"] == recipient["path_code"]
        assert different["current_state"] == (recipient["current_state"] + 1) % 8


def test_frozen_screen_summary_reuses_saved_analysis(tmp_path) -> None:
    run_path = tmp_path / "run"
    handoff_path = run_path / "depth_relief/explicit_handoff"
    handoff_path.mkdir(parents=True)
    accuracy = {"n": 8, "mean": 0.75, "ci95": [0.5, 1.0]}
    factorization = {
        "controls": {
            "read": accuracy,
            "update": accuracy,
            "constituent_steps": accuracy,
        },
        "by_history": {
            "2": {
                "accuracy": {
                    "compose": accuracy,
                    "synthesize": accuracy,
                }
            }
        },
    }
    handoff = {
        "by_horizon": {
            "2": {
                "conditions": {
                    condition: {"accuracy": accuracy}
                    for condition in (
                        "gold_handoff",
                        "lm_self_handoff",
                        "stepwise_explicit",
                    )
                }
            }
        }
    }
    (run_path / "depth_relief/factorization_summary.json").write_text(
        json.dumps(factorization)
    )
    (handoff_path / "summary.json").write_text(json.dumps(handoff))

    summary = _frozen_screen_summary(run_path)

    assert summary["read_accuracy"] == accuracy
    assert summary["by_horizon"]["2"] == {
        "one_pass_compose_accuracy": accuracy,
        "synthesize_accuracy": accuracy,
        "gold_handoff_accuracy": accuracy,
        "self_handoff_accuracy": accuracy,
        "stepwise_handoff_accuracy": accuracy,
    }


def test_duplicate_resumed_metrics_are_rejected(tmp_path) -> None:
    path = tmp_path / "training/outcome_only/metrics.jsonl"
    path.parent.mkdir(parents=True)
    row = {"optimizer_step": 1}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate optimizer steps"):
        read_training_metrics(tmp_path, "outcome_only")


def test_checkpoint_metrics_recover_partial_flush_without_duplicates(tmp_path) -> None:
    metrics = [
        {"optimizer_step": 1, "total_loss": 2.0},
        {"optimizer_step": 2, "total_loss": 1.0},
    ]
    path = tmp_path / "training/outcome_only/metrics.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(metrics[0]) + "\n")

    recovered = _flush_checkpoint_metrics(tmp_path, "outcome_only", metrics)
    repeated = _flush_checkpoint_metrics(tmp_path, "outcome_only", metrics)

    assert recovered == repeated == metrics
    assert read_training_metrics(tmp_path, "outcome_only") == metrics
    with pytest.raises(ValueError, match="changed at optimizer step 2"):
        _flush_checkpoint_metrics(
            tmp_path,
            "outcome_only",
            [{"optimizer_step": 2, "total_loss": 9.0}],
        )


def test_tiny_lora_resume_save_reload_and_evaluation_smoke(tmp_path) -> None:
    result = run_tiny_smoke(tmp_path)
    assert result == {
        "one_optimizer_step_completed": True,
        "finite_losses": True,
        "adapter_roundtrip_preserved_predictions": True,
        "evaluation_without_training_dataset": True,
        "resume_metric_steps": [1, 2],
        "duplicate_metrics_or_cases": False,
    }


def test_continuation_programs_do_not_replace_pilot_test_data(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    original = (run_path / TEST_PATH).read_bytes()

    manifest = prepare_continuation_programs(run_path, "probe")
    cases = read_continuation_programs(run_path, "probe")

    assert (run_path / TEST_PATH).read_bytes() == original
    assert manifest["horizons"] == [2, 4, 8, 16]
    assert manifest["case_count"] == 10 * 4 * 8 * 2
    assert len(cases) == manifest["case_count"]


def test_recursive_blocks_accept_the_previous_block_state(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    case = next(
        row
        for row in read_programs(run_path / TEST_PATH)
        if row["history_steps"] == 4
    )
    first = block_case(
        case, input_state=case["initial_state"], start=0, block_size=2
    )
    second = block_case(
        case, input_state=first["current_state"], start=2, block_size=2
    )

    assert first["current_state"] == case["state_path"][2]
    assert second["current_state"] == case["current_state"]
    assert continuation_case_id(case, 2).endswith("__block2")


def test_continuation_summary_separates_local_closure_from_global_state() -> None:
    rows = []
    for path_code, final_state in enumerate((3, 3)):
        rows.append(
            {
                "id": f"case_{path_code}",
                "program_context": "context",
                "history_steps": 4,
                "block_size": 2,
                "current_state": 3,
                "predicted_state": final_state,
                "state_correct": True,
                "predicted_steps": [
                    {"locally_correct": True},
                    {"locally_correct": True},
                ],
                "final": {
                    "is_expected_unconstrained": True,
                    "follows_supplied_state": True,
                },
            }
        )

    summary = summarize_continuation_rows(rows)["by_cell"]["block2_h4"]

    assert summary["state_accuracy"]["mean"] == 1.0
    assert summary["local_closure_accuracy"]["mean"] == 1.0
    assert summary["same_state_code_agreement"]["mean"] == 1.0


def test_exact_information_metrics_separate_state_and_path_bits() -> None:
    rows = []
    for state in range(8):
        for path in range(2):
            rows.append(
                {
                    "current_state": state,
                    "conditions": {
                        "state": {"unconstrained_prediction": state}
                    },
                    "path_code": path,
                }
            )
    summary = summarize_code_information(rows)

    assert discrete_entropy(range(8)) == 3.0
    assert conditional_entropy((state, state) for state in range(8)) == 0.0
    assert mutual_information((state, state) for state in range(8)) == 3.0
    assert summary["state_information_bits"] == 3.0
    assert summary["code_given_state_bits"] == 0.0
    assert summary["path_invariance_exact"] is True
    assert conditional_mutual_information(
        (state, state, "context") for state in range(8)
    ) == 3.0


def test_rate_table_has_exact_three_bit_threshold() -> None:
    table = {row["codebook_size"]: row for row in rate_capacity_table()}

    assert table[4]["deterministic_balanced_state_accuracy_ceiling"] == 0.5
    assert table[4]["lossless_possible"] is False
    assert table[8]["capacity_bits"] == 3.0
    assert table[8]["lossless_possible"] is True
    assert table[16]["deterministic_balanced_state_accuracy_ceiling"] == 1.0


def test_interface_code_contracts_have_expected_rate_and_invariance(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    case = read_programs(run_path / TRAIN_PATH)[0]
    config = {}

    canonical = [
        interface_code_index(
            condition="canonical_opaque",
            case=case,
            state=state,
            interface_config=config,
        )
        for state in range(8)
    ]
    compressed = semantic_states_for_code(
        condition="compressed_2bit",
        case=case,
        code_index=0,
        interface_config=config,
    )
    redundant = {
        interface_code_index(
            condition="redundant_4bit",
            case=case,
            state=3,
            variant=variant,
            interface_config=config,
        )
        for variant in (0, 1)
    }

    assert sorted(canonical) == list(range(8))
    assert compressed == (0, 4)
    assert redundant == {6, 7}
    assert CODEBOOK_SIZES == {
        "canonical_opaque": 8,
        "context_bound": 8,
        "compressed_2bit": 4,
        "redundant_4bit": 16,
    }


def test_interface_training_pairs_mask_prompts_and_match_compute(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    cases = read_programs(run_path / TRAIN_PATH)
    tokenizer = CharacterTokenizer()
    pair = interface_training_sequence_pair(
        tokenizer=tokenizer,
        case=cases[0],
        prompt_config={"mode": "plain"},
        condition="canonical_opaque",
        interface_config={"independent_module_contexts": False},
        max_length=1000,
    )
    manifest = matched_interface_compute_manifest(
        tokenizer=tokenizer,
        cases=cases[:8],
        prompt_config={"mode": "plain"},
        conditions=(
            "canonical_opaque",
            "context_bound",
            "compressed_2bit",
            "redundant_4bit",
        ),
        interface_config={"independent_module_contexts": False},
        max_length=1000,
    )

    assert [row["mapping"] for row in pair] == ["state", "answer"]
    assert all(sum(label != -100 for label in row["labels"]) == 1 for row in pair)
    assert all(len(row["input_ids"]) == 1000 for row in pair)
    assert manifest["matched_forward_passes_and_tokens"] is True
    assert {
        values["fixed_padding_compute_tokens"]
        for values in manifest["conditions"].values()
    } == {16_000}


def test_interface_producers_and_consumers_use_disjoint_contexts(tmp_path) -> None:
    run_path = tmp_path / "run"
    _write_config(run_path)
    prepare_state_handoff_datasets(run_path)
    cases = read_programs(run_path / TRAIN_PATH)
    pairs = build_interface_training_pairs(
        tokenizer=CharacterTokenizer(),
        cases=cases,
        prompt_config={"mode": "plain"},
        condition="canonical_opaque",
        interface_config={"independent_module_contexts": True},
        max_length=1000,
    )
    producer_contexts = {pair[0]["program_context"] for pair in pairs}
    consumer_contexts = {pair[1]["program_context"] for pair in pairs}

    assert producer_contexts.isdisjoint(consumer_contexts)
    assert len(pairs) == len(cases)


def test_tiny_opaque_interface_training_and_evaluation_smoke(tmp_path) -> None:
    result = run_tiny_interface_smoke(tmp_path)

    assert result == {
        "one_optimizer_step_completed": True,
        "finite_losses": True,
        "evaluation_without_training_dataset": True,
        "matched_interface_compute": True,
    }
