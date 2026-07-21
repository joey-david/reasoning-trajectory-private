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
from src.experiments.depth_relief.state_handoff_evaluation import (
    build_code_donors,
)
from src.experiments.depth_relief.state_handoff_training import (
    read_training_metrics,
)
from src.experiments.depth_relief.state_handoff_smoke import run_tiny_smoke


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


def test_duplicate_resumed_metrics_are_rejected(tmp_path) -> None:
    path = tmp_path / "training/outcome_only/metrics.jsonl"
    path.parent.mkdir(parents=True)
    row = {"optimizer_step": 1}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate optimizer steps"):
        read_training_metrics(tmp_path, "outcome_only")


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
