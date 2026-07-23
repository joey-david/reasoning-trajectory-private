from __future__ import annotations

import json

import pytest

from src.experiments.depth_relief.state_handoff_data import build_test_programs
from src.experiments.depth_relief.state_handoff_training import _base_and_adapter
from src.experiments.depth_relief.state_interface_data import (
    interface_training_sequence_pair,
)
from src.experiments.depth_relief.state_interface_equivalence import (
    _consumer_table,
)
from src.experiments.depth_relief.state_interface_stress import (
    prepare_stress_profile,
    read_stress_programs,
)


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def test_stress_histories_are_deterministic_balanced_and_exact(tmp_path) -> None:
    run_path = tmp_path / "stress"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_stress:
  conditions:
    decimal:
      kind: decimal
      source_run: runs/source
      source_condition: explicit_handoff
  profiles:
    probe:
      bits: 3
      seed: 17
      context_count: 2
      paths_per_state: 2
      horizons: [2, 4]
      families: [structured, iid, shuffled, cancellation, repeated]
""".strip()
        + "\n"
    )
    first = prepare_stress_profile(run_path, "probe")
    rows = read_stress_programs(run_path, "probe")
    second = prepare_stress_profile(run_path, "probe")
    assert first == second
    assert len(rows) == 2 * 2 * 8 * 2 * 5
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["state_path"][-1] == row["current_state"] for row in rows)
    cells = {
        (row["stress_family"], row["history_steps"], row["current_state"])
        for row in rows
    }
    assert len(cells) == 5 * 2 * 8


def test_interface_producer_mode_selects_the_requested_prompt() -> None:
    case = build_test_programs(
        horizons=(2,),
        context_count=1,
        paths_per_state=1,
        width=3,
        seed=19,
    )[0]
    tokenizer = CharacterTokenizer()
    common = {
        "tokenizer": tokenizer,
        "case": case,
        "prompt_config": {"mode": "plain"},
        "condition": "canonical_opaque",
        "interface_config": {},
        "max_length": 512,
    }
    encoder = interface_training_sequence_pair(**common, producer_mode="encoder")
    transition = interface_training_sequence_pair(
        **common, producer_mode="transition"
    )
    assert encoder[0]["producer_prompt_kind"] == "encoder"
    assert transition[0]["producer_prompt_kind"] == "transition"
    assert encoder[0]["input_ids"] != transition[0]["input_ids"]


def test_consumer_table_rejects_context_code_disagreement() -> None:
    rows = [
        {
            "program_context": "c0",
            "true_code": 1,
            "gold_final": {"unconstrained_prediction": 3},
        },
        {
            "program_context": "c0",
            "true_code": 1,
            "gold_final": {"unconstrained_prediction": 4},
        },
    ]
    try:
        _consumer_table(rows)
    except ValueError as error:
        assert "not deterministic" in str(error)
    else:
        raise AssertionError("Conflicting consumer calls were accepted")


def test_stress_manifest_hash_matches_saved_programs(tmp_path) -> None:
    run_path = tmp_path / "stress"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_stress:
  conditions:
    decimal:
      kind: decimal
      source_run: runs/source
      source_condition: explicit_handoff
  profiles:
    probe:
      bits: 3
      seed: 23
      context_count: 1
      paths_per_state: 1
      horizons: [2]
      families: [iid]
""".strip()
        + "\n"
    )
    manifest = prepare_stress_profile(run_path, "probe")
    saved = json.loads(
        (run_path / "evaluation/stress/probe/manifest.json").read_text()
    )
    assert saved == manifest


def test_training_can_warm_start_from_an_existing_adapter(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    source = peft.get_peft_model(
        transformers.LlamaForCausalLM(config),
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj"],
        ),
    )
    for parameter in source.parameters():
        if parameter.requires_grad:
            parameter.data.fill_(0.125)
    adapter_path = tmp_path / "source_adapter"
    source.save_pretrained(adapter_path, safe_serialization=True)

    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        f"""
state_handoff_training:
  interfaces:
    initial_adapters:
      canonical_opaque: {adapter_path}
""".strip()
        + "\n"
    )
    loaded, tokenizer = _base_and_adapter(
        run_path=run_path,
        condition="canonical_opaque",
        checkpoint=None,
        model=transformers.LlamaForCausalLM(config),
        tokenizer=object(),
    )
    trainable = [
        parameter.detach()
        for parameter in loaded.parameters()
        if parameter.requires_grad
    ]
    assert tokenizer is not None
    assert trainable
    assert all(torch.allclose(value, torch.full_like(value, 0.125)) for value in trainable)
