from __future__ import annotations

import json

import pytest

from src.experiments.depth_relief.state_handoff_data import build_test_programs
from src.experiments.depth_relief.benchmark import apply_rule
from src.experiments.depth_relief.state_handoff_training import _base_and_adapter
from src.experiments.depth_relief.state_interface_data import (
    interface_training_sequence_pair,
)
from src.experiments.depth_relief.state_interface_contract import (
    interface_code_index,
    semantic_states_for_code,
)
from src.experiments.depth_relief.state_interface_equivalence import (
    _consumer_table,
)
from src.experiments.depth_relief.state_interface_stress import (
    prepare_stress_profile,
    read_stress_programs,
)
from src.orchestration.jobs.state_handoff_training import pending_tasks


class CharacterTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def test_linked_training_runs_share_one_dynamic_gpu_queue(tmp_path) -> None:
    interface = tmp_path / "interface"
    outcome = tmp_path / "outcome"
    interface.mkdir()
    outcome.mkdir()
    (interface / "config.yaml").write_text(
        "state_handoff_training:\n"
        "  conditions: [canonical_opaque]\n"
        "  linked_runs:\n"
        f"    - {outcome}\n"
    )
    (outcome / "config.yaml").write_text(
        "state_handoff_training:\n  conditions: [outcome_only]\n"
    )
    tasks, total, complete = pending_tasks(interface)
    assert total == 2
    assert complete == 0
    assert tasks == [
        {"run_path": str(interface), "condition": "canonical_opaque"},
        {"run_path": str(outcome), "condition": "outcome_only"},
    ]


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


def test_mixed_replay_fraction_is_deterministic_and_preserves_both_calls() -> None:
    cases = build_test_programs(
        horizons=(2,),
        context_count=8,
        paths_per_state=2,
        width=3,
        seed=29,
    )
    tokenizer = CharacterTokenizer()

    def prompt_kinds() -> list[str]:
        return [
            interface_training_sequence_pair(
                tokenizer=tokenizer,
                case=case,
                prompt_config={"mode": "plain"},
                condition="canonical_opaque",
                interface_config={},
                max_length=512,
                producer_mode="mixed",
                transition_fraction=0.75,
            )[0]["producer_prompt_kind"]
            for case in cases
        ]

    first = prompt_kinds()
    assert first == prompt_kinds()
    transition_rate = first.count("transition") / len(first)
    assert 0.65 <= transition_rate <= 0.85
    assert set(first) == {"encoder", "transition"}


def test_mixed_algebra_holds_out_ordered_pairs_without_changing_endpoints() -> None:
    cases = build_test_programs(
        horizons=(2, 4),
        context_count=2,
        paths_per_state=1,
        width=3,
        seed=31,
        dataset={
            "domain": "mixed_algebra",
            "test_composition_splits": ["seen", "heldout"],
        },
    )
    seen = {
        pair
        for case in cases
        if case["composition_split"] == "seen"
        for pair in case["operation_pairs"]
    }
    heldout = {
        pair
        for case in cases
        if case["composition_split"] == "heldout"
        for pair in case["operation_pairs"]
    }
    assert seen.isdisjoint(heldout)
    assert len(cases) == 2 * 2 * 8 * 1 * 2
    assert all(case["state_path"][-1] == case["current_state"] for case in cases)
    assert {case["domain"] for case in cases} == {"mixed_algebra"}


def test_reasoning_mixture_has_exact_program_and_proof_state_paths() -> None:
    cases = build_test_programs(
        horizons=(4,),
        context_count=4,
        paths_per_state=2,
        width=4,
        seed=37,
        dataset={
            "domain": "reasoning_mixture",
            "test_composition_splits": ["seen", "heldout"],
        },
    )
    assert {case["domain"] for case in cases} == {
        "mixed_algebra",
        "horn_proof",
    }
    assert all(case["state_path"][-1] == case["current_state"] for case in cases)
    proof = [case for case in cases if case["domain"] == "horn_proof"]
    assert all(case["final_rule"]["kind"] == "proof_query" for case in proof)
    seen = [case for case in proof if case["composition_split"] == "seen"]
    heldout = [case for case in proof if case["composition_split"] == "heldout"]
    assert all(not case["proof_composition_active"] for case in seen)
    assert sum(case["proof_composition_active"] for case in heldout) == (
        5 * 2 * 2
    )
    assert all(
        any(
            len(rule["premises"]) == 2
            and all(state & (1 << bit) for bit in rule["premises"])
            and apply_rule(rule, state, 16) != state
            for state, rule in zip(case["state_path"], case["history"])
        )
        for case in heldout
        if case["proof_composition_active"]
    )
    assert all(
        apply_rule(case["final_rule"], case["current_state"], 16)
        == case["next_state"]
        for case in proof
    )


def test_four_bit_rate_contracts_have_expected_fibers() -> None:
    case = build_test_programs(
        horizons=(2,),
        context_count=1,
        paths_per_state=1,
        width=4,
        seed=41,
        dataset={"domain": "mixed_algebra"},
    )[0]
    canonical = {
        interface_code_index(
            condition="canonical_4bit",
            case=case,
            state=state,
            interface_config={},
        )
        for state in range(16)
    }
    assert canonical == set(range(16))
    assert semantic_states_for_code(
        condition="compressed_3bit",
        case=case,
        code_index=3,
        interface_config={},
    ) == (3, 11)
    assert {
        interface_code_index(
            condition="redundant_5bit",
            case=case,
            state=9,
            interface_config={},
            variant=variant,
        )
        for variant in (0, 1)
    } == {18, 19}


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
