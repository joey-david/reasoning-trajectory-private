from __future__ import annotations

import json

import pytest

from src.experiments.depth_relief.state_handoff_data import (
    TRAIN_PATH,
    build_test_programs,
    prepare_state_handoff_datasets,
    read_programs,
)
from src.experiments.depth_relief.benchmark import answer_symbols, apply_rule
from src.experiments.depth_relief.state_handoff_training import _base_and_adapter
from src.experiments.depth_relief.state_interface_data import (
    interface_training_sequence_pair,
    render_interface_encoder_prompt,
    validate_state_interface_training_data,
)
from src.experiments.depth_relief.factorization import render_factorization_prompts
from src.experiments.depth_relief.state_interface_contract import (
    interface_codebook_size,
    interface_code_index,
    semantic_states_for_code,
)
from src.experiments.depth_relief.state_handoff_challenge_programs import (
    build_full_support_proof_programs,
)
from src.experiments.depth_relief.state_interface_challenge import (
    _write_summary,
    configured_challenge_profiles,
    prepare_interface_challenges,
)
from src.experiments.depth_relief.state_interface_substitution import (
    _balanced_prefix,
)
from src.experiments.depth_relief.state_interface_equivalence import (
    _consumer_table,
)
from src.experiments.depth_relief.state_interface_stress import (
    prepare_stress_profile,
    read_stress_programs,
)
from src.orchestration.jobs.state_handoff_training import (
    linked_training_status,
    pending_tasks,
)


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
    assert linked_training_status(interface) == {
        "run_path": str(interface),
        "total": 2,
        "complete": 0,
        "pending_count": 2,
        "pending": tasks,
    }

    for run_path, condition, evaluation_path in (
        (
            interface,
            "canonical_opaque",
            interface / "evaluation/interfaces/canonical_opaque/summary.json",
        ),
        (
            outcome,
            "outcome_only",
            outcome / "evaluation/outcome_only/summary.json",
        ),
    ):
        manifest_path = (
            run_path / "training" / condition / "checkpoint_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text('{"status": "complete"}\n')
        evaluation_path.parent.mkdir(parents=True)
        evaluation_path.write_text('{"complete": true}\n')

    assert linked_training_status(interface) == {
        "run_path": str(interface),
        "total": 2,
        "complete": 2,
        "pending_count": 0,
        "pending": [],
    }


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
    transition = interface_training_sequence_pair(**common, producer_mode="transition")
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
    assert sum(case["proof_composition_active"] for case in heldout) == (5 * 2 * 2)
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
        apply_rule(case["final_rule"], case["current_state"], 16) == case["next_state"]
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


def test_padded_five_bit_code_has_no_alias_entropy() -> None:
    case = build_test_programs(
        horizons=(2,),
        context_count=1,
        paths_per_state=1,
        width=4,
        seed=415,
        dataset={"domain": "horn_proof"},
    )[0]
    indices = [
        interface_code_index(
            condition="padded_5bit",
            case=case,
            state=state,
            interface_config={},
        )
        for state in range(16)
    ]
    assert len(set(indices)) == 16
    assert indices == list(range(0, 32, 2))
    assert all(
        semantic_states_for_code(
            condition="padded_5bit",
            case=case,
            code_index=index,
            interface_config={},
        )
        == (state,)
        for state, index in enumerate(indices)
    )
    assert all(
        semantic_states_for_code(
            condition="padded_5bit",
            case=case,
            code_index=index,
            interface_config={},
        )
        == ()
        for index in range(1, 32, 2)
    )
    assert all(
        interface_code_index(
            condition="padded_5bit",
            case=case,
            state=state,
            interface_config={},
        )
        == interface_code_index(
            condition="redundant_5bit",
            case=case,
            state=state,
            interface_config={},
            variant=0,
        )
        for state in range(16)
    )


def test_full_support_proofs_cover_and_separate_all_states() -> None:
    rows = build_full_support_proof_programs(
        horizon=8,
        context_count=8,
        width=4,
        seed=417,
        split="full_support",
    )
    assert len(rows) == 8 * 16
    assert {row["current_state"] for row in rows} == set(range(16))
    assert {row["path_code"] for row in rows} == {0, 1}
    assert {row["proof_query_bit"] for row in rows} == {0, 1, 2, 3}
    assert all(row["state_path"][-1] == row["current_state"] for row in rows)
    assert all(
        row["active_transition_count"] == row["current_state"].bit_count()
        for row in rows
    )
    for query_bit in range(4):
        selected = [row for row in rows if row["proof_query_bit"] == query_bit]
        assert sum(row["next_state"] for row in selected) == len(selected) // 2
        assert {row["path_code"] for row in selected} == {0, 1}
        for state in range(16):
            matched = [row for row in selected if row["current_state"] == state]
            assert len(matched) == 2
            assert matched[0]["history"] != matched[1]["history"]


def test_challenge_preparation_routes_full_state_support(tmp_path) -> None:
    run_path = tmp_path / "full"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  all_states:
    interface_run: runs/interface
    interface_condition: canonical_4bit
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: query
    bits: 4
    seed: 419
    horizons: [8]
    full_state_support: true
    program_contexts: 8
    block_size: 1
""".strip()
        + "\n"
    )
    manifest = prepare_interface_challenges(run_path)
    profile = manifest["profiles"]["all_states"]
    assert profile["case_count"] == 128
    assert profile["states"] == list(range(16))
    assert profile["active_depths"] == [0, 1, 2, 3, 4]


def test_configured_rate_sweep_has_exact_information_fibers() -> None:
    case = build_test_programs(
        horizons=(2,),
        context_count=1,
        paths_per_state=1,
        width=3,
        seed=42,
    )[0]
    assert [
        interface_codebook_size(condition, {})
        for condition in ("rate_4", "rate_8", "rate_16", "rate_32")
    ] == [4, 8, 16, 32]
    assert semantic_states_for_code(
        condition="rate_4",
        case=case,
        code_index=3,
        interface_config={},
    ) == (3, 7)
    assert semantic_states_for_code(
        condition="rate_8",
        case=case,
        code_index=3,
        interface_config={},
    ) == (3,)
    assert {
        interface_code_index(
            condition="rate_32",
            case=case,
            state=3,
            interface_config={},
            variant=variant,
        )
        for variant in range(4)
    } == {3, 11, 19, 27}


def test_primitive_algebra_learns_locally_and_tests_unseen_orders() -> None:
    cases = build_test_programs(
        horizons=(1, 8),
        context_count=3,
        paths_per_state=2,
        width=3,
        seed=44,
        dataset={
            "domain": "algebra_primitives",
            "test_composition_splits": ["seen", "heldout"],
        },
    )
    one_step = [case for case in cases if case["history_steps"] == 1]
    assert {case["history"][0]["kind"] for case in one_step} == {"add", "xor", "affine"}
    heldout = [
        case
        for case in cases
        if case["history_steps"] == 8 and case["composition_split"] == "heldout"
    ]
    assert all(len(set(case["operation_pairs"])) > 1 for case in heldout)
    assert all(case["state_path"][-1] == case["current_state"] for case in cases)


def test_proof_actions_expose_the_full_fact_ledger() -> None:
    cases = build_test_programs(
        horizons=(8,),
        context_count=2,
        paths_per_state=1,
        width=4,
        seed=46,
        dataset={
            "domain": "horn_proof",
            "proof_final": "action",
            "test_composition_splits": ["heldout"],
        },
    )
    assert all(case["final_rule"]["kind"] == "proof_action" for case in cases)
    assert all(len(answer_symbols(case)) == 16 for case in cases)
    assert all(
        case["next_state"] == apply_rule(case["final_rule"], case["current_state"], 16)
        for case in cases
    )
    assert any(case["proof_composition_active"] for case in cases)


def test_register_machine_programs_are_balanced_and_exact() -> None:
    cases = build_test_programs(
        horizons=(1, 16),
        context_count=2,
        paths_per_state=2,
        width=4,
        seed=48,
        dataset={
            "domain": "register_machine",
            "test_composition_splits": ["seen", "heldout"],
        },
    )
    assert {case["current_state"] for case in cases} == set(range(16))
    assert all(case["state_path"][-1] == case["current_state"] for case in cases)
    heldout = [
        case
        for case in cases
        if case["history_steps"] == 16 and case["composition_split"] == "heldout"
    ]
    assert all(len(set(case["instruction_families"])) == 4 for case in heldout)


def test_register_entry_prompts_expose_both_register_values() -> None:
    case = build_test_programs(
        horizons=(2,),
        context_count=1,
        paths_per_state=1,
        width=4,
        seed=49,
        dataset={
            "domain": "register_machine",
            "state_symbols": list("0123456789ABCDEF"),
        },
    )[0]
    tokenizer = CharacterTokenizer()
    encoder = render_interface_encoder_prompt(
        tokenizer=tokenizer,
        case=case,
        prompt_config={"mode": "plain"},
        condition="canonical_4bit",
    )
    compose = next(
        row["text"]
        for row in render_factorization_prompts(
            tokenizer=tokenizer,
            case=case,
            config={"mode": "plain"},
        )
        if row["name"] == "compose"
    )
    expected = (
        f"Start state: R0={case['initial_state'] & 3}, "
        f"R1={(case['initial_state'] >> 2) & 3}."
    )
    assert expected in encoder
    assert expected in compose
    assert case["state_symbols"] == list("0123456789ABCDEF")


def test_long_horizon_challenges_are_small_balanced_and_deterministic(
    tmp_path,
) -> None:
    run_path = tmp_path / "challenge"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  addition_h128:
    interface_run: runs/interface
    interface_condition: rate_16
    outcome_run: runs/outcome
    domain: addition
    bits: 3
    seed: 51
    horizons: [128]
    program_contexts: 1
    paths_per_state: 1
    block_size: 2
""".strip()
        + "\n"
    )
    first = prepare_interface_challenges(run_path)
    second = prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (run_path / "evaluation/challenges/addition_h128/programs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert first == second
    assert first["profiles"]["addition_h128"]["case_count"] == 8
    assert {row["current_state"] for row in rows} == set(range(8))
    assert all(row["history_steps"] == 128 for row in rows)
    assert all(row["state_path"][-1] == row["current_state"] for row in rows)


def test_proof_depth_challenge_controls_real_deductions(tmp_path) -> None:
    run_path = tmp_path / "challenge"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  proof_depth:
    interface_run: runs/interface
    interface_condition: compressed_3bit
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: query
    bits: 4
    seed: 57
    horizons: [64]
    active_depths: [0, 1, 2, 3, 4]
    program_contexts: 2
    paths_per_depth: 2
    block_size: 2
""".strip()
        + "\n"
    )
    first = prepare_interface_challenges(run_path)
    second = prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (run_path / "evaluation/challenges/proof_depth/programs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert first == second
    assert first["profiles"]["proof_depth"]["case_count"] == 20
    assert first["profiles"]["proof_depth"]["active_depths"] == [0, 1, 2, 3, 4]
    assert all(row["history_steps"] == 64 for row in rows)
    assert all(
        row["active_transition_count"]
        == sum(
            left != right
            for left, right in zip(row["state_path"], row["state_path"][1:])
        )
        for row in rows
    )
    assert {row["active_transition_count"] for row in rows} == {0, 1, 2, 3, 4}
    assert all(row["final_rule"]["kind"] == "proof_query" for row in rows)
    assert all(row["answer_symbols"] == ["0", "1"] for row in rows)
    assert all(
        apply_rule(row["final_rule"], row["current_state"], 16) == row["next_state"]
        for row in rows
    )


def test_closed_horn_training_covers_active_and_identity_updates(tmp_path) -> None:
    run_path = tmp_path / "closed"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_handoff_training:
  dataset:
    domain: horn_proof
    proof_training_contract: closed_one_rule
    proof_consumers: [query, next_rule]
    bits: 4
    seed: 58
    train_examples: 768
    validation_examples: 384
    test_horizons: [2]
    train_program_contexts: 4
    validation_program_contexts: 2
    test_program_contexts: 2
    test_paths_per_state: 1
""".strip()
        + "\n"
    )
    first = prepare_state_handoff_datasets(run_path)
    second = prepare_state_handoff_datasets(run_path)
    rows = read_programs(run_path / TRAIN_PATH)
    assert first == second
    assert first["splits"]["train"]["proof_transition_class_counts"].keys() == {
        "active_conjunction",
        "active_unary",
        "active_unconditional",
        "blocked_conjunction",
        "blocked_unary",
        "idempotent",
    }
    assert all(row["history_steps"] == 1 for row in rows)
    assert all(len(row["history"]) == 1 for row in rows)
    assert {row["proof_consumer"] for row in rows} == {"query", "next_rule"}
    for row in rows:
        changed = row["initial_state"] != row["current_state"]
        assert changed == row["proof_transition_class"].startswith("active_")
        assert (
            apply_rule(row["history"][0], row["initial_state"], 16)
            == row["current_state"]
        )
        if row["proof_consumer"] == "next_rule":
            assert row["final_rule"]["kind"] == "proof_next_rule"
            assert row["answer_symbols"] == ["0", "1", "2", "3", "4"]
            assert (
                apply_rule(row["final_rule"], row["current_state"], 16)
                == row["next_state"]
            )


def test_interface_validation_rejects_incomplete_evaluation_blocks(
    tmp_path, monkeypatch
) -> None:
    run_path = tmp_path / "incomplete-block"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
model:
  name: test
state_handoff_training:
  conditions: [canonical_opaque]
  interfaces:
    independent_module_contexts: false
  dataset:
    test_horizons: [1, 2]
  evaluation:
    block_size: 2
""".strip()
        + "\n"
    )
    for relative, rows in (
        (TRAIN_PATH, []),
        ("training/data/validation_programs.jsonl", []),
        (
            "evaluation/test_programs.jsonl",
            [{"id": "h1", "history_steps": 1}],
        ),
    ):
        path = run_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    monkeypatch.setattr(
        "src.models.hf_loader.load_hf_tokenizer", lambda _config: CharacterTokenizer()
    )

    with pytest.raises(
        ValueError,
        match=r"block size 2 does not divide test horizons \[1\]",
    ):
        validate_state_interface_training_data(run_path)


def test_endpoint_balanced_depth_keeps_targets_fixed_across_depth(tmp_path) -> None:
    run_path = tmp_path / "balanced-depth"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  proof_depth:
    interface_run: runs/interface
    interface_condition: redundant_5bit
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: query
    bits: 4
    seed: 60
    horizons: [32]
    active_depths: [1, 2, 3]
    endpoint_cardinality: 3
    balanced_queries: true
    program_contexts: 4
    paths_per_depth: 2
    block_size: 1
""".strip()
        + "\n"
    )
    manifest = prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (
            run_path / "evaluation/challenges/proof_depth/programs.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert manifest["profiles"]["proof_depth"]["endpoint_cardinalities"] == [3]
    assert {row["current_state"] for row in rows} == {7, 11, 13, 14}
    assert all(row["current_state"].bit_count() == 3 for row in rows)
    assert all(
        row["initial_state"].bit_count()
        == 3 - row["active_transition_count"]
        for row in rows
    )
    for context in {row["program_context"] for row in rows}:
        for path in {row["path_code"] for row in rows}:
            selected = [
                row
                for row in rows
                if row["program_context"] == context and row["path_code"] == path
            ]
            assert len({row["current_state"] for row in selected}) == 1


def test_five_fact_depth_four_has_multiple_matched_endpoints(tmp_path) -> None:
    run_path = tmp_path / "five-fact"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  proof_depth:
    interface_run: runs/interface
    interface_condition: rate_32
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: query
    bits: 5
    seed: 62
    horizons: [32]
    active_depths: [1, 2, 3, 4]
    endpoint_cardinality: 4
    balanced_queries: true
    program_contexts: 5
    paths_per_depth: 1
    block_size: 1
""".strip()
        + "\n"
    )
    prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (
            run_path / "evaluation/challenges/proof_depth/programs.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert {row["current_state"] for row in rows} == {15, 23, 27, 29, 30}
    depth_four = [
        row for row in rows if row["active_transition_count"] == 4
    ]
    assert len({row["current_state"] for row in depth_four}) == 5
    assert all(len(row["state_symbols"]) == 32 for row in rows)


def test_next_rule_consumer_selects_one_applicable_proof_action(tmp_path) -> None:
    run_path = tmp_path / "next-rule"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  proof_action:
    interface_run: runs/interface
    interface_condition: redundant_5bit
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: next_rule
    bits: 4
    seed: 63
    horizons: [16]
    active_depths: [1, 2, 3]
    endpoint_cardinality: 3
    program_contexts: 4
    paths_per_depth: 2
    block_size: 1
""".strip()
        + "\n"
    )
    prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (
            run_path / "evaluation/challenges/proof_action/programs.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    assert {row["proof_consumer"] for row in rows} == {"next_rule"}
    assert {row["next_state"] for row in rows} == {0, 1, 2, 3}
    for row in rows:
        candidates = row["final_rule"]["candidates"]
        active = [
            index
            for index, rule in enumerate(candidates)
            if apply_rule(rule, row["current_state"], 16)
            != row["current_state"]
        ]
        assert active == [row["next_state"]]


def test_balanced_proof_queries_hold_answer_rate_fixed_by_depth(tmp_path) -> None:
    run_path = tmp_path / "challenge"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenges:
  proof_depth:
    interface_run: runs/interface
    interface_condition: redundant_5bit
    outcome_run: runs/outcome
    domain: horn_proof
    proof_final: query
    bits: 4
    seed: 59
    horizons: [64]
    active_depths: [1, 2, 3, 4]
    proof_topologies: [independent, chain, conjunction]
    balanced_queries: true
    program_contexts: 2
    paths_per_depth: 2
    block_size: 2
""".strip()
        + "\n"
    )
    first = prepare_interface_challenges(run_path)
    second = prepare_interface_challenges(run_path)
    rows = [
        json.loads(line)
        for line in (run_path / "evaluation/challenges/proof_depth/programs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert first == second
    assert first["profiles"]["proof_depth"]["case_count"] == 48
    assert first["profiles"]["proof_depth"]["proof_topologies"] == [
        "chain",
        "conjunction",
        "independent",
    ]
    for depth in (1, 2, 3, 4):
        subset = [row for row in rows if row["active_transition_count"] == depth]
        assert sum(row["next_state"] for row in subset) == len(subset) / 2
        assert all(row["next_state"] == row["balanced_query_target"] for row in subset)


def test_challenge_matrix_expands_sources_conditions_and_templates(tmp_path) -> None:
    run_path = tmp_path / "matrix"
    run_path.mkdir()
    (run_path / "config.yaml").write_text(
        """
state_interface_challenge_matrix:
  evaluation_adapter: best
  templates:
    depth:
      domain: horn_proof
      proof_final: query
      bits: 4
      horizons: [64]
      active_depths: [1, 2]
      balanced_queries: true
      program_contexts: 2
      paths_per_depth: 2
  sources:
    - name: small
      interface_run: runs/small-interface
      outcome_run: runs/small-outcome
      challenge_seed: 61
      conditions:
        - {name: canonical_4bit, templates: [depth]}
        - redundant_5bit
""".strip()
        + "\n"
    )
    profiles = configured_challenge_profiles(run_path)
    assert set(profiles) == {
        "small__canonical_4bit__depth",
        "small__redundant_5bit__depth",
    }
    assert profiles["small__canonical_4bit__depth"]["seed"] == 61
    assert profiles["small__canonical_4bit__depth"]["evaluation_adapter"] == "best"
    assert (
        profiles["small__redundant_5bit__depth"]["outcome_owner_profile"]
        == "small__canonical_4bit__depth"
    )
    assert (
        profiles["small__redundant_5bit__depth"]["program_split"]
        == "proof_weekend_small_depth"
    )


def test_proof_depth_summary_pairs_results_by_active_count(tmp_path) -> None:
    root = tmp_path / "evaluation/challenges/proof_depth"
    root.mkdir(parents=True)
    programs = []
    interface = []
    outcome = []
    for depth in (0, 1):
        for context in ("c0", "c1"):
            case_id = f"{context}-d{depth}"
            programs.append(
                {
                    "id": case_id,
                    "program_context": context,
                    "active_transition_count": depth,
                    "current_state": depth,
                }
            )
            interface.append(
                {
                    "id": case_id,
                    "predicted_final": {
                        "is_expected": True,
                        "is_expected_unconstrained": True,
                        "unconstrained_prediction": depth,
                        "candidate_probability_mass": 1.0,
                        "prompt_token_count": 10,
                    },
                    "gold_final": {
                        "is_expected_unconstrained": True,
                    },
                    "predicted_semantic_states": (
                        [depth, depth + 8] if depth == 0 else [depth]
                    ),
                    "steps": [],
                }
            )
            outcome.append(
                {
                    "id": case_id,
                    "prompt_token_count": 10,
                    "conditions": {
                        "one_pass_compose": {"is_expected_unconstrained": depth == 0}
                    },
                }
            )
    for name, rows in (
        ("programs", programs),
        ("interface_cases", interface),
        ("outcome_cases", outcome),
    ):
        (root / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    summary = _write_summary(tmp_path, "proof_depth")
    assert set(summary["by_active_transition_count"]) == {"0", "1"}
    assert set(summary["by_current_state"]) == {"0", "1"}
    assert (
        summary["by_active_transition_count"]["1"]["interface_minus_outcome"]["mean"]
        == 1.0
    )
    assert (
        summary["by_active_transition_count"]["1"]["interface_accuracy"]["cluster_n"]
        == 2
    )
    assert (
        summary["by_active_transition_count"]["1"]["interface_constrained_accuracy"][
            "mean"
        ]
        == 1.0
    )
    assert (
        summary["by_active_transition_count"]["1"]["interface_valid_output_rate"][
            "mean"
        ]
        == 1.0
    )
    assert (
        summary["by_active_transition_count"]["0"]["semantic_state_accuracy"]["mean"]
        == 1.0
    )
    assert (
        summary["by_active_transition_count"]["0"]["exact_state_accuracy"]["mean"]
        == 0.0
    )
    assert summary["overall"]["gold_code_continuation_accuracy"]["mean"] == 1.0


def test_substitution_subset_balances_contexts_and_code_variants() -> None:
    cases = build_test_programs(
        horizons=(2, 8),
        context_count=2,
        paths_per_state=4,
        width=3,
        seed=53,
    )
    selected = _balanced_prefix(cases, len(cases) // 2)
    assert {case["program_context"] for case in selected} == {
        "test_c000",
        "test_c001",
    }
    assert {case["history_steps"] for case in selected} == {2, 8}
    assert {case["current_state"] for case in selected} == set(range(8))
    assert {case["path_code"] for case in selected} == {0, 1}


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
    saved = json.loads((run_path / "evaluation/stress/probe/manifest.json").read_text())
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
    assert all(
        torch.allclose(value, torch.full_like(value, 0.125)) for value in trainable
    )
