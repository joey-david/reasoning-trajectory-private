"""Rate-controlled code contracts for recursive state-interface training."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .benchmark import (
    answer_symbols,
    answer_text,
    candidate_token_ids,
    format_model_prompt,
    state_symbols,
    state_text,
)
from .factorization import render_factorization_history, render_factorization_rule
from .state_interface_contract import (
    CODEBOOK_SIZES,
    INTERFACE_CONDITIONS,
    interface_code_index,
    interface_codebook_size,
    interface_code_symbols,
    is_interface_condition,
    semantic_states_for_code,
)

__all__ = [
    "CODEBOOK_SIZES",
    "INTERFACE_CONDITIONS",
    "interface_code_index",
    "interface_code_symbols",
    "semantic_states_for_code",
]


def _formatted_prompt(
    tokenizer: Any, text: str, prompt_config: dict[str, Any]
) -> str:
    return format_model_prompt(tokenizer, text, {"prompt": prompt_config})


def _code_prompt_preamble(case: dict[str, Any], condition: str) -> str:
    context = (
        f"Interface context: {case['program_context']}.\n"
        if condition == "context_bound"
        else ""
    )
    if case.get("domain") == "horn_proof":
        labels = state_symbols(case)
        facts = tuple(chr(ord("A") + bit) for bit in range(int(case["bits"])))
        entries = []
        for state, label in enumerate(labels):
            present = ",".join(
                facts[bit]
                for bit in range(len(facts))
                if state & (1 << bit)
            ) or "none"
            entries.append(f"{label}={{{present}}}")
        semantics = (
            f"The hidden state is the established-fact bitmask for "
            f"{case['bits']} facts. Public state labels are "
            f"{'; '.join(entries)}."
        )
    elif case.get("domain") in {"mixed_algebra", "algebra_primitives"}:
        semantics = f"The hidden state is a {case['bits']}-bit program register."
    elif case.get("domain") == "register_machine":
        semantics = (
            "The hidden state packs two two-bit registers: R0 is the low pair "
            "and R1 is the high pair."
        )
    else:
        semantics = f"States evolve modulo {2 ** int(case['bits'])}."
    return f"{semantics} Interface codes are opaque single tokens.\n" + context


def render_interface_encoder_prompt(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
) -> str:
    """Render decimal start plus a short history to an opaque code."""
    text = (
        _code_prompt_preamble(case, condition)
        + f"Start state: {state_text(case, int(case['initial_state']))}.\n"
        + render_factorization_history(case)
        + "\nApply every step and return only the resulting interface code.\nAnswer="
    )
    return _formatted_prompt(tokenizer, text, prompt_config)


def render_interface_transition_prompt(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
    input_code: str,
) -> str:
    """Render opaque code plus a short operation block to an opaque code."""
    text = (
        _code_prompt_preamble(case, condition)
        + f"Current interface code: {input_code}.\n"
        + render_factorization_history(case)
        + "\nApply every step and return only the updated interface code.\nAnswer="
    )
    return _formatted_prompt(tokenizer, text, prompt_config)


def render_interface_consumer_prompt(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
    code: str,
) -> str:
    """Render an opaque code plus a final rule to a decimal answer."""
    final = render_factorization_rule(case, case["final_rule"])
    text = (
        _code_prompt_preamble(case, condition)
        + f"Current interface code: {code}.\n"
        + f"FINAL: {final}.\n"
        + "Apply FINAL exactly once and return only the decimal result.\nAnswer="
    )
    return _formatted_prompt(tokenizer, text, prompt_config)


def _target_sequence(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt: str,
    target_symbol: str,
    candidates: tuple[str, ...],
    mapping: str,
) -> dict[str, Any]:
    candidates.index(target_symbol)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + target_symbol, add_special_tokens=False)
    if full_ids[:-1] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
        raise ValueError("Interface target does not extend its prompt by one token")
    target_token_id = int(full_ids[-1])
    return {
        "case_id": str(case["id"]),
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["program_context"]),
        "mapping": mapping,
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + [target_token_id],
        "target_token_id": target_token_id,
    }


def interface_training_sequence_pair(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
    interface_config: dict[str, Any],
    max_length: int,
    producer_mode: str = "mixed",
    transition_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Render one state-contract target and one consumer target."""
    symbols = interface_code_symbols(condition, interface_config)
    variant = (
        int(case["path_code"])
        if condition.startswith("rate_")
        else int(case["path_code"]) % 2
    )
    input_index = interface_code_index(
        condition=condition,
        case=case,
        state=int(case["initial_state"]),
        interface_config=interface_config,
        variant=variant,
    )
    output_index = interface_code_index(
        condition=condition,
        case=case,
        state=int(case["current_state"]),
        interface_config=interface_config,
        variant=variant,
    )
    if producer_mode not in {"mixed", "encoder", "transition"}:
        raise ValueError(f"Unknown interface producer mode: {producer_mode!r}")
    if not 0.0 <= transition_fraction <= 1.0:
        raise ValueError("Interface transition fraction must be in [0, 1]")
    digest = hashlib.sha256(str(case["id"]).encode()).digest()
    if transition_fraction == 0.5:
        selected_transition = digest[0] % 2 == 1
    else:
        draw = int.from_bytes(digest[:8], "big") / 2**64
        selected_transition = draw < transition_fraction
    use_encoder = producer_mode == "encoder" or (
        producer_mode == "mixed" and not selected_transition
    )
    if use_encoder:
        state_prompt = render_interface_encoder_prompt(
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt_config,
            condition=condition,
        )
    else:
        state_prompt = render_interface_transition_prompt(
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt_config,
            condition=condition,
            input_code=symbols[input_index],
        )
    state_sequence = _target_sequence(
        tokenizer=tokenizer,
        case=case,
        prompt=state_prompt,
        target_symbol=symbols[output_index],
        candidates=symbols,
        mapping="state",
    )
    state_sequence["producer_mode"] = producer_mode
    state_sequence["producer_transition_fraction"] = transition_fraction
    state_sequence["producer_prompt_kind"] = (
        "encoder" if use_encoder else "transition"
    )
    consumer_prompt = render_interface_consumer_prompt(
        tokenizer=tokenizer,
        case=case,
        prompt_config=prompt_config,
        condition=condition,
        code=symbols[output_index],
    )
    candidates = answer_symbols(case)
    answer_sequence = _target_sequence(
        tokenizer=tokenizer,
        case=case,
        prompt=consumer_prompt,
        target_symbol=answer_text(case, int(case["next_state"])),
        candidates=candidates,
        mapping="answer",
    )
    pair = [state_sequence, answer_sequence]
    target_tokens = max_length * len(pair)
    for row in pair:
        if len(row["input_ids"]) > max_length:
            raise ValueError(f"Interface case {case['id']} exceeds {max_length} tokens")
        added = max_length - len(row["input_ids"])
        row["input_ids"].extend([int(tokenizer.pad_token_id)] * added)
        row["labels"].extend([-100] * added)
        row["control_tail_tokens"] = added
    if sum(len(row["input_ids"]) for row in pair) != target_tokens:
        raise AssertionError("Interface fixed-padding compute contract failed")
    return pair


def build_interface_training_pairs(
    *,
    tokenizer: Any,
    cases: Iterable[dict[str, Any]],
    prompt_config: dict[str, Any],
    condition: str,
    interface_config: dict[str, Any],
    max_length: int,
) -> list[list[dict[str, Any]]]:
    """Tokenize producer and consumer pairs, with disjoint contexts by default."""
    cases = list(cases)
    producer_mode = str(
        interface_config.get("producer_modes", {}).get(condition, "mixed")
    )
    transition_fraction = float(
        interface_config.get("producer_transition_fractions", {}).get(
            condition, 0.5
        )
    )
    if cases:
        symbols = interface_code_symbols(condition, interface_config)
        sample = cases[0]
        candidate_token_ids(
            tokenizer,
            render_interface_encoder_prompt(
                tokenizer=tokenizer,
                case=sample,
                prompt_config=prompt_config,
                condition=condition,
            ),
            symbols,
        )
        candidate_token_ids(
            tokenizer,
            render_interface_consumer_prompt(
                tokenizer=tokenizer,
                case=sample,
                prompt_config=prompt_config,
                condition=condition,
                code=symbols[0],
            ),
            answer_symbols(sample),
        )
    if not bool(interface_config.get("independent_module_contexts", True)):
        return [
            interface_training_sequence_pair(
                tokenizer=tokenizer,
                case=case,
                prompt_config=prompt_config,
                condition=condition,
                interface_config=interface_config,
                max_length=max_length,
                producer_mode=producer_mode,
                transition_fraction=transition_fraction,
            )
            for case in cases
        ]
    contexts = sorted({str(case["program_context"]) for case in cases})
    if len(contexts) < 2:
        raise ValueError("Independent interface modules require at least two contexts")
    producer_contexts = set(contexts[::2])
    consumer_contexts = set(contexts[1::2])
    producers = [case for case in cases if case["program_context"] in producer_contexts]
    consumers = [case for case in cases if case["program_context"] in consumer_contexts]
    if not producers or not consumers:
        raise AssertionError("Producer/consumer context split is empty")
    pairs = []
    for index in range(len(cases)):
        producer_pair = interface_training_sequence_pair(
            tokenizer=tokenizer,
            case=producers[index % len(producers)],
            prompt_config=prompt_config,
            condition=condition,
            interface_config=interface_config,
            max_length=max_length,
            producer_mode=producer_mode,
            transition_fraction=transition_fraction,
        )
        consumer_pair = interface_training_sequence_pair(
            tokenizer=tokenizer,
            case=consumers[index % len(consumers)],
            prompt_config=prompt_config,
            condition=condition,
            interface_config=interface_config,
            max_length=max_length,
            producer_mode=producer_mode,
            transition_fraction=transition_fraction,
        )
        pairs.append([producer_pair[0], consumer_pair[1]])
    return pairs


def matched_interface_compute_manifest(
    *,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    prompt_config: dict[str, Any],
    conditions: tuple[str, ...],
    interface_config: dict[str, Any],
    max_length: int,
) -> dict[str, Any]:
    """Prove equal fixed-padding budgets across every code condition."""
    summaries = {}
    for condition in conditions:
        pairs = build_interface_training_pairs(
            tokenizer=tokenizer,
            cases=cases,
            prompt_config=prompt_config,
            condition=condition,
            interface_config=interface_config,
            max_length=max_length,
        )
        sequences = [sequence for pair in pairs for sequence in pair]
        summaries[condition] = {
            "semantic_programs": len(pairs),
            "forward_passes": len(sequences),
            "fixed_padding_compute_tokens": len(sequences) * max_length,
            "active_input_tokens": sum(len(row["input_ids"]) for row in sequences),
            "target_tokens": sum(
                sum(label != -100 for label in row["labels"]) for row in sequences
            ),
            "codebook_size": interface_codebook_size(
                condition, interface_config
            ),
            "capacity_bits": math.log2(
                interface_codebook_size(condition, interface_config)
            ),
            "independent_module_contexts": bool(
                interface_config.get("independent_module_contexts", True)
            ),
            "producer_mode": str(
                interface_config.get("producer_modes", {}).get(condition, "mixed")
            ),
            "producer_transition_fraction": float(
                interface_config.get("producer_transition_fractions", {}).get(
                    condition, 0.5
                )
            ),
            "producer_prompt_counts": {
                kind: sum(
                    row.get("producer_prompt_kind") == kind
                    for row in sequences
                )
                for kind in ("encoder", "transition")
            },
        }
    comparable = list(summaries.values())
    keys = ("semantic_programs", "forward_passes", "fixed_padding_compute_tokens", "active_input_tokens", "target_tokens")
    matched = all(
        row[key] == comparable[0][key]
        for row in comparable[1:]
        for key in keys
    )
    return {
        "schema_version": 1,
        "max_length": max_length,
        "conditions": summaries,
        "matched_forward_passes_and_tokens": matched,
        "matched_forward_passes_and_compute_tokens": matched,
        "token_matching_contract": (
            "Every condition uses two fixed-length forwards and two supervised "
            "targets per semantic program. Loss-masked tail tokens follow targets."
        ),
        "module_split_contract": (
            "State producers use even-indexed program contexts; code consumers "
            "use odd-indexed contexts. Test contexts are disjoint from both."
            if interface_config.get("independent_module_contexts", True)
            else "Producer and consumer contexts are shared."
        ),
    }


def validate_state_interface_training_data(run_path: Path) -> dict[str, Any]:
    """Validate all code alphabets and matched budgets before GPU work."""
    from src.models.hf_loader import load_hf_tokenizer

    from .state_handoff_data import (
        COMPUTE_MANIFEST_PATH,
        TEST_PATH,
        TRAIN_PATH,
        VALIDATION_PATH,
        configured_training_conditions,
        read_programs,
    )

    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    conditions = configured_training_conditions(run_path)
    if not all(is_interface_condition(condition) for condition in conditions):
        raise ValueError("Interface validation requires only code conditions")
    tokenizer = load_hf_tokenizer(config["model"])
    maximum = int(experiment.get("training", {}).get("max_sequence_length", 256))
    evaluation_maximum = int(
        experiment.get("evaluation", {}).get(
            "max_sequence_length", maximum
        )
    )
    common = {
        "tokenizer": tokenizer,
        "prompt_config": experiment.get("prompt", {}),
        "conditions": conditions,
        "interface_config": experiment.get("interfaces", {}),
        "max_length": maximum,
    }
    train = matched_interface_compute_manifest(
        cases=read_programs(run_path / TRAIN_PATH), **common
    )
    validation = matched_interface_compute_manifest(
        cases=read_programs(run_path / VALIDATION_PATH), **common
    )
    if not train["matched_forward_passes_and_tokens"]:
        raise ValueError("State-interface training budgets are not matched")
    test_max = 0
    for case in read_programs(run_path / TEST_PATH):
        local = {
            **case,
            "history": list(case["history"][:2]),
            "history_steps": 2,
        }
        for condition in conditions:
            symbols = interface_code_symbols(
                condition, experiment.get("interfaces", {})
            )
            prompts = (
                render_interface_encoder_prompt(
                    tokenizer=tokenizer,
                    case=local,
                    prompt_config=experiment.get("prompt", {}),
                    condition=condition,
                ),
                render_interface_transition_prompt(
                    tokenizer=tokenizer,
                    case=local,
                    prompt_config=experiment.get("prompt", {}),
                    condition=condition,
                    input_code=symbols[0],
                ),
                render_interface_consumer_prompt(
                    tokenizer=tokenizer,
                    case=case,
                    prompt_config=experiment.get("prompt", {}),
                    condition=condition,
                    code=symbols[0],
                ),
            )
            test_max = max(
                test_max,
                *(
                    len(tokenizer.encode(prompt, add_special_tokens=False)) + 1
                    for prompt in prompts
                ),
            )
    if test_max > evaluation_maximum:
        raise ValueError(
            f"Recursive evaluation sequence length {test_max} exceeds "
            f"{evaluation_maximum}"
        )
    train["validation"] = validation["conditions"]
    train["test_max_active_sequence_length"] = test_max
    train["evaluation_max_sequence_length"] = evaluation_maximum
    train["validated"] = True
    write_json(run_path / COMPUTE_MANIFEST_PATH, train)
    return train
