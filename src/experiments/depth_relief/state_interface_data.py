"""Rate-controlled code contracts for recursive state-interface training."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .benchmark import candidate_token_ids, format_model_prompt
from .factorization import render_factorization_history
from .state_handoff_data import INTERFACE_CONDITIONS


DEFAULT_CODE_SYMBOLS = tuple("αβγδεζηθικλμνξοπ")
DEFAULT_CANONICAL_PERMUTATION = (5, 2, 7, 1, 6, 0, 3, 4)
CODEBOOK_SIZES = {
    "canonical_opaque": 8,
    "context_bound": 8,
    "compressed_2bit": 4,
    "redundant_4bit": 16,
}


def interface_code_symbols(
    condition: str, interface_config: dict[str, Any]
) -> tuple[str, ...]:
    """Return the declared one-token alphabet for one interface condition."""
    if condition not in INTERFACE_CONDITIONS:
        raise ValueError(f"Unknown state-interface condition: {condition!r}")
    size = CODEBOOK_SIZES[condition]
    configured = interface_config.get(condition, {}).get("symbols")
    symbols = tuple(str(value) for value in configured) if configured else DEFAULT_CODE_SYMBOLS[:size]
    if len(symbols) != size or len(set(symbols)) != size:
        raise ValueError(f"{condition} requires {size} unique code symbols")
    return symbols


def _context_permutation(case: dict[str, Any], seed: int) -> tuple[int, ...]:
    digest = hashlib.sha256(
        f"{seed}:{case['program_context']}".encode()
    ).digest()
    values = list(range(8))
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(values)
    return tuple(values)


def interface_code_index(
    *,
    condition: str,
    case: dict[str, Any],
    state: int,
    interface_config: dict[str, Any],
    variant: int | None = None,
) -> int:
    """Map a semantic state to a rate-controlled code index."""
    if not 0 <= int(state) < 8:
        raise ValueError("Interface experiments require an eight-state process")
    if condition == "canonical_opaque":
        permutation = tuple(
            int(value)
            for value in interface_config.get(condition, {}).get(
                "permutation", DEFAULT_CANONICAL_PERMUTATION
            )
        )
        if sorted(permutation) != list(range(8)):
            raise ValueError("Canonical opaque mapping must be a permutation of 0..7")
        return permutation[int(state)]
    if condition == "context_bound":
        seed = int(interface_config.get(condition, {}).get("seed", 721_701))
        return _context_permutation(case, seed)[int(state)]
    if condition == "compressed_2bit":
        return int(state) % 4
    if condition == "redundant_4bit":
        nuisance = int(case.get("path_code", 0)) % 2 if variant is None else int(variant)
        if nuisance not in (0, 1):
            raise ValueError("The redundant code variant must be one bit")
        return 2 * int(state) + nuisance
    raise ValueError(f"Unknown state-interface condition: {condition!r}")


def semantic_states_for_code(
    *,
    condition: str,
    case: dict[str, Any],
    code_index: int,
    interface_config: dict[str, Any],
) -> tuple[int, ...]:
    """Return every semantic state compatible with one code."""
    return tuple(
        state
        for state in range(8)
        if interface_code_index(
            condition=condition,
            case=case,
            state=state,
            interface_config=interface_config,
            variant=code_index % 2,
        )
        == code_index
    )


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
    return (
        "States evolve modulo 8. Interface codes are opaque single tokens.\n"
        + context
    )


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
        + f"Start state: {case['initial_state']}.\n"
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
    """Render an opaque code plus a decimal FINAL table to a decimal answer."""
    table = ", ".join(str(value) for value in case["final_rule"]["mapping"])
    text = (
        _code_prompt_preamble(case, condition)
        + f"Current interface code: {code}.\n"
        + f"FINAL: look up the decoded state in [{table}].\n"
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
    candidate_ids = candidate_token_ids(tokenizer, prompt, candidates)
    target_index = candidates.index(target_symbol)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + target_symbol, add_special_tokens=False)
    if full_ids != [*prompt_ids, candidate_ids[target_index]]:
        raise ValueError("Interface target does not extend its prompt by one token")
    return {
        "case_id": str(case["id"]),
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["program_context"]),
        "mapping": mapping,
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + [candidate_ids[target_index]],
        "target_token_id": candidate_ids[target_index],
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
) -> list[dict[str, Any]]:
    """Render one state-contract target and one consumer target."""
    symbols = interface_code_symbols(condition, interface_config)
    variant = int(case["path_code"]) % 2
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
    mapping_selector = hashlib.sha256(str(case["id"]).encode()).digest()[0] % 2
    use_encoder = producer_mode == "encoder" or (
        producer_mode == "mixed" and mapping_selector == 0
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
    answer_symbols = tuple(str(value) for value in range(8))
    answer_sequence = _target_sequence(
        tokenizer=tokenizer,
        case=case,
        prompt=consumer_prompt,
        target_symbol=str(case["next_state"]),
        candidates=answer_symbols,
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
        )
        consumer_pair = interface_training_sequence_pair(
            tokenizer=tokenizer,
            case=consumers[index % len(consumers)],
            prompt_config=prompt_config,
            condition=condition,
            interface_config=interface_config,
            max_length=max_length,
            producer_mode=producer_mode,
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
            "codebook_size": CODEBOOK_SIZES[condition],
            "capacity_bits": math.log2(CODEBOOK_SIZES[condition]),
            "independent_module_contexts": bool(
                interface_config.get("independent_module_contexts", True)
            ),
            "producer_mode": str(
                interface_config.get("producer_modes", {}).get(condition, "mixed")
            ),
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
        TRAIN_PATH,
        VALIDATION_PATH,
        configured_training_conditions,
        read_programs,
    )

    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    conditions = configured_training_conditions(run_path)
    if not set(conditions).issubset(INTERFACE_CONDITIONS):
        raise ValueError("Interface validation requires only code conditions")
    tokenizer = load_hf_tokenizer(config["model"])
    maximum = int(experiment.get("training", {}).get("max_sequence_length", 256))
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
    train["validation"] = validation["conditions"]
    train["validated"] = True
    write_json(run_path / COMPUTE_MANIFEST_PATH, train)
    return train
