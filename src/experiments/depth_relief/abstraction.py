"""Matched-history assay for formation of causal state abstractions."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np

from .benchmark import (
    apply_rule,
    candidate_token_ids,
    state_symbols,
    state_text,
)
from .factorization import (
    FORMATS,
    compose_capture_positions,
    factorization_record,
    render_factorization_prompts,
)
from .qualification import evaluate_prompt_conditions_hf
from .transfer import capture_prompt_states_hf


CAPTURED_CONDITIONS = ("compose", "synthesize", "update")


def _state_path(initial: int, history: list[dict[str, Any]], modulus: int) -> list[int]:
    states = [initial]
    for rule in history:
        states.append(apply_rule(rule, states[-1], modulus))
    return states


def _history_for_target(
    *,
    initial: int,
    target: int,
    path_code: int,
    history_steps: int,
    group_index: int,
    modulus: int,
) -> list[dict[str, Any]]:
    if history_steps == 1:
        value = (target - initial) % modulus + path_code * modulus
        return [{"kind": "add", "value": value}]
    values = [path_code]
    values.extend(
        (3 * group_index + 2 * step + 1) % modulus
        for step in range(1, history_steps - 1)
    )
    current = (initial + sum(values)) % modulus
    values.append((target - current) % modulus)
    return [{"kind": "add", "value": value} for value in values]


def _diagnostic_targets(case: dict[str, Any]) -> dict[str, int]:
    modulus = 2 ** int(case["bits"])
    return {
        "correct_composition": int(case["next_state"]),
        "history_only": int(case["current_state"]),
        "final_on_start": apply_rule(
            case["final_rule"], int(case["initial_state"]), modulus
        ),
        "identity": int(case["initial_state"]),
    }


def build_state_abstraction_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build balanced history equivalence classes with minimal state contrasts."""
    width = int(config.get("bits", 3))
    modulus = 2**width
    history_steps_values = tuple(
        int(value) for value in config.get("history_steps", (2, 4))
    )
    groups_per_horizon = int(config.get("groups_per_horizon", 3))
    path_count = int(config.get("paths_per_state", modulus))
    formats = tuple(str(value) for value in config.get("formats", ("prose",)))
    seed = int(config.get("seed", 0))
    if width < 3:
        raise ValueError("State abstraction requires at least eight states")
    if groups_per_horizon < 3:
        raise ValueError("Each horizon needs train, validation, and test groups")
    if groups_per_horizon % 3:
        raise ValueError("Groups per horizon must balance equally across splits")
    if path_count != modulus:
        raise ValueError("Paths per state must equal the state count for balance")
    if any(value < 1 for value in history_steps_values):
        raise ValueError("Matched histories require at least one operation")
    if not set(formats).issubset(FORMATS):
        raise ValueError(f"Unknown factorization formats: {formats}")

    cases: list[dict[str, Any]] = []
    example_index = 0
    split_names = ("train", "validation", "test")
    for history_steps in history_steps_values:
        for group_index in range(groups_per_horizon):
            rng = random.Random(
                seed + 100_003 * history_steps + 10_007 * group_index
            )
            initial = rng.randrange(modulus)
            mapping = list(range(modulus))
            rng.shuffle(mapping)
            final_rule = {"kind": "pointer", "mapping": mapping}
            group = f"h{history_steps}_g{group_index}"
            split = split_names[group_index % len(split_names)]
            for target in range(modulus):
                for path_code in range(path_count):
                    history = _history_for_target(
                        initial=initial,
                        target=target,
                        path_code=path_code,
                        history_steps=history_steps,
                        group_index=group_index,
                        modulus=modulus,
                    )
                    state_path = _state_path(initial, history, modulus)
                    if state_path[-1] != target:
                        raise AssertionError("Matched history missed its target state")
                    counterfactual = (target + 1) % modulus
                    semantic = {
                        "family": "add_to_pointer",
                        "history_family": "add",
                        "final_family": "pointer",
                        "bits": width,
                        "example_index": example_index,
                        "initial_state": initial,
                        "history": history,
                        "final_rule": final_rule,
                        "current_state": target,
                        "next_state": apply_rule(final_rule, target, modulus),
                        "counterfactual_state": counterfactual,
                        "counterfactual_next_state": apply_rule(
                            final_rule, counterfactual, modulus
                        ),
                        "random_state": (target + 2) % modulus,
                        "random_next_state": apply_rule(
                            final_rule, (target + 2) % modulus, modulus
                        ),
                        "history_steps": history_steps,
                        "state_path": state_path,
                        "abstraction_group": group,
                        "abstraction_split": split,
                        "path_code": path_code,
                    }
                    semantic["diagnostic_targets"] = _diagnostic_targets(semantic)
                    for representation in formats:
                        row = dict(semantic)
                        row["format"] = representation
                        digest = hashlib.sha256(
                            json.dumps(
                                {
                                    "initial_state": initial,
                                    "history": history,
                                    "final_rule": final_rule,
                                    "format": representation,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()[:10]
                        row["id"] = (
                            f"abstraction_{group}_s{target}_p{path_code}_"
                            f"{representation}_{digest}"
                        )
                        cases.append(row)
                    example_index += 1
    return cases


def _token_span(
    tokenizer: Any, text: str, marker: str, value: str
) -> tuple[int, ...]:
    if text.count(marker) != 1:
        raise ValueError(f"Prompt marker is not unique: {marker!r}")
    value_start = text.index(marker) + marker.index(value)
    value_end = value_start + len(value)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = tuple(
        index
        for index, (start, end) in enumerate(encoded["offset_mapping"])
        if int(end) > value_start and int(start) < value_end
    )
    if len(indices) != 1:
        raise ValueError("Explicit state does not occupy exactly one token")
    return indices


def update_capture_positions(
    *, tokenizer: Any, case: dict[str, Any], text: str
) -> list[dict[str, int | str]]:
    """Locate the explicit state token and final answer anchor."""
    value = state_text(case, int(case["current_state"]))
    marker = f"Current state: {value}."
    state_index = _token_span(tokenizer, text, marker, value)[0]
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if state_index >= token_count - 1:
        raise ValueError("Explicit state must precede the answer anchor")
    return [
        {"name": "state", "token_index": state_index},
        {"name": "answer", "token_index": token_count - 1},
    ]


def validate_abstraction_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Validate state tokens and all semantic capture positions."""
    prompts = render_factorization_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    indexed = {str(prompt["name"]): prompt for prompt in prompts}
    for prompt in prompts:
        candidate_token_ids(tokenizer, prompt["text"], state_symbols(case))
    compose_positions = compose_capture_positions(
        tokenizer=tokenizer,
        case=case,
        text=indexed["compose"]["text"],
    )
    update_positions = update_capture_positions(
        tokenizer=tokenizer,
        case=case,
        text=indexed["update"]["text"],
    )
    compose_token_ids = tokenizer.encode(
        indexed["compose"]["text"], add_special_tokens=False
    )
    token_counts = [
        len(tokenizer.encode(prompt["text"], add_special_tokens=False))
        for prompt in prompts
    ]
    return {
        "id": str(case["id"]),
        "condition_count": len(prompts),
        "token_count_range": [min(token_counts), max(token_counts)],
        "compose_positions": compose_positions,
        "compose_position_token_ids": {
            str(position["name"]): int(
                compose_token_ids[int(position["token_index"])]
            )
            for position in compose_positions
        },
        "update_positions": update_positions,
    }


def _finish_record(
    *,
    tokenizer: Any,
    prompt: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    expected = int(prompt["expected_next_state"])
    record.update(
        expected_next_state=expected,
        is_expected=record["prediction"] == expected,
        is_expected_unconstrained=record["unconstrained_prediction"] == expected,
        token_count=len(
            tokenizer.encode(prompt["text"], add_special_tokens=False)
        ),
        checkpoint_char_span=None,
    )
    return record


def capture_abstraction_case_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score all controls while retaining only state-relevant residual streams."""
    prompts = render_factorization_prompts(
        tokenizer=tokenizer, case=case, config=config
    )
    indexed = {str(prompt["name"]): prompt for prompt in prompts}
    validation = validate_abstraction_case(
        tokenizer=tokenizer, case=case, config=config
    )
    conditions: dict[str, dict[str, Any]] = {}
    activations: dict[str, np.ndarray] = {}
    for name in CAPTURED_CONDITIONS:
        prompt = indexed[name]
        if name == "compose":
            positions = validation["compose_positions"]
        elif name == "update":
            positions = validation["update_positions"]
        else:
            positions = [{"name": "answer", "token_index": -1}]
        record, states = capture_prompt_states_hf(
            model=model,
            tokenizer=tokenizer,
            text=prompt["text"],
            candidate_ids=candidate_token_ids(
                tokenizer, prompt["text"], state_symbols(case)
            ),
            token_indices=tuple(int(row["token_index"]) for row in positions),
        )
        conditions[name] = _finish_record(
            tokenizer=tokenizer, prompt=prompt, record=record
        )
        activations[f"{name}_trace"] = states

    controls = [
        prompt for prompt in prompts if str(prompt["name"]) not in CAPTURED_CONDITIONS
    ]
    conditions.update(
        evaluate_prompt_conditions_hf(
            model=model,
            tokenizer=tokenizer,
            prompts=controls,
            candidate_symbols=state_symbols(case),
        )
    )
    row = factorization_record(case, conditions)
    row.update(
        abstraction_group=str(case["abstraction_group"]),
        abstraction_split=str(case["abstraction_split"]),
        path_code=int(case["path_code"]),
        compose_positions=validation["compose_positions"],
        update_positions=validation["update_positions"],
    )
    return row, activations
