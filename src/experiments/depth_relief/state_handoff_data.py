"""Deterministic program splits and masked one-token training sequences."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl

from .benchmark import (
    answer_symbols,
    answer_text,
    candidate_token_ids,
    state_symbols,
    state_text,
)
from .factorization import (
    render_factorization_prompts,
    render_factorization_update_prompt,
)
from .state_interface_contract import (
    INTERFACE_CONDITIONS,
    is_interface_condition,
)
from .state_handoff_programs import (
    _balanced_programs,
    build_closed_horn_programs,
    build_test_programs,
)


TRAIN_PATH = Path("training/data/train_programs.jsonl")
VALIDATION_PATH = Path("training/data/validation_programs.jsonl")
TEST_PATH = Path("evaluation/test_programs.jsonl")
DATA_MANIFEST_PATH = Path("training/data/manifest.json")
COMPUTE_MANIFEST_PATH = Path("training/compute_manifest.json")
TRAINING_CONDITIONS = ("outcome_only", "explicit_handoff")
ALL_TRAINING_CONDITIONS = TRAINING_CONDITIONS + INTERFACE_CONDITIONS


def configured_training_conditions(run_path: Path) -> tuple[str, ...]:
    """Return and validate the conditions owned by one run configuration."""
    configured = tuple(
        str(value)
        for value in load_config(run_path)
        .get("state_handoff_training", {})
        .get("conditions", TRAINING_CONDITIONS)
    )
    unknown = sorted(
        condition
        for condition in configured
        if condition not in TRAINING_CONDITIONS
        and not is_interface_condition(condition)
    )
    if unknown:
        raise ValueError(f"Unknown state-handoff training conditions: {unknown}")
    if not configured or len(configured) != len(set(configured)):
        raise ValueError("State-handoff training conditions must be unique and nonempty")
    return configured


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_summary(rows: list[dict[str, Any]], width: int) -> dict[str, Any]:
    result = {
        "semantic_program_count": len(rows),
        "program_context_count": len({row["program_context"] for row in rows}),
        "horizons": sorted({int(row["history_steps"]) for row in rows}),
        "states": sorted({int(row["current_state"]) for row in rows}),
        "minimum_histories_per_state": min(
            sum(int(row["current_state"]) == state for row in rows)
            for state in range(2**width)
        ),
    }
    if any("domain" in row for row in rows):
        result["domains"] = sorted({str(row["domain"]) for row in rows})
        result["composition_splits"] = sorted(
            {str(row["composition_split"]) for row in rows}
        )
    if any("proof_transition_class" in row for row in rows):
        result["proof_transition_class_counts"] = {
            transition_class: sum(
                row.get("proof_transition_class") == transition_class
                for row in rows
            )
            for transition_class in sorted(
                {
                    str(row["proof_transition_class"])
                    for row in rows
                    if "proof_transition_class" in row
                }
            )
        }
    return result


def prepare_state_handoff_datasets(run_path: Path) -> dict[str, Any]:
    """Write group-disjoint pilot programs and their stable hashes."""
    config = load_config(run_path).get("state_handoff_training", {})
    dataset = config.get("dataset", {})
    width = int(dataset.get("bits", 3))
    if not 2 <= width <= 5:
        raise ValueError("Discrete state-handoff training supports two to five bits")
    sequences_per_condition = int(dataset.get("train_examples", 20_000))
    validation_sequences = int(dataset.get("validation_examples", 2_000))
    if sequences_per_condition % 2 or validation_sequences % 2:
        raise ValueError("Matched two-call training example counts must be even")
    seed = int(dataset.get("seed", 721_301))
    closed_horn = (
        str(dataset.get("proof_training_contract", "endpoint"))
        == "closed_one_rule"
    )
    builder = build_closed_horn_programs if closed_horn else _balanced_programs
    shared = {
        "width": width,
        "seed": seed,
        "dataset": dataset,
    }
    if closed_horn:
        train = builder(
            split="train",
            semantic_count=sequences_per_condition // 2,
            context_count=int(dataset.get("train_program_contexts", 50)),
            **shared,
        )
        validation = builder(
            split="validation",
            semantic_count=validation_sequences // 2,
            context_count=int(dataset.get("validation_program_contexts", 15)),
            **shared,
        )
    else:
        train = builder(
            split="train",
            semantic_count=sequences_per_condition // 2,
            horizons=tuple(
                int(value) for value in dataset.get("train_horizons", (1, 2))
            ),
            context_count=int(dataset.get("train_program_contexts", 50)),
            **shared,
        )
        validation = builder(
            split="validation",
            semantic_count=validation_sequences // 2,
            horizons=tuple(
                int(value)
                for value in dataset.get("validation_horizons", (1, 2))
            ),
            context_count=int(dataset.get("validation_program_contexts", 15)),
            **shared,
        )
    test = build_test_programs(
        horizons=tuple(int(value) for value in dataset.get("test_horizons", (2, 4, 8))),
        context_count=int(dataset.get("test_program_contexts", 30)),
        paths_per_state=int(dataset.get("test_paths_per_state", 4)),
        width=width,
        seed=seed,
        dataset=dataset,
    )
    context_sets = {
        split: {row["program_context"] for row in rows}
        for split, rows in (("train", train), ("validation", validation), ("test", test))
    }
    if any(
        context_sets[left] & context_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise AssertionError("Program-context splits overlap")
    paths = {
        "train": run_path / TRAIN_PATH,
        "validation": run_path / VALIDATION_PATH,
        "test": run_path / TEST_PATH,
    }
    for split, rows in (("train", train), ("validation", validation), ("test", test)):
        write_jsonl(paths[split], rows)
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "bits": width,
        "state_slots": 1,
        "codebook_size": 2**width,
        "capacity_bits": width,
        "training_sequences_per_condition": 2 * len(train),
        "validation_sequences_per_condition": 2 * len(validation),
        "splits": {
            split: {**_split_summary(rows, width), "sha256": _sha256(paths[split])}
            for split, rows in (("train", train), ("validation", validation), ("test", test))
        },
        "group_disjoint": True,
    }
    write_json(run_path / DATA_MANIFEST_PATH, manifest)
    return manifest


def read_programs(path: Path) -> list[dict[str, Any]]:
    """Read semantic programs and reject duplicate IDs."""
    rows = load_samples(path)
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate state-handoff program IDs in {path}")
    return rows


def _sequence(
    *, tokenizer: Any, case: dict[str, Any], prompt: dict[str, Any], mapping: str
) -> dict[str, Any]:
    target = int(prompt["expected_next_state"])
    prompt_ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
    target_text = (
        answer_text(case, target) if mapping == "answer" else state_text(case, target)
    )
    full_ids = tokenizer.encode(prompt["text"] + target_text, add_special_tokens=False)
    if full_ids[:-1] != prompt_ids or len(full_ids) != len(prompt_ids) + 1:
        raise ValueError("Training target does not extend the prompt by one token")
    target_id = int(full_ids[-1])
    return {
        "case_id": str(case["id"]),
        "history_steps": int(case["history_steps"]),
        "program_context": str(case["program_context"]),
        "mapping": mapping,
        "input_ids": full_ids,
        "labels": [-100] * len(prompt_ids) + [target_id],
        "target_token_id": target_id,
    }


def training_sequence_pair(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
    condition: str,
    max_length: int,
    fixed_sequence_padding: bool = False,
) -> list[dict[str, Any]]:
    """Render two matched forwards for one semantic program."""
    if condition not in TRAINING_CONDITIONS:
        raise ValueError(f"Unknown state-handoff training condition: {condition!r}")
    prompts = {
        str(prompt["name"]): prompt
        for prompt in render_factorization_prompts(
            tokenizer=tokenizer, case=case, config=prompt_config
        )
    }
    outcome_pair = [
        _sequence(
            tokenizer=tokenizer,
            case=case,
            prompt=prompts["compose"],
            mapping="answer",
        )
        for _ in range(2)
    ]
    answer_prompt = render_factorization_update_prompt(
        tokenizer=tokenizer,
        case=case,
        config=prompt_config,
        state=int(case["current_state"]),
        rule=case["final_rule"],
        name="handoff_answer",
        label="FINAL",
    )
    handoff_pair = [
        _sequence(
            tokenizer=tokenizer,
            case=case,
            prompt=prompts["synthesize"],
            mapping="state",
        ),
        _sequence(
            tokenizer=tokenizer,
            case=case,
            prompt=answer_prompt,
            mapping="answer",
        ),
    ]
    pairs = {
        "outcome_only": outcome_pair,
        "explicit_handoff": handoff_pair,
    }
    target_tokens = (
        2 * max_length
        if fixed_sequence_padding
        else max(sum(len(row["input_ids"]) for row in pair) for pair in pairs.values())
    )
    for pair in pairs.values():
        remaining = target_tokens - sum(len(row["input_ids"]) for row in pair)
        for row in reversed(pair):
            added = min(remaining, max_length - len(row["input_ids"]))
            row["input_ids"].extend([int(tokenizer.pad_token_id)] * added)
            row["labels"].extend([-100] * added)
            row["control_tail_tokens"] = added
            remaining -= added
        if remaining:
            raise ValueError(
                f"Matched token padding for {case['id']} exceeds max length {max_length}"
            )
    pair = pairs[condition]
    if any(len(sequence["input_ids"]) > max_length for sequence in pair):
        raise ValueError(f"Training case {case['id']} exceeds max length {max_length}")
    return pair


def build_training_pairs(
    *,
    tokenizer: Any,
    cases: Iterable[dict[str, Any]],
    prompt_config: dict[str, Any],
    condition: str,
    max_length: int,
    fixed_sequence_padding: bool = False,
) -> list[list[dict[str, Any]]]:
    """Tokenize semantic programs into stable two-forward training pairs."""
    cases = list(cases)
    if cases:
        prompts = render_factorization_prompts(
            tokenizer=tokenizer, case=cases[0], config=prompt_config
        )
        candidate_token_ids(
            tokenizer,
            next(row["text"] for row in prompts if row["name"] == "compose"),
            answer_symbols(cases[0]),
        )
    return [
        training_sequence_pair(
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt_config,
            condition=condition,
            max_length=max_length,
            fixed_sequence_padding=fixed_sequence_padding,
        )
        for case in cases
    ]


def matched_compute_manifest(
    *,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    prompt_config: dict[str, Any],
    max_length: int,
    fixed_sequence_padding: bool = False,
) -> dict[str, Any]:
    """Prove equal forward and fixed-padding token budgets across conditions."""
    summaries = {}
    for condition in TRAINING_CONDITIONS:
        pairs = build_training_pairs(
            tokenizer=tokenizer,
            cases=cases,
            prompt_config=prompt_config,
            condition=condition,
            max_length=max_length,
            fixed_sequence_padding=fixed_sequence_padding,
        )
        sequences = [sequence for pair in pairs for sequence in pair]
        summaries[condition] = {
            "semantic_programs": len(pairs),
            "forward_passes": len(sequences),
            "fixed_padding_compute_tokens": len(sequences) * max_length,
            "active_input_tokens": sum(len(row["input_ids"]) for row in sequences),
            "max_active_sequence_length": max(
                (len(row["input_ids"]) for row in sequences), default=0
            ),
            "target_tokens": sum(
                sum(label != -100 for label in row["labels"]) for row in sequences
            ),
        }
    left, right = (summaries[name] for name in TRAINING_CONDITIONS)
    matched = all(
        left[key] == right[key]
        for key in (
            "semantic_programs",
            "forward_passes",
            "fixed_padding_compute_tokens",
            "active_input_tokens",
        )
    )
    return {
        "schema_version": 1,
        "max_length": max_length,
        "fixed_sequence_padding": fixed_sequence_padding,
        "conditions": summaries,
        "matched_forward_passes_and_tokens": matched,
        "matched_forward_passes_and_compute_tokens": matched,
        "token_matching_contract": (
            "Loss-masked control tokens follow the only supervised target and cannot "
            "affect it under causal attention."
        ),
    }


def validate_state_handoff_training_data(run_path: Path) -> dict[str, Any]:
    """Validate tokenizer, length, and matched-compute contracts before GPU use."""
    conditions = configured_training_conditions(run_path)
    if all(is_interface_condition(condition) for condition in conditions):
        from .state_interface_data import validate_state_interface_training_data

        return validate_state_interface_training_data(run_path)
    if not set(conditions).issubset(TRAINING_CONDITIONS):
        raise ValueError("A run cannot mix terminal and code-interface conditions")
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    max_length = int(experiment.get("training", {}).get("max_sequence_length", 256))
    evaluation_max_length = int(
        experiment.get("evaluation", {}).get(
            "max_sequence_length", max_length
        )
    )
    fixed_sequence_padding = bool(
        experiment.get("training", {}).get("fixed_sequence_padding", False)
    )
    tokenizer = load_hf_tokenizer(config["model"])
    prompt = experiment.get("prompt", {})
    train = read_programs(run_path / TRAIN_PATH)
    validation = read_programs(run_path / VALIDATION_PATH)
    test = read_programs(run_path / TEST_PATH)
    compute = matched_compute_manifest(
        tokenizer=tokenizer,
        cases=train,
        prompt_config=prompt,
        max_length=max_length,
        fixed_sequence_padding=fixed_sequence_padding,
    )
    validation_compute = matched_compute_manifest(
        tokenizer=tokenizer,
        cases=validation,
        prompt_config=prompt,
        max_length=max_length,
        fixed_sequence_padding=fixed_sequence_padding,
    )
    if not compute["matched_forward_passes_and_tokens"]:
        raise ValueError("Pilot training compute is not matched")
    test_max = 0
    alphabet_validated = False
    for case in test:
        prompts = {
            str(row["name"]): row
            for row in render_factorization_prompts(
                tokenizer=tokenizer, case=case, config=prompt
            )
        }
        update = render_factorization_update_prompt(
            tokenizer=tokenizer,
            case=case,
            config=prompt,
            state=int(case["current_state"]),
            rule=case["final_rule"],
            name="gold_handoff",
            label="FINAL",
        )
        for row in (prompts["compose"], prompts["synthesize"], update):
            if not alphabet_validated:
                candidates = (
                    answer_symbols(case)
                    if row.get("output_kind") == "answer"
                    else state_symbols(case)
                )
                candidate_token_ids(tokenizer, row["text"], candidates)
                if row["name"] == "synthesize":
                    alphabet_validated = True
            test_max = max(
                test_max,
                len(tokenizer.encode(row["text"], add_special_tokens=False)) + 1,
            )
    if test_max > evaluation_max_length:
        raise ValueError(
            f"Evaluation sequence length {test_max} exceeds "
            f"{evaluation_max_length}"
        )
    compute["validation"] = validation_compute["conditions"]
    compute["test_max_active_sequence_length"] = test_max
    compute["evaluation_max_sequence_length"] = evaluation_max_length
    compute["validated"] = True
    write_json(run_path / COMPUTE_MANIFEST_PATH, compute)
    return compute
