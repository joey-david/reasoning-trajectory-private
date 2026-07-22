"""Deterministic program splits and masked one-token training sequences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl

from .abstraction import matched_addition_history
from .benchmark import apply_rule, candidate_token_ids, state_symbols, state_text
from .explicit_handoff import discrete_capacity_bits
from .factorization import (
    render_factorization_prompts,
    render_factorization_update_prompt,
)


TRAIN_PATH = Path("training/data/train_programs.jsonl")
VALIDATION_PATH = Path("training/data/validation_programs.jsonl")
TEST_PATH = Path("evaluation/test_programs.jsonl")
DATA_MANIFEST_PATH = Path("training/data/manifest.json")
COMPUTE_MANIFEST_PATH = Path("training/compute_manifest.json")
TRAINING_CONDITIONS = ("outcome_only", "explicit_handoff")


def _state_path(
    initial: int, history: list[dict[str, Any]], modulus: int
) -> list[int]:
    values = [initial]
    for rule in history:
        values.append(apply_rule(rule, values[-1], modulus))
    return values


def _program_contexts(
    *, split: str, count: int, width: int, seed: int
) -> list[dict[str, Any]]:
    modulus = 2**width
    contexts = []
    signatures: set[tuple[int, tuple[int, ...]]] = set()
    candidate = 0
    while len(contexts) < count:
        rng = random.Random(
            seed + 1_000_003 * (sum(map(ord, split)) + 1) + 10_007 * candidate
        )
        candidate += 1
        initial = rng.randrange(modulus)
        mapping = list(range(modulus))
        rng.shuffle(mapping)
        signature = (initial, tuple(mapping))
        if signature in signatures:
            continue
        signatures.add(signature)
        contexts.append(
            {
                "id": f"{split}_c{len(contexts):03d}",
                "index": len(contexts),
                "initial_state": initial,
                "final_rule": {"kind": "pointer", "mapping": mapping},
            }
        )
    return contexts


def _program_case(
    *,
    split: str,
    context: dict[str, Any],
    horizon: int,
    target: int,
    path_code: int,
    width: int,
) -> dict[str, Any]:
    modulus = 2**width
    initial = int(context["initial_state"])
    history = matched_addition_history(
        initial=initial,
        target=target,
        path_code=path_code,
        history_steps=horizon,
        group_index=int(context["index"]),
        modulus=modulus,
    )
    states = _state_path(initial, history, modulus)
    if states[-1] != target:
        raise AssertionError("Training history missed its requested state")
    final_rule = context["final_rule"]
    semantic = {
        "family": "add_to_pointer",
        "history_family": "add",
        "final_family": "pointer",
        "format": "prose",
        "bits": width,
        "initial_state": initial,
        "history": history,
        "final_rule": final_rule,
        "current_state": target,
        "next_state": apply_rule(final_rule, target, modulus),
        "history_steps": horizon,
        "state_path": states,
        "path_code": path_code,
        "program_context": str(context["id"]),
        "program_context_split": split,
        "abstraction_group": str(context["id"]),
        "abstraction_split": split,
    }
    digest = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    semantic["id"] = (
        f"handoff_{split}_{context['id']}_h{horizon}_s{target}_"
        f"p{path_code}_{digest}"
    )
    return semantic


def _balanced_programs(
    *,
    split: str,
    semantic_count: int,
    horizons: tuple[int, ...],
    context_count: int,
    width: int,
    seed: int,
) -> list[dict[str, Any]]:
    if semantic_count < context_count * len(horizons) * 2**width:
        raise ValueError(f"{split} is too small to cover every context-state cell")
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    cells = [
        (context, horizon, target)
        for context in contexts
        for horizon in horizons
        for target in range(2**width)
    ]
    rows = []
    for index in range(semantic_count):
        context, horizon, target = cells[index % len(cells)]
        path_code = index // len(cells)
        rows.append(
            _program_case(
                split=split,
                context=context,
                horizon=horizon,
                target=target,
                path_code=path_code,
                width=width,
            )
        )
    return rows


def build_test_programs(
    *,
    horizons: tuple[int, ...],
    context_count: int,
    paths_per_state: int,
    width: int,
    seed: int,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Build a balanced fixed test bank for any named artifact split."""
    contexts = _program_contexts(
        split=split, count=context_count, width=width, seed=seed
    )
    return [
        _program_case(
            split=split,
            context=context,
            horizon=horizon,
            target=target,
            path_code=path,
            width=width,
        )
        for context in contexts
        for horizon in horizons
        for target in range(2**width)
        for path in range(paths_per_state)
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "semantic_program_count": len(rows),
        "program_context_count": len({row["program_context"] for row in rows}),
        "horizons": sorted({int(row["history_steps"]) for row in rows}),
        "states": sorted({int(row["current_state"]) for row in rows}),
        "minimum_histories_per_state": min(
            sum(int(row["current_state"]) == state for row in rows)
            for state in range(8)
        ),
    }


def prepare_state_handoff_datasets(run_path: Path) -> dict[str, Any]:
    """Write group-disjoint pilot programs and their stable hashes."""
    config = load_config(run_path).get("state_handoff_training", {})
    dataset = config.get("dataset", {})
    width = int(dataset.get("bits", 3))
    if discrete_capacity_bits(slots=1, codebook_size=2**width) != 3:
        raise ValueError("The first state-handoff training pass must use three bits")
    sequences_per_condition = int(dataset.get("train_examples", 20_000))
    validation_sequences = int(dataset.get("validation_examples", 2_000))
    if sequences_per_condition % 2 or validation_sequences % 2:
        raise ValueError("Matched two-call training example counts must be even")
    seed = int(dataset.get("seed", 721_301))
    train = _balanced_programs(
        split="train",
        semantic_count=sequences_per_condition // 2,
        horizons=tuple(int(value) for value in dataset.get("train_horizons", (1, 2))),
        context_count=int(dataset.get("train_program_contexts", 50)),
        width=width,
        seed=seed,
    )
    validation = _balanced_programs(
        split="validation",
        semantic_count=validation_sequences // 2,
        horizons=tuple(
            int(value) for value in dataset.get("validation_horizons", (1, 2))
        ),
        context_count=int(dataset.get("validation_program_contexts", 15)),
        width=width,
        seed=seed,
    )
    test = build_test_programs(
        horizons=tuple(int(value) for value in dataset.get("test_horizons", (2, 4, 8))),
        context_count=int(dataset.get("test_program_contexts", 30)),
        paths_per_state=int(dataset.get("test_paths_per_state", 4)),
        width=width,
        seed=seed,
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
        "capacity_bits": 3,
        "training_sequences_per_condition": 2 * len(train),
        "validation_sequences_per_condition": 2 * len(validation),
        "splits": {
            split: {**_split_summary(rows), "sha256": _sha256(paths[split])}
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
    symbols = state_symbols(case)
    target_id = candidate_token_ids(tokenizer, prompt["text"], symbols)[target]
    prompt_ids = tokenizer.encode(prompt["text"], add_special_tokens=False)
    full_ids = tokenizer.encode(
        prompt["text"] + state_text(case, target), add_special_tokens=False
    )
    if full_ids != [*prompt_ids, target_id]:
        raise ValueError("Training target does not extend the prompt by one token")
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
    target_tokens = max(
        sum(len(row["input_ids"]) for row in pair) for pair in pairs.values()
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
) -> list[list[dict[str, Any]]]:
    """Tokenize semantic programs into stable two-forward training pairs."""
    return [
        training_sequence_pair(
            tokenizer=tokenizer,
            case=case,
            prompt_config=prompt_config,
            condition=condition,
            max_length=max_length,
        )
        for case in cases
    ]


def matched_compute_manifest(
    *,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    prompt_config: dict[str, Any],
    max_length: int,
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
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("state_handoff_training", {})
    max_length = int(experiment.get("training", {}).get("max_sequence_length", 256))
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
    )
    validation_compute = matched_compute_manifest(
        tokenizer=tokenizer,
        cases=validation,
        prompt_config=prompt,
        max_length=max_length,
    )
    if not compute["matched_forward_passes_and_tokens"]:
        raise ValueError("Pilot training compute is not matched")
    test_max = 0
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
            candidate_token_ids(tokenizer, row["text"], state_symbols(case))
            test_max = max(
                test_max,
                len(tokenizer.encode(row["text"], add_special_tokens=False)) + 1,
            )
    if test_max > max_length:
        raise ValueError(f"Evaluation sequence length {test_max} exceeds {max_length}")
    compute["validation"] = validation_compute["conditions"]
    compute["test_max_active_sequence_length"] = test_max
    compute["validated"] = True
    write_json(run_path / COMPUTE_MANIFEST_PATH, compute)
    return compute
