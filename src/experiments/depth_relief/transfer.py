"""Cross-case causal transfer of a materialized state register."""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.models.introspection import get_decoder_layers

from .benchmark import candidate_token_ids, state_symbols
from .factorization import compose_capture_positions, render_factorization_prompts
from .hf import patched_logits
from .metrics import bootstrap_mean_ci
from .qualification import score_logits
from .routing import render_routing_prompts


CONDITIONS = ("compose", "materialized", "counterfactual")
PATCH_MODES = (
    "state_gold",
    "state_counterfactual",
    "full_gold",
    "random_gold",
    "random_counterfactual",
)


def render_transfer_prompts(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reuse the exact discovery Compose and confirmed routing prompts."""
    shared_config = {"prompt": config.get("prompt", {})}
    compose = next(
        prompt
        for prompt in render_factorization_prompts(
            tokenizer=tokenizer,
            case=case,
            config=shared_config,
        )
        if prompt["name"] == "compose"
    )
    routing = render_routing_prompts(
        tokenizer=tokenizer,
        case=case,
        config=shared_config,
    )
    return [
        {
            "name": "compose",
            "text": compose["text"],
            "expected_next_state": int(case["next_state"]),
            "register_state": None,
        },
        {
            **routing[0],
            "register_state": int(case["current_state"]),
        },
        {
            **routing[1],
            "register_state": int(case["counterfactual_state"]),
        },
    ]


def validate_transfer_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Require one-token outputs and a stable final answer anchor."""
    prompts = render_transfer_prompts(tokenizer=tokenizer, case=case, config=config)
    token_rows = [
        tokenizer.encode(prompt["text"], add_special_tokens=False)
        for prompt in prompts
    ]
    for prompt in prompts:
        candidate_token_ids(tokenizer, prompt["text"], state_symbols(case))
    anchor_ids = [tokens[-1] for tokens in token_rows]
    if len(set(anchor_ids)) != 1:
        raise ValueError("Transfer prompts do not share the same final answer anchor")
    compose_positions = compose_capture_positions(
        tokenizer=tokenizer,
        case=case,
        text=prompts[0]["text"],
    )
    return {
        "id": case["id"],
        "condition_count": len(prompts),
        "token_count_range": [min(map(len, token_rows)), max(map(len, token_rows))],
        "answer_anchor_token_ids": anchor_ids,
        "compose_positions": compose_positions,
    }


def capture_prompt_states_hf(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    candidate_ids: list[int],
    token_indices: tuple[int, ...],
) -> tuple[dict[str, Any], np.ndarray]:
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    states: dict[int, torch.Tensor] = {}

    def capture(index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            states[index] = hidden[0, list(token_indices)].detach().to(torch.float16).cpu()
        return hook

    layers = get_decoder_layers(model)
    with ExitStack() as stack, torch.inference_mode():
        for index, layer in enumerate(layers):
            stack.callback(layer.register_forward_hook(capture(index)).remove)
        output = model(**encoded, use_cache=False, return_dict=True)
    if len(states) != len(layers):
        raise RuntimeError("Did not capture every decoder layer")
    return (
        score_logits(output.logits[0, -1].float().cpu().numpy(), candidate_ids),
        np.stack([states[index].numpy() for index in range(len(layers))]).transpose(
            1, 0, 2
        ),
    )


def capture_transfer_case_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Score exact Compose/routing prompts and capture their answer anchors."""
    prompts = render_transfer_prompts(tokenizer=tokenizer, case=case, config=config)
    validation = validate_transfer_case(tokenizer=tokenizer, case=case, config=config)
    conditions: dict[str, dict[str, Any]] = {}
    activations: dict[str, np.ndarray] = {}
    compose_positions = validation["compose_positions"]
    for prompt in prompts:
        token_indices = (
            tuple(int(row["token_index"]) for row in compose_positions)
            if prompt["name"] == "compose"
            else (-1,)
        )
        record, states = capture_prompt_states_hf(
            model=model,
            tokenizer=tokenizer,
            text=prompt["text"],
            candidate_ids=candidate_token_ids(
                tokenizer, prompt["text"], state_symbols(case)
            ),
            token_indices=token_indices,
        )
        expected = int(prompt["expected_next_state"])
        record.update(
            expected_next_state=expected,
            is_expected_unconstrained=record["unconstrained_prediction"] == expected,
            register_state=prompt["register_state"],
        )
        conditions[str(prompt["name"])] = record
        activations[str(prompt["name"])] = states[-1]
        if prompt["name"] == "compose":
            activations["compose_trace"] = states
    row = {
        "schema_version": 1,
        "id": case["id"],
        "format": case["format"],
        "bits": int(case["bits"]),
        "answer_anchor_token_ids": validation["answer_anchor_token_ids"],
        "compose_positions": compose_positions,
        "current_state": int(case["current_state"]),
        "counterfactual_state": int(case["counterfactual_state"]),
        "next_state": int(case["next_state"]),
        "counterfactual_next_state": int(case["counterfactual_next_state"]),
        "conditions": conditions,
    }
    return row, activations


def _stable_order(cases: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        cases,
        key=lambda case: hashlib.sha256(f"{seed}:{case['id']}".encode()).hexdigest(),
    )


def _donor(
    recipient: dict[str, Any], state: int, target: int, train: list[dict[str, Any]]
) -> dict[str, str] | None:
    for case in train:
        if case["format"] != recipient["format"] or case["id"] == recipient["id"]:
            continue
        for condition, state_key, target_key in (
            ("materialized", "current_state", "next_state"),
            ("counterfactual", "counterfactual_state", "counterfactual_next_state"),
        ):
            if int(case[state_key]) == state and int(case[target_key]) != target:
                return {"case_id": str(case["id"]), "condition": condition}
    return None


def build_transfer_split(
    cases: list[dict[str, Any]], *, seed: int = 47, train_fraction: float = 0.55
) -> dict[str, Any]:
    """Make train/validation/test splits with answer-disjoint state donors."""
    ordered = _stable_order(cases, seed)
    train_count = max(1, round(len(ordered) * train_fraction))
    train = ordered[:train_count]
    evaluation = ordered[train_count:]
    while True:
        missing = [
            case for case in evaluation
            if _donor(case, int(case["current_state"]), int(case["next_state"]), train) is None
            or _donor(
                case,
                int(case["counterfactual_state"]),
                int(case["counterfactual_next_state"]),
                train,
            ) is None
        ]
        if not missing:
            break
        train.extend(missing)
        missing_ids = {case["id"] for case in missing}
        evaluation = [case for case in evaluation if case["id"] not in missing_ids]
    midpoint = len(evaluation) // 2
    validation, test = evaluation[:midpoint], evaluation[midpoint:]
    if min(len(validation), len(test)) < 15:
        raise ValueError("Transfer split has fewer than 15 validation or test cases")
    donors = {}
    for case in evaluation:
        gold = _donor(case, int(case["current_state"]), int(case["next_state"]), train)
        counterfactual = _donor(
            case,
            int(case["counterfactual_state"]),
            int(case["counterfactual_next_state"]),
            train,
        )
        if gold is None or counterfactual is None:
            raise AssertionError("Missing donor after split construction")
        donors[str(case["id"])] = {"gold": gold, "counterfactual": counterfactual}
    return {
        "schema_version": 1,
        "seed": seed,
        "train": [str(case["id"]) for case in train],
        "validation": [str(case["id"]) for case in validation],
        "test": [str(case["id"]) for case in test],
        "donors": donors,
    }


def fit_state_subspaces(
    *, cases: dict[str, dict[str, Any]], split: dict[str, Any], activation_dir: Path, rank: int, seed: int
) -> dict[str, np.ndarray]:
    """Fit state directions from paired explicit-minus-Compose anchor deltas."""
    grouped: dict[int, list[np.ndarray]] = {}
    for case_id in split["train"]:
        case = cases[case_id]
        arrays = np.load(activation_dir / f"{case_id}.npz")
        for condition, state_key in (
            ("materialized", "current_state"),
            ("counterfactual", "counterfactual_state"),
        ):
            grouped.setdefault(int(case[state_key]), []).append(
                arrays[condition].astype(np.float32) - arrays["compose"].astype(np.float32)
            )
    if len(grouped) != 2 ** int(next(iter(cases.values()))["bits"]):
        raise ValueError("Training split does not cover every register state")
    centroids = np.stack([np.mean(grouped[state], axis=0) for state in sorted(grouped)])
    layer_count, hidden_size = centroids.shape[1:]
    # The uncentered span retains both state identity (at most seven affine
    # directions for eight values) and the shared materialization direction
    # that distinguishes a usable materialized state from implicit composition.
    fitted_rank = min(rank, len(grouped), hidden_size)
    bases, random_bases = [], []
    rng = np.random.default_rng(seed)
    for layer in range(layer_count):
        _, _, vt = np.linalg.svd(centroids[:, layer, :], full_matrices=False)
        basis = vt[:fitted_rank].T
        random_basis, _ = np.linalg.qr(rng.normal(size=(hidden_size, fitted_rank)))
        bases.append(basis.astype(np.float32))
        random_bases.append(random_basis.astype(np.float32))
    return {
        "state_basis": np.stack(bases),
        "random_basis": np.stack(random_bases),
        "rank": np.asarray(fitted_rank),
    }


def score_transfer_patches_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any], layer: int,
    recipient: np.ndarray, gold_delta: np.ndarray, counterfactual_delta: np.ndarray,
    state_basis: np.ndarray, random_basis: np.ndarray,
) -> dict[str, Any]:
    """Patch cross-case full, state-subspace, and random-subspace deltas."""
    prompt = render_transfer_prompts(tokenizer=tokenizer, case=case, config=config)[0]
    candidate_ids = candidate_token_ids(tokenizer, prompt["text"], state_symbols(case))
    vectors = {
        "state_gold": state_basis @ (state_basis.T @ gold_delta),
        "state_counterfactual": state_basis @ (state_basis.T @ counterfactual_delta),
        "full_gold": gold_delta,
        "random_gold": random_basis @ (random_basis.T @ gold_delta),
        "random_counterfactual": random_basis @ (random_basis.T @ counterfactual_delta),
    }
    expected = {
        "state_gold": int(case["next_state"]),
        "state_counterfactual": int(case["counterfactual_next_state"]),
        "full_gold": int(case["next_state"]),
        "random_gold": int(case["next_state"]),
        "random_counterfactual": int(case["counterfactual_next_state"]),
    }
    conditions = {}
    for mode in PATCH_MODES:
        state = torch.from_numpy((recipient + vectors[mode]).astype(np.float32))[None, :]
        logits = patched_logits(
            model=model,
            tokenizer=tokenizer,
            text=prompt["text"],
            patches={layer: ((-1,), state)},
        )
        record = score_logits(logits, candidate_ids)
        target = expected[mode]
        record.update(
            expected_next_state=target,
            is_expected_unconstrained=record["unconstrained_prediction"] == target,
        )
        conditions[mode] = record
    return {"schema_version": 1, "id": case["id"], "layer": layer, "conditions": conditions}


def summarize_transfer(
    *, cases: dict[str, dict[str, Any]], split: dict[str, Any], captures: dict[str, dict[str, Any]],
    patches: list[dict[str, Any]], gate: dict[str, Any]
) -> dict[str, Any]:
    """Select a layer on validation and report its held-out causal effect."""
    indexed = {(str(row["id"]), int(row["layer"])): row for row in patches}
    layers = sorted({int(row["layer"]) for row in patches})

    def target_probability(record: dict[str, Any], target: int) -> float:
        return float(record["final_candidate_probabilities"][target])

    def shifts(case_id: str, layer: int, mode: str) -> float:
        case = cases[case_id]
        target = int(
            case["counterfactual_next_state"]
            if mode in {"state_counterfactual", "random_counterfactual"}
            else case["next_state"]
        )
        patched = indexed[(case_id, layer)]["conditions"][mode]
        baseline = captures[case_id]["conditions"]["compose"]
        return target_probability(patched, target) - target_probability(baseline, target)

    validation_scores = {
        layer: float(np.mean([
            0.5 * (
                shifts(case_id, layer, "state_gold")
                + shifts(case_id, layer, "state_counterfactual")
                - shifts(case_id, layer, "random_gold")
                - shifts(case_id, layer, "random_counterfactual")
            )
            for case_id in split["validation"]
        ]))
        for layer in layers
    }
    selected = max(layers, key=lambda layer: (validation_scores[layer], -layer))
    heldout = split["test"]
    metrics = {}
    for mode_index, mode in enumerate(PATCH_MODES):
        metrics[mode] = {
            "target_probability_shift": bootstrap_mean_ci(
                [shifts(case_id, selected, mode) for case_id in heldout],
                seed=1000 + mode_index,
            ),
            "accuracy": bootstrap_mean_ci(
                [indexed[(case_id, selected)]["conditions"][mode]["is_expected_unconstrained"] for case_id in heldout],
                seed=1010 + mode_index,
            ),
        }
    gold_over_random = bootstrap_mean_ci(
        [shifts(case_id, selected, "state_gold") - shifts(case_id, selected, "random_gold") for case_id in heldout],
        seed=1020,
    )
    cf_over_random = bootstrap_mean_ci(
        [shifts(case_id, selected, "state_counterfactual") - shifts(case_id, selected, "random_counterfactual") for case_id in heldout],
        seed=1021,
    )
    thresholds = {
        "min_state_shift_lower": 0.10,
        "min_state_over_random_lower": 0.10,
        **gate,
    }
    def lower(stat: dict[str, Any]) -> float:
        return float(stat["ci95"][0])

    checks = {
        "gold_state_shift": lower(metrics["state_gold"]["target_probability_shift"]) >= float(thresholds["min_state_shift_lower"]),
        "counterfactual_state_shift": lower(metrics["state_counterfactual"]["target_probability_shift"]) >= float(thresholds["min_state_shift_lower"]),
        "gold_over_random": lower(gold_over_random) >= float(thresholds["min_state_over_random_lower"]),
        "counterfactual_over_random": lower(cf_over_random) >= float(thresholds["min_state_over_random_lower"]),
    }
    return {
        "schema_version": 1,
        "split_counts": {name: len(split[name]) for name in ("train", "validation", "test")},
        "layer_selection": {"selected": selected, "validation_scores": validation_scores},
        "heldout": {"case_count": len(heldout), "metrics": metrics, "gold_over_random": gold_over_random, "counterfactual_over_random": cf_over_random},
        "gate": {"thresholds": thresholds, "checks": checks, "passed": all(checks.values())},
    }
