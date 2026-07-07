"""Type-corrected residual patching and matched subspace ablation."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from src.models.activation_capture import SelectedLayerCapture
from src.models.introspection import get_decoder_layers, resolve_layer_indices


def swap_subspace(
    target: np.ndarray,
    donor: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Replace target coordinates in an orthonormal row-space with donor values."""
    delta = np.asarray(donor, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    return np.asarray(target, dtype=np.float32) + (delta @ basis.T) @ basis


def ablate_subspace(
    state: np.ndarray,
    mean: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Remove centered coordinates in an orthonormal row-space."""
    centered = np.asarray(state, dtype=np.float32) - mean
    return np.asarray(state, dtype=np.float32) - (centered @ basis.T) @ basis


def select_causal_pairs(
    records: list[dict[str, Any]],
    *,
    max_pairs_per_condition: int,
) -> list[dict[str, Any]]:
    """Select deterministic pairs spanning all plan-prescribed contrasts."""
    candidates = [
        row
        for row in records
        if row["edit_type"] == "OPERATE"
        and row["is_correct"]
        and row["split"] in {"validation", "heldout_vocab", "heldout_template"}
    ]
    pairs: list[dict[str, Any]] = []
    used: defaultdict[str, int] = defaultdict(int)
    for target_index, target in enumerate(candidates):
        for donor in candidates[target_index + 1 :]:
            condition = pair_condition(target, donor)
            if condition is None or used[condition] >= max_pairs_per_condition:
                continue
            pairs.append(
                {
                    "pair_id": len(pairs),
                    "condition": condition,
                    "target": target,
                    "donor": donor,
                }
            )
            used[condition] += 1
    return pairs


def pair_condition(
    target: dict[str, Any], donor: dict[str, Any]
) -> str | None:
    """Classify one donor-target pair by graph and surface controls."""
    same_graph = target["canonical_graph_id"] == donor["canonical_graph_id"]
    same_template = (
        target["surface"]["template_id"] == donor["surface"]["template_id"]
    )
    same_vocab = (
        target["surface"]["lexical_family"]
        == donor["surface"]["lexical_family"]
    )
    same_answer = target["causal_result"] == donor["causal_result"]
    if same_graph and not same_vocab:
        return "same_object_different_wording"
    if not same_graph and same_answer and not same_vocab:
        return "different_object_same_answer"
    if not same_graph and same_template:
        return "different_object_same_template"
    if not same_graph and not same_vocab:
        return "different_object_different_wording"
    return None


def causal_reports(
    *,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    layer: int,
    object_mean: np.ndarray,
    object_basis: np.ndarray,
    random_mean: np.ndarray,
    random_basis: np.ndarray,
    lexical_mean: np.ndarray,
    lexical_basis: np.ndarray,
    max_pairs_per_condition: int,
    continuation_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Run small type-corrected patching and ablation experiments."""
    pairs = select_causal_pairs(
        records, max_pairs_per_condition=max_pairs_per_condition
    )
    if not pairs:
        raise ValueError("No causal pairs satisfy the configured contrasts")
    decoder_layers = get_decoder_layers(model)
    resolved = resolve_layer_indices([layer], len(decoder_layers))[0]
    cache: dict[str, tuple[np.ndarray, torch.Tensor]] = {}

    def cached(row: dict[str, Any]) -> tuple[np.ndarray, torch.Tensor]:
        key = str(row["record_id"])
        if key not in cache:
            cache[key] = forward_state_and_logits(
                model, tokenizer, row["causal_prefix"], layer
            )
        return cache[key]

    detail_rows: list[dict[str, Any]] = []
    modes = {
        "object_subspace": (object_mean, object_basis),
        "random_subspace": (random_mean, random_basis),
        "lexical_subspace": (lexical_mean, lexical_basis),
    }
    for pair in tqdm(pairs, desc="causal patching", unit="pair"):
        target_state, baseline_logits = cached(pair["target"])
        donor_state, _ = cached(pair["donor"])
        target_token = answer_token(tokenizer, pair["target"]["causal_result"])
        donor_token = answer_token(tokenizer, pair["donor"]["causal_result"])
        interventions = {
            "full_vector": donor_state,
            **{
                mode: swap_subspace(target_state, donor_state, basis)
                for mode, (_mean, basis) in modes.items()
            },
        }
        for mode, patched_state in interventions.items():
            patched_logits = forward_with_patch(
                model,
                tokenizer,
                pair["target"]["causal_prefix"],
                resolved,
                patched_state,
            )
            continuation = (
                generate_with_patch(
                    model,
                    tokenizer,
                    pair["target"]["causal_prefix"],
                    resolved,
                    patched_state,
                    max_new_tokens=continuation_tokens,
                )
                if continuation_tokens > 0
                else None
            )
            detail_rows.append(
                patch_detail(
                    pair,
                    mode,
                    baseline_logits,
                    patched_logits,
                    target_token,
                    donor_token,
                    continuation,
                )
            )

    ablation_rows: list[dict[str, Any]] = []
    ablation_candidates = [
        row
        for row in records
        if row["edit_type"] == "OPERATE"
        and row["is_correct"]
        and row["split"] in {"validation", "heldout_vocab", "heldout_template"}
    ][: max_pairs_per_condition * 3]
    for row in tqdm(ablation_candidates, desc="subspace ablation", unit="prompt"):
        state, baseline_logits = cached(row)
        correct_token = answer_token(tokenizer, row["causal_result"])
        for mode, (mean, basis) in modes.items():
            ablated = ablate_subspace(state, mean, basis)
            logits = forward_with_patch(
                model, tokenizer, row["causal_prefix"], resolved, ablated
            )
            baseline_probability = token_probability(baseline_logits, correct_token)
            probability = token_probability(logits, correct_token)
            ablation_rows.append(
                {
                    "record_id": row["record_id"],
                    "mode": mode,
                    "baseline_correct_probability": baseline_probability,
                    "ablated_correct_probability": probability,
                    "correct_probability_change": probability
                    - baseline_probability,
                    "baseline_entropy": distribution_entropy(baseline_logits),
                    "ablated_entropy": distribution_entropy(logits),
                }
            )
    return (
        summarize_patching(detail_rows, layer, len(pairs)),
        summarize_ablation(ablation_rows, layer),
        detail_rows,
    )


def forward_state_and_logits(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Capture one layer's final-token state and unpatched next-token logits."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = get_decoder_layers(model)
    resolved = resolve_layer_indices([layer], len(layers))
    with (
        torch.inference_mode(),
        SelectedLayerCapture(
            decoder_layers=layers,
            requested_layers=[layer],
            resolved_layers=resolved,
        ) as capture,
    ):
        output = model(**encoded, use_cache=False, return_dict=True)
    state = capture.outputs[layer][0, -1].float().cpu().numpy()
    return state, output.logits[0, -1].float().cpu()


def forward_with_patch(
    model: Any,
    tokenizer: Any,
    text: str,
    resolved_layer: int,
    patched_state: np.ndarray,
) -> torch.Tensor:
    """Patch one decoder-block output at the final prompt position."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layer = get_decoder_layers(model)[resolved_layer]

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        replaced = hidden.clone()
        replaced[0, -1] = torch.as_tensor(
            patched_state, device=hidden.device, dtype=hidden.dtype
        )
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
    return output.logits[0, -1].float().cpu()


def generate_with_patch(
    model: Any,
    tokenizer: Any,
    text: str,
    resolved_layer: int,
    patched_state: np.ndarray,
    *,
    max_new_tokens: int,
) -> str:
    """Greedily continue after patching only the initial prompt forward pass."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = int(encoded["input_ids"].shape[1])
    layer = get_decoder_layers(model)[resolved_layer]
    applied = False

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        nonlocal applied
        if applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        replaced = hidden.clone()
        replaced[0, -1] = torch.as_tensor(
            patched_state, device=hidden.device, dtype=hidden.dtype
        )
        applied = True
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    handle = layer.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    return tokenizer.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def answer_token(tokenizer: Any, value: int | float) -> int:
    """Return the first numeric token after the prefix's trailing space."""
    text = str(int(value)) if float(value).is_integer() else str(value)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError(f"Could not tokenize answer {text!r}")
    return int(tokens[0])


def token_probability(logits: torch.Tensor, token: int) -> float:
    """Return one token probability from logits."""
    return float(torch.softmax(logits, dim=-1)[token])


def distribution_entropy(logits: torch.Tensor) -> float:
    """Return categorical entropy in nats."""
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    return float(-(probabilities * log_probabilities).sum())


def patch_detail(
    pair: dict[str, Any],
    mode: str,
    baseline: torch.Tensor,
    patched: torch.Tensor,
    target_token: int,
    donor_token: int,
    continuation: str | None,
) -> dict[str, Any]:
    """Build one type-corrected patch result."""
    target_before = token_probability(baseline, target_token)
    donor_before = token_probability(baseline, donor_token)
    target_after = token_probability(patched, target_token)
    donor_after = token_probability(patched, donor_token)
    donor_terms = {
        str(pair["donor"]["surface"]["container_term"]).lower(),
        str(pair["donor"]["surface"]["item_term"]).lower(),
    }
    target_terms = {
        str(pair["target"]["surface"]["container_term"]).lower(),
        str(pair["target"]["surface"]["item_term"]).lower(),
    }
    continuation_lower = (continuation or "").lower()
    parsed_answer = parse_final_number(continuation)
    return {
        "pair_id": pair["pair_id"],
        "condition": pair["condition"],
        "mode": mode,
        "target_record_id": pair["target"]["record_id"],
        "donor_record_id": pair["donor"]["record_id"],
        "same_answer": target_token == donor_token,
        "target_probability_change": target_after - target_before,
        "type_corrected_donor_probability_change": donor_after - donor_before,
        "donor_beats_target_after": (
            donor_after > target_after if target_token != donor_token else None
        ),
        "entropy_change": distribution_entropy(patched)
        - distribution_entropy(baseline),
        "continuation": continuation,
        "continuation_answer": parsed_answer,
        "object_consistent_continuation": (
            parsed_answer == float(pair["donor"]["causal_result"])
            if parsed_answer is not None
            else False
        ),
        "target_consistent_continuation": (
            parsed_answer == float(pair["target"]["causal_result"])
            if parsed_answer is not None
            else False
        ),
        "donor_lexical_leakage": any(
            term in continuation_lower for term in donor_terms - target_terms
        ),
        "target_surface_mention": any(
            term in continuation_lower for term in target_terms
        ),
    }


def parse_final_number(text: str | None) -> float | None:
    """Parse the final numeric value in a short continuation."""
    if not text:
        return None
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(matches[-1]) if matches else None


def summarize_patching(
    rows: list[dict[str, Any]], layer: int, pair_count: int
) -> dict[str, Any]:
    """Aggregate patch effects by mode and condition."""
    cells: list[dict[str, Any]] = []
    keys = sorted({(row["mode"], row["condition"]) for row in rows})
    for mode, condition in keys:
        selected = [
            row
            for row in rows
            if row["mode"] == mode and row["condition"] == condition
        ]
        shifts = [
            row["donor_beats_target_after"]
            for row in selected
            if row["donor_beats_target_after"] is not None
        ]
        cells.append(
            {
                "mode": mode,
                "condition": condition,
                "pairs": len(selected),
                "mean_type_corrected_donor_probability_change": float(
                    np.mean(
                        [
                            row["type_corrected_donor_probability_change"]
                            for row in selected
                        ]
                    )
                ),
                "mean_target_probability_change": float(
                    np.mean(
                        [row["target_probability_change"] for row in selected]
                    )
                ),
                "answer_shift_rate": float(np.mean(shifts)) if shifts else None,
                "mean_entropy_change": float(
                    np.mean([row["entropy_change"] for row in selected])
                ),
                "object_consistent_continuation_rate": float(
                    np.mean(
                        [
                            row["object_consistent_continuation"]
                            for row in selected
                        ]
                    )
                ),
                "target_consistent_continuation_rate": float(
                    np.mean(
                        [
                            row["target_consistent_continuation"]
                            for row in selected
                        ]
                    )
                ),
                "lexical_leakage_rate": float(
                    np.mean([row["donor_lexical_leakage"] for row in selected])
                ),
                "target_surface_mention_rate": float(
                    np.mean([row["target_surface_mention"] for row in selected])
                ),
            }
        )
    return {
        "protocol": "single-step type-corrected next-token causal mediation",
        "layer": layer,
        "pairs": pair_count,
        "cells": cells,
        "lexical_leakage_definition": (
            "donor-only controlled vocabulary term appears in the patched "
            "continuation"
        ),
    }


def summarize_ablation(rows: list[dict[str, Any]], layer: int) -> dict[str, Any]:
    """Aggregate ablation effects by subspace."""
    return {
        "protocol": "matched-rank final-position next-token ablation",
        "layer": layer,
        "cells": [
            {
                "mode": mode,
                "prompts": len(selected),
                "mean_correct_probability_change": float(
                    np.mean(
                        [row["correct_probability_change"] for row in selected]
                    )
                ),
                "mean_entropy_change": float(
                    np.mean(
                        [
                            row["ablated_entropy"] - row["baseline_entropy"]
                            for row in selected
                        ]
                    )
                ),
            }
            for mode in sorted({row["mode"] for row in rows})
            if (
                selected := [row for row in rows if row["mode"] == mode]
            )
        ],
    }
