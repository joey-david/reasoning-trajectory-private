"""Hugging Face depth screens and causal checkpoint interchange interventions."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models.activation_capture import project_hidden_state
from src.models.introspection import get_decoder_layers, get_final_norm, get_lm_head

from .benchmark import (
    candidate_token_ids,
    condition_specs,
    decimal_state_symbols,
    format_model_prompt,
    format_prompt_spec,
    render_prompt,
    render_write_prompt,
)
from .metrics import jensen_shannon, normalized_recovery, settling_depth, softmax


@dataclass(slots=True)
class PromptEvaluation:
    """Prompt metrics plus transient residual states used by interventions."""

    record: dict[str, Any]
    final_states: dict[int, torch.Tensor]
    checkpoint_states: dict[int, torch.Tensor]
    checkpoint_token_indices: tuple[int, ...]
    token_count: int


def checkpoint_token_indices(tokenizer: Any, spec: Any) -> tuple[int, ...]:
    """Map the checkpoint's exact character span to overlapping token indices."""
    encoded = tokenizer(
        spec.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer must expose offset mappings for causal alignment")
    indices = tuple(
        index
        for index, (start, end) in enumerate(offsets)
        if end > spec.checkpoint_start and start < spec.checkpoint_end
    )
    if not indices:
        raise ValueError("Checkpoint span did not align to any tokens")
    return indices


def _extract_hidden(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def evaluate_prompt(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    candidate_ids: list[int],
    threshold: float,
    checkpoint_indices: tuple[int, ...] = (),
) -> PromptEvaluation:
    """Capture token-position residuals and compute per-layer candidate distributions."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    token_count = int(encoded["input_ids"].shape[1])
    layers = get_decoder_layers(model)
    final_states: dict[int, torch.Tensor] = {}
    source_states: dict[int, torch.Tensor] = {}

    def capture(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = _extract_hidden(output)
            final_states[layer_index] = hidden[0, -1].detach().float().cpu()
            if checkpoint_indices:
                source_states[layer_index] = (
                    hidden[0, list(checkpoint_indices)].detach().float().cpu()
                )
        return hook

    with ExitStack() as stack, torch.inference_mode():
        for index, layer in enumerate(layers):
            stack.callback(layer.register_forward_hook(capture(index)).remove)
        output = model(**encoded, use_cache=False, return_dict=True)

    lm_head = get_lm_head(model)
    final_norm = get_final_norm(model)
    final_full = torch.softmax(output.logits[0, -1].float(), dim=-1).cpu().numpy()
    layer_candidates = []
    divergences = []
    for index in range(len(layers)):
        logits = project_hidden_state(
            final_states[index][None, :], lm_head=lm_head, final_norm=final_norm
        )[0].detach().cpu().numpy()
        layer_candidates.append(softmax(logits[candidate_ids]))
        divergences.append(jensen_shannon(softmax(logits), final_full))
    distributions = np.stack(layer_candidates)
    final_logits = output.logits[0, -1, candidate_ids].float().cpu().numpy()
    final_distribution = softmax(final_logits)
    full_top_token_id = int(output.logits[0, -1].argmax().item())
    unconstrained_prediction = (
        candidate_ids.index(full_top_token_id)
        if full_top_token_id in candidate_ids
        else None
    )
    # The model's real final head is the reference endpoint; replacing the last
    # logit-lens row avoids measuring harmless projection-stack discrepancies.
    distributions[-1] = final_distribution
    prediction = int(np.argmax(final_distribution))
    threshold_grid = sorted(
        {float(value) for value in [threshold, *[0.25, 0.5, 0.75]]}
    )
    return PromptEvaluation(
        record={
            "prediction": prediction,
            "unconstrained_prediction": unconstrained_prediction,
            "unconstrained_token_id": full_top_token_id,
            "candidate_probability_mass": float(final_full[candidate_ids].sum()),
            "settling_depth": settling_depth(np.asarray(divergences), threshold=threshold),
            "settling_depth_by_threshold": {
                str(value): settling_depth(np.asarray(divergences), threshold=value)
                for value in threshold_grid
            },
            "dtr_jsd": divergences,
            "dtr_jsd_auc": float(np.mean(divergences)),
            "candidate_probabilities": distributions.tolist(),
            "final_candidate_probabilities": final_distribution.tolist(),
            "token_count": token_count,
        },
        final_states=final_states,
        checkpoint_states=source_states,
        checkpoint_token_indices=checkpoint_indices,
        token_count=token_count,
    )


def patched_logits(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    patches: dict[int, tuple[tuple[int, ...], torch.Tensor]],
) -> np.ndarray:
    """Return final logits after exact residual replacements."""
    device = model.get_input_embeddings().weight.device
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    layers = get_decoder_layers(model)

    def patch(layer_index: int):
        positions, values = patches[layer_index]

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _extract_hidden(output)
            replaced = hidden.clone()
            replaced[0, list(positions)] = values.to(
                device=hidden.device, dtype=hidden.dtype
            )
            return _replace_hidden(output, replaced)
        return hook

    with ExitStack() as stack, torch.inference_mode():
        for index in sorted(patches):
            stack.callback(layers[index].register_forward_hook(patch(index)).remove)
        output = model(**encoded, use_cache=False, return_dict=True)
    return output.logits[0, -1].float().cpu().numpy()


def patched_probabilities(
    *,
    model: Any,
    tokenizer: Any,
    text: str,
    candidate_ids: list[int],
    patches: dict[int, tuple[tuple[int, ...], torch.Tensor]],
) -> np.ndarray:
    """Return candidate-normalized probabilities after residual replacement."""
    logits = patched_logits(
        model=model,
        tokenizer=tokenizer,
        text=text,
        patches=patches,
    )
    return softmax(logits[candidate_ids])


def _probe_layers(config: dict[str, Any], layer_count: int) -> list[int]:
    requested = config.get("causal_layers", "[:]")
    if requested == "[:]":
        return list(range(layer_count))
    values = [int(value) for value in requested]
    if any(value < 0 or value >= layer_count for value in values):
        raise ValueError(f"causal_layers must lie in [0, {layer_count})")
    return sorted(set(values))


def causal_interventions(
    *,
    model: Any,
    tokenizer: Any,
    candidate_ids: list[int],
    evaluations: dict[str, PromptEvaluation],
    prompts: dict[str, Any],
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Measure target-state control, read cutoff, and visible/hidden contribution."""
    layer_count = len(get_decoder_layers(model))
    probes = _probe_layers(config, layer_count)
    gold = evaluations["gold"]
    counterfactual = evaluations["counterfactual"]
    none = evaluations["none"]
    if not (
        gold.token_count == counterfactual.token_count == none.token_count
        and gold.checkpoint_token_indices
        == counterfactual.checkpoint_token_indices
        == none.checkpoint_token_indices
    ):
        raise ValueError("Causal checkpoint conditions must be exactly token-aligned")
    correct = int(case["next_state"])
    wrong_branch = int(case["counterfactual_next_state"])
    gold_base = np.asarray(gold.record["final_candidate_probabilities"])
    cf_base = np.asarray(counterfactual.record["final_candidate_probabilities"])
    none_base = np.asarray(none.record["final_candidate_probabilities"])

    target_rows = []
    for layer in probes:
        gold_into_cf = patched_probabilities(
            model=model,
            tokenizer=tokenizer,
            text=prompts["counterfactual"].text,
            candidate_ids=candidate_ids,
            patches={layer: ((-1,), gold.final_states[layer][None, :])},
        )
        cf_into_gold = patched_probabilities(
            model=model,
            tokenizer=tokenizer,
            text=prompts["gold"].text,
            candidate_ids=candidate_ids,
            patches={layer: ((-1,), counterfactual.final_states[layer][None, :])},
        )
        target_rows.append(
            {
                "layer": layer,
                "gold_into_counterfactual_correct_probability": float(gold_into_cf[correct]),
                "gold_recovery": normalized_recovery(
                    float(gold_into_cf[correct]), float(cf_base[correct]), float(gold_base[correct])
                ),
                "counterfactual_into_gold_branch_probability": float(cf_into_gold[wrong_branch]),
                "counterfactual_branch_probability": float(cf_into_gold[wrong_branch]),
                "counterfactual_recovery": normalized_recovery(
                    float(cf_into_gold[wrong_branch]), float(gold_base[wrong_branch]), float(cf_base[wrong_branch])
                ),
            }
        )

    read_rows = []
    positions = gold.checkpoint_token_indices
    checkpoint_effect = float(gold_base[correct] - none_base[correct])
    for cutoff in probes:
        probabilities = patched_probabilities(
            model=model,
            tokenizer=tokenizer,
            text=prompts["gold"].text,
            candidate_ids=candidate_ids,
            patches={
                layer: (positions, none.checkpoint_states[layer])
                for layer in range(cutoff, layer_count)
            },
        )
        effect = float(gold_base[correct] - probabilities[correct])
        read_rows.append(
            {
                "cutoff_layer": cutoff,
                "correct_probability": float(probabilities[correct]),
                "effect_from_neutralizing_checkpoint": effect,
                "normalized_remaining_checkpoint_effect": (
                    effect / checkpoint_effect
                    if abs(checkpoint_effect) >= 1e-8
                    else None
                ),
            }
        )

    recovery_threshold = float(config.get("causal_recovery_threshold", 0.8))
    state_control_depth = next(
        (
            int(row["layer"])
            for row in target_rows
            if row["gold_recovery"] is not None
            and row["counterfactual_recovery"] is not None
            and row["gold_recovery"] >= recovery_threshold
            and row["counterfactual_recovery"] >= recovery_threshold
        ),
        None,
    )
    read_threshold = float(config.get("read_effect_fraction_threshold", 0.1))
    read_depth = next(
        (
            int(row["cutoff_layer"])
            for row in read_rows
            if row["normalized_remaining_checkpoint_effect"] is not None
            and abs(row["normalized_remaining_checkpoint_effect"]) <= read_threshold
        ),
        None,
    )

    gold_into_cf = patched_probabilities(
        model=model,
        tokenizer=tokenizer,
        text=prompts["counterfactual"].text,
        candidate_ids=candidate_ids,
        patches={
            layer: (positions, gold.checkpoint_states[layer])
            for layer in range(layer_count)
        },
    )
    cf_into_gold = patched_probabilities(
        model=model,
        tokenizer=tokenizer,
        text=prompts["gold"].text,
        candidate_ids=candidate_ids,
        patches={
            layer: (positions, counterfactual.checkpoint_states[layer])
            for layer in range(layer_count)
        },
    )
    return {
        "probe_layers": probes,
        "state_control_depth": state_control_depth,
        "read_depth": read_depth,
        "checkpoint_correct_probability_effect": checkpoint_effect,
        "causal_recovery_threshold": recovery_threshold,
        "read_effect_fraction_threshold": read_threshold,
        "target_interchange": target_rows,
        "read_cutoff": read_rows,
        "channel_decomposition": {
            "gold_hidden_into_counterfactual": {
                "correct_probability": float(gold_into_cf[correct]),
                "counterfactual_probability": float(gold_into_cf[wrong_branch]),
            },
            "counterfactual_hidden_into_gold": {
                "correct_probability": float(cf_into_gold[correct]),
                "counterfactual_probability": float(cf_into_gold[wrong_branch]),
            },
        },
    }


def evaluate_case_hf(
    *,
    model: Any,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
    run_causal: bool,
) -> dict[str, Any]:
    """Evaluate all checkpoint doses, self-writing, and optional causal interventions."""
    candidate_count = 2 ** int(case["bits"])
    candidates = decimal_state_symbols(candidate_count)
    threshold = float(config.get("settling_jsd_threshold", 0.5))
    write_prompt = format_model_prompt(tokenizer, render_write_prompt(case), config)
    write_ids = candidate_token_ids(tokenizer, write_prompt, candidates)
    writer = evaluate_prompt(
        model=model,
        tokenizer=tokenizer,
        text=write_prompt,
        candidate_ids=write_ids,
        threshold=threshold,
    )
    self_state = int(writer.record["prediction"])
    specs = condition_specs(case, self_state=self_state)
    prompts = {
        spec["name"]: format_prompt_spec(tokenizer, render_prompt(case, spec), config)
        for spec in specs
    }
    candidate_ids = candidate_token_ids(tokenizer, prompts["none"].text, candidates)
    evaluations: dict[str, PromptEvaluation] = {}
    token_contract: tuple[int, tuple[int, ...]] | None = None
    for spec in specs:
        name = str(spec["name"])
        prompt = prompts[name]
        indices = checkpoint_token_indices(tokenizer, prompt)
        evaluation = evaluate_prompt(
            model=model,
            tokenizer=tokenizer,
            text=prompt.text,
            candidate_ids=candidate_ids,
            threshold=threshold,
            checkpoint_indices=indices,
        )
        contract = (evaluation.token_count, indices)
        if token_contract is None:
            token_contract = contract
        elif contract != token_contract:
            raise ValueError(
                f"Condition {name!r} breaks exact token alignment: {contract} != {token_contract}"
            )
        evaluation.record.update(
            {
                "revealed_bits": spec["revealed_bits"],
                "register_state": spec["state"],
                "expected_next_state": int(spec["expected_next_state"]),
                "is_expected": evaluation.record["prediction"] == int(spec["expected_next_state"]),
                "is_expected_unconstrained": evaluation.record["unconstrained_prediction"]
                == int(spec["expected_next_state"]),
                "is_correct": evaluation.record["prediction"] == int(case["next_state"]),
                "is_correct_unconstrained": evaluation.record["unconstrained_prediction"]
                == int(case["next_state"]),
            }
        )
        evaluations[name] = evaluation

    writer_top = [
        int(np.argmax(row)) for row in writer.record["candidate_probabilities"]
    ]
    result = {
        "schema_version": 1,
        "id": case["id"],
        "family": case["family"],
        "bits": case["bits"],
        "next_state": case["next_state"],
        "counterfactual_next_state": case["counterfactual_next_state"],
        "writer": {
            **writer.record,
            "true_state": case["current_state"],
            "self_state": self_state,
            "is_correct": self_state == int(case["current_state"]),
            "is_correct_unconstrained": writer.record["unconstrained_prediction"]
            == int(case["current_state"]),
            "correct_top1_at_any_layer": int(case["current_state"]) in writer_top,
        },
        "conditions": {name: evaluation.record for name, evaluation in evaluations.items()},
        "causal": None,
    }
    if run_causal:
        result["causal"] = causal_interventions(
            model=model,
            tokenizer=tokenizer,
            candidate_ids=candidate_ids,
            evaluations=evaluations,
            prompts=prompts,
            case=case,
            config=config,
        )
    return result
