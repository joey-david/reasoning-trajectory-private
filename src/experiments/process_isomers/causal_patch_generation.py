"""Generate causal-patching continuations and select matched donors."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any

import torch
from transformers import StoppingCriteriaList

from src.analysis.answers import answers_match, extract_answer
from src.experiments.process_isomers.causal_patch_mechanics import (
    FirstForwardComponentPatch,
    ProjectionSubspace,
    completion_state_index,
    output_degeneration_reasons,
)
from src.experiments.replay_capture import load_source_sample
from src.models.generation_pipeline import GeneratedTextRegexStop, set_seed
from src.models.introspection import get_input_device
from src.runtime.artifact_store import load_component_states_npz


def generate_patched_continuation(
    *,
    model: Any,
    tokenizer: Any,
    activation_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    pairs: list[dict[str, Any]],
    pair: dict[str, Any],
    patch_mode: str,
    condition: str,
    continuation: int,
    seed: int,
    component: str,
    layer: int,
    projection: ProjectionSubspace | None,
    patch_cfg: dict[str, Any],
    analysis_cfg: dict[str, Any],
    vector_cache: dict[tuple[str, int, str, int, int], torch.Tensor],
) -> dict[str, Any]:
    """Generate one continuation after applying the requested H3 intervention.

    Args:
        model: Loaded model used for inference or transformation.
        tokenizer: Tokenizer aligned with the loaded model.
        activation_run: Run directory containing captured activations.
        rows: Generation or analysis records to process.
        pairs: Matched treatment/control or process-isomer pairs.
        pair: Matched pair to evaluate or intervene on.
        patch_mode: Full-space or projected-subspace patch mode.
        condition: Intervention or prompt condition name.
        continuation: Continuation replicate index.
        seed: Random seed for reproducible sampling or generation.
        component: Activation component name.
        layer: Model layer index.
        projection: Optional learned projection subspace.
        patch_cfg: Causal patching configuration.
        analysis_cfg: Answer extraction and scoring configuration.
        vector_cache: Activation vectors cached by trace, component, layer, and state.

    Returns:
        The resulting keyed records or metrics.
    """
    target = pair["target"]
    target_key = (str(target["sample_id"]), int(target["seed"]))
    target_row = rows[target_key]
    target_sample = load_source_sample(activation_run, target_key[0])
    generated_ids = [int(token) for token in target_row["generated_token_ids"]]
    target_completion_index = completion_state_index(target, target_row)
    token_end = target_completion_index - 1
    prefix_ids = [*target_sample["input_ids"], *generated_ids[: token_end + 1]]
    if prefix_ids[-1] != generated_ids[token_end]:
        raise AssertionError("Target prefix is not aligned to symbolic token_end")

    patch_vector = None
    donor_description = None
    donor_gold_answer = None
    reconstruction = None
    if condition != "baseline":
        donor_point, donor_state_index = select_control_donor(
            condition=condition,
            pair=pair,
            pairs=pairs,
            rows=rows,
            target_row=target_row,
        )
        donor_key = (str(donor_point["sample_id"]), int(donor_point["seed"]))
        donor_vector = load_activation_vector(
            activation_run=activation_run,
            rows=rows,
            key=donor_key,
            state_index=donor_state_index,
            component=component,
            layer=layer,
            cache=vector_cache,
        )
        donor_sample = load_source_sample(activation_run, donor_key[0])
        donor_gold_answer = extract_answer(
            str(donor_sample.get("gold_answer", "")),
            analysis_cfg.get("gold_answer_regex"),
        )
        if patch_mode == "full":
            patch_vector = donor_vector
        elif patch_mode == "subspace":
            if projection is None:
                raise ValueError("Subspace patching requires a projection")
            target_vector = load_activation_vector(
                activation_run=activation_run,
                rows=rows,
                key=target_key,
                state_index=target_completion_index,
                component=component,
                layer=layer,
                cache=vector_cache,
            )
            patch_vector, reconstruction = projection.swap(
                target=target_vector,
                donor=donor_vector,
            )
        else:
            raise ValueError(f"Unknown patch mode: {patch_mode!r}")
        donor_description = {
            "sample_id": donor_key[0],
            "seed": donor_key[1],
            "state_index": donor_state_index,
            "token_end": donor_point.get("token_end"),
            "operator": donor_point.get("operator"),
            "operation_signature": donor_point.get("operation_signature"),
        }

    set_seed(seed)
    input_device = get_input_device(model)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    patch_context = (
        FirstForwardComponentPatch(
            model=model,
            layer=layer,
            component=component,
            vector=patch_vector,
            expected_sequence_length=len(prefix_ids),
        )
        if patch_vector is not None
        else nullcontext()
    )
    max_new_tokens = int(patch_cfg.get("max_new_tokens", 768))
    answer_pattern = analysis_cfg.get("produced_answer_regex")
    stopping_criteria = (
        StoppingCriteriaList(
            [GeneratedTextRegexStop(tokenizer, len(prefix_ids), answer_pattern)]
        )
        if answer_pattern
        else None
    )
    with patch_context:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=float(patch_cfg.get("temperature", 0.0)) > 0,
            temperature=float(patch_cfg.get("temperature", 0.6)),
            top_p=float(patch_cfg.get("top_p", 0.95)),
            top_k=int(patch_cfg.get("top_k", 20)),
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    continuation_ids = output[0, len(prefix_ids) :].detach().cpu().tolist()
    text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    produced_answer = extract_answer(text, analysis_cfg.get("produced_answer_regex"))
    target_gold_answer = extract_answer(
        str(target_sample.get("gold_answer", "")),
        analysis_cfg.get("gold_answer_regex"),
    )
    degeneration_reasons = output_degeneration_reasons(continuation_ids, text)
    return {
        "pair_id": pair["pair_id"],
        "target_question": target_key[0],
        "patch_mode": patch_mode,
        "condition": condition,
        "continuation": continuation,
        "seed": seed,
        "component": component,
        "layer": layer,
        "projection_path": projection.path.as_posix() if projection else None,
        "target": target,
        "alignment": {
            "mode": "symbolic_step_end",
            "target_token_end": token_end,
            "target_completion_state_index": target_completion_index,
            "target_prefix_tokens": len(prefix_ids),
        },
        "donor": donor_description,
        "reconstruction": reconstruction,
        "generated_token_ids": continuation_ids,
        "produced_text": text,
        "produced_answer": produced_answer,
        "target_gold_answer": target_gold_answer,
        "donor_gold_answer": donor_gold_answer,
        "matches_target_answer": answers_match(produced_answer, target_gold_answer),
        "matches_donor_answer": (
            answers_match(produced_answer, donor_gold_answer)
            if donor_gold_answer is not None
            else None
        ),
        "matches_neither_answer": (
            produced_answer is not None
            and not answers_match(produced_answer, target_gold_answer)
            and (
                donor_gold_answer is None
                or not answers_match(produced_answer, donor_gold_answer)
            )
        ),
        "is_correct": answers_match(produced_answer, target_gold_answer),
        "has_valid_answer": produced_answer is not None,
        "degenerate_output": bool(degeneration_reasons),
        "degeneration_reasons": degeneration_reasons,
        "ended_with_eos": bool(
            continuation_ids
            and tokenizer.eos_token_id is not None
            and continuation_ids[-1] == tokenizer.eos_token_id
        ),
        "hit_token_limit": len(continuation_ids) >= max_new_tokens,
    }


def load_activation_vector(
    *,
    activation_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    key: tuple[str, int],
    state_index: int,
    component: str,
    layer: int,
    cache: dict[tuple[str, int, str, int, int], torch.Tensor],
) -> torch.Tensor:
    """Load and cache one component activation at a captured state.

    Args:
        activation_run: Run directory containing captured activations.
        rows: Generation or analysis records to process.
        key: Sample and seed key identifying a trace.
        state_index: Completion-state index to load.
        component: Activation component name.
        layer: Model layer index.
        cache: Cached arrays or records used by the computation.

    Returns:
        The resulting numeric array or tensor.
    """
    cache_key = (key[0], key[1], component, layer, state_index)
    if cache_key not in cache:
        row = rows[key]
        states, layers = load_component_states_npz(
            activation_run / row["hidden_states_file"],
            component,
        )
        if layer not in layers:
            raise ValueError(
                f"Layer {layer} is unavailable in {row['hidden_states_file']}"
            )
        if not 0 <= state_index < len(states):
            raise ValueError(
                f"State index {state_index} is unavailable in "
                f"{row['hidden_states_file']}"
            )
        cache[cache_key] = torch.from_numpy(
            states[state_index, layers.index(layer)].astype("float32")
        )
    return cache[cache_key]


def select_control_donor(
    *,
    condition: str,
    pair: dict[str, Any],
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
    target_row: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Select the donor point required by an H3 condition.

    Args:
        condition: Intervention or prompt condition name.
        pair: Matched pair to evaluate or intervene on.
        pairs: Matched treatment/control or process-isomer pairs.
        rows: Generation or analysis records to process.
        target_row: Generation record receiving the patch.

    Returns:
        The computed aligned values described above.
    """
    if condition == "equivalent":
        donor = pair["donor"]
        donor_row = rows[(str(donor["sample_id"]), int(donor["seed"]))]
        return donor, completion_state_index(donor, donor_row)

    alternatives = [
        candidate
        for candidate in pairs
        if candidate["pair_id"] != pair["pair_id"]
        and candidate["graph_signature"] != pair["graph_signature"]
    ]
    if not alternatives:
        raise ValueError("H3 controls require at least two distinct graph states")
    if condition == "mismatched":
        candidate = random.Random(int(pair["pair_id"])).choice(alternatives)
        donor = candidate["donor"]
        donor_row = rows[(str(donor["sample_id"]), int(donor["seed"]))]
        return donor, completion_state_index(donor, donor_row)
    if condition == "position_random":
        target_position = completion_state_index(pair["target"], target_row)
        current_keys = {
            (str(pair[side]["sample_id"]), int(pair[side]["seed"]))
            for side in ("donor", "target")
        }
        eligible_points = [
            candidate[side]
            for candidate in alternatives
            for side in ("donor", "target")
            if (
                str(candidate[side]["sample_id"]),
                int(candidate[side]["seed"]),
            )
            not in current_keys
            and len(
                rows[
                    (
                        str(candidate[side]["sample_id"]),
                        int(candidate[side]["seed"]),
                    )
                ]["generated_token_ids"]
            )
            > target_position
        ]
        if not eligible_points:
            raise ValueError(
                f"No unrelated trace reaches position {target_position} for "
                f"pair {pair['pair_id']}"
            )
        donor = random.Random(int(pair["pair_id"])).choice(eligible_points)
        return donor, target_position
    raise ValueError(f"Unknown patching condition: {condition!r}")
