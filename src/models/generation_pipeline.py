"""Orchestrate generation and persist teacher-forced activation captures."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.activation_capture import (
    capture_selected_activations,
    compute_timestep_artifacts,
)
from src.models.generation_inference import generate_sequence
from src.models.generation_types import GeneratedSequence, GenerationRequest
from src.models.generation_utils import (
    GeneratedTextRegexStop,
    LiveGenerationProgress,
    gold_answer_from_sample,
    sample_id_from_sample,
    set_seed,
    single_token_id,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.models.introspection import (
    assert_unique_layers,
    get_decoder_layers,
    get_input_device,
)
from src.prompting.templates import build_prompt
from src.runtime.artifact_store import save_generation_output
from src.runtime.config import RunConfig
from src.runtime.generation_output import (
    HIDDEN_STATE_CONVENTION,
    CompleteGenerationOutput,
)
from src.runtime.run_io import load_generation_index


def generate_run(
    run_path: str | Path,
    config: RunConfig | Mapping[str, Any],
    samples: list[dict[str, Any]],
    *,
    sample_index_offset: int = 0,
) -> None:
    """Generate and persist every configured rollout for selected samples.

    Args:
        run_path: Run folder receiving generated artifacts.
        config: Typed or mapping-compatible model, generation, capture, and prompt config.
        samples: Normalized dataset samples to generate.
        sample_index_offset: Global index of the first sample, used for sharded seeds.

    Returns:
        None; completed rollout artifacts are written under the run folder.
    """
    run_path = Path(run_path)
    cfg = (
        config
        if isinstance(config, RunConfig)
        else RunConfig.from_dict(run_path, dict(config))
    )
    model_cfg = cfg["model"]
    if model_cfg.get("backend", "hf") != "hf":
        raise ValueError(
            f"Unsupported generation backend: {model_cfg.get('backend')!r}"
        )
    model, tokenizer = load_hf_model_and_tokenizer(model_cfg)
    generation_cfg = cfg["generation"]
    samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    existing_generations = load_generation_index(run_path)

    with tqdm(
        total=len(samples) * samples_per_item,
        desc="generation",
        unit="iter",
    ) as progress:
        for local_sample_index, sample in enumerate(samples):
            sample_index = sample_index_offset + local_sample_index
            for sample_iter in range(samples_per_item):
                key = generation_key_for(
                    sample, sample_index, sample_iter, generation_cfg
                )
                label = (
                    f"item {local_sample_index + 1}/{len(samples)} {key[0]} "
                    f"iter {sample_iter + 1}/{samples_per_item}"
                )
                if key in existing_generations:
                    progress.set_description(f"skipping {label}")
                    progress.update(1)
                    continue
                generate_task(
                    run_path=run_path,
                    config=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    sample=sample,
                    sample_index=sample_index,
                    sample_iter=sample_iter,
                    progress=progress,
                    progress_label=label,
                )
                progress.update(1)
                existing_generations.add(key)


def generation_key_for(
    sample: dict[str, Any],
    sample_index: int,
    sample_iter: int,
    generation_cfg: Mapping[str, Any],
) -> tuple[str, int, float]:
    """Build the deterministic persisted identity for one rollout.

    Args:
        sample: Normalized dataset sample.
        sample_index: Global sample index used in the seed formula.
        sample_iter: Zero-based rollout iteration for the sample.
        generation_cfg: Seed and temperature configuration.

    Returns:
        ``(sample_id, seed, temperature)`` generation key.
    """
    sample_id = sample_id_from_sample(sample)
    seed = int(generation_cfg.get("base_seed", 0)) + sample_index * 10_000 + sample_iter
    temperature = float(generation_cfg.get("temperature", 0.0))
    return sample_id, seed, temperature


def generate_task(
    *,
    run_path: Path,
    config: Mapping[str, Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample: dict[str, Any],
    sample_index: int,
    sample_iter: int,
    progress: Any | None,
    progress_label: str,
) -> CompleteGenerationOutput:
    """Generate and persist one rollout using an already loaded model.

    Args:
        run_path: Run folder receiving artifacts.
        config: Complete run configuration.
        model: Long-lived causal language model.
        tokenizer: Tokenizer paired with the model.
        sample: Normalized sample to generate.
        sample_index: Global sample index used in the seed.
        sample_iter: Zero-based rollout iteration.
        progress: Optional tqdm-compatible progress sink.
        progress_label: Human-readable item and iteration label.

    Returns:
        The completed and persisted generation output.
    """
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    capture_cfg = config.get("capture", {})
    prompt_cfg = config.get("prompt", {})
    sample_id, seed, temperature = generation_key_for(
        sample, sample_index, sample_iter, generation_cfg
    )
    forced_prefix = generation_cfg.get("forced_prefix")
    cap_fallback = generation_cfg.get("cap_fallback", {})
    capture_enabled = bool(capture_cfg.get("enabled", True))
    layer_indices = (
        resolve_capture_layer_indices(capture_cfg, model) if capture_enabled else []
    )
    gold_answer = gold_answer_from_sample(sample)
    output, hidden_states, component_states = generate_one_twopass(
        model=model,
        tokenizer=tokenizer,
        request=GenerationRequest(
            prompt=build_prompt(sample, prompt_cfg, tokenizer),
            sample_id=sample_id,
            seed=seed,
            temperature=temperature,
            max_new_tokens=int(generation_cfg.get("max_new_tokens", 1024)),
            forced_prefix="" if forced_prefix is None else str(forced_prefix),
            stop_regex=generation_cfg.get(
                "stop_regex",
                config.get("analysis", {}).get("produced_answer_regex"),
            ),
            cap_fallback_prefix=str(cap_fallback.get("prefix", "")),
            cap_fallback_min_new_tokens=int(cap_fallback.get("min_new_tokens", 3)),
            cap_fallback_max_new_tokens=int(cap_fallback.get("max_new_tokens", 4)),
            layer_indices=layer_indices,
            capture_components=[
                str(component) for component in capture_cfg.get("components", [])
            ],
            model_name=str(model_cfg["name"]),
            gold_answer=gold_answer,
            gold_token_id=single_token_id(tokenizer, gold_answer),
            capture_diagnostics=bool(capture_cfg.get("diagnostics", False)),
            top_p=generation_cfg.get("top_p"),
            top_k=generation_cfg.get("top_k"),
            progress=progress,
            progress_label=progress_label,
        ),
    )
    save_generation_output(
        run_path=run_path,
        output=output,
        hidden_states=hidden_states if capture_enabled else None,
        storage_dtype=str(capture_cfg.get("activation_storage_dtype", "float16")),
        component_states=component_states if capture_enabled else None,
    )
    return output


def resolve_capture_layer_indices(
    capture_cfg: Mapping[str, Any],
    model: PreTrainedModel,
) -> list[int]:
    """Resolve configured capture layers, including ``[:]`` for every layer."""
    raw_layers = capture_cfg.get("layers", [-1])
    if isinstance(raw_layers, str):
        if raw_layers.strip() == "[:]":
            return list(range(len(get_decoder_layers(model))))
        raise ValueError(
            "capture.layers must be a list of layer indices or the [:] sentinel"
        )
    return [int(layer) for layer in raw_layers]


def generate_one_twopass(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    request: GenerationRequest,
) -> tuple[
    CompleteGenerationOutput,
    torch.Tensor | None,
    dict[str, torch.Tensor],
]:
    """Generate one rollout, then optionally capture and diagnose its hidden states.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer paired with ``model``.
        request: Fully resolved prompt, sampling, capture, and progress options.

    Returns:
        The JSON-facing output, optional residual states, and any requested
        component states. Activation tensors are shaped
        ``[generated_tokens, selected_layers, hidden_size]``.
    """
    if request.layer_indices:
        assert_unique_layers(request.layer_indices)
    set_seed(request.seed)

    input_device = get_input_device(model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(request.prompt, return_tensors="pt")
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    prompt_token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    prompt_len = len(prompt_token_ids)

    sequence = generate_sequence(
        model=model,
        tokenizer=tokenizer,
        encoded=encoded,
        prompt_len=prompt_len,
        request=request,
    )

    # -------------------------------------------------------------------------
    # Pass 2: teacher-forced selected-layer capture
    # -------------------------------------------------------------------------
    hidden_states = None
    component_states: dict[str, torch.Tensor] = {}
    if request.layer_indices:
        if request.progress is not None:
            request.progress.set_postfix({}, refresh=True)
            request.progress.set_description(
                f"activation capture {request.progress_label}".strip()
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        hidden_states, component_states = capture_selected_activations(
            model=model,
            full_seq_ids=sequence.full_ids,
            prompt_len=prompt_len,
            num_generated=len(sequence.token_ids),
            layer_indices=request.layer_indices,
            components=request.capture_components,
        )
    elif request.progress is not None:
        request.progress.set_postfix({}, refresh=True)
        request.progress.set_description(
            f"activation capture skipped {request.progress_label}".strip()
        )

    if request.capture_diagnostics and hidden_states is not None and sequence.token_ids:
        timestep_artifacts = compute_timestep_artifacts(
            model=model,
            tokenizer=tokenizer,
            hidden_states=hidden_states,
            generated_token_ids=sequence.token_ids,
            prompt_len=prompt_len,
            gold_token_id=request.gold_token_id,
        )
    else:
        timestep_artifacts = []

    output = CompleteGenerationOutput(
        sample_id=request.sample_id,
        seed=request.seed,
        temperature=request.temperature,
        model_name=request.model_name,
        layer_indices=request.layer_indices,
        hidden_state_convention=HIDDEN_STATE_CONVENTION,
        prompt=request.prompt,
        input_ids=prompt_token_ids,
        generated_token_ids=sequence.token_ids,
        dp1_idx=prompt_len,
        dp2_idx=None,
        reasoning_length=None,
        produced_text=sequence.text,
        produced_answer=None,
        gold_answer=request.gold_answer,
        is_correct=None,
        timestep_artifacts=timestep_artifacts,
        hidden_states_file=None,
    )

    return output, hidden_states, component_states


__all__ = [
    "GeneratedSequence",
    "GeneratedTextRegexStop",
    "GenerationRequest",
    "LiveGenerationProgress",
    "generate_one_twopass",
    "generate_run",
    "generate_task",
    "generation_key_for",
    "gold_answer_from_sample",
    "sample_id_from_sample",
    "set_seed",
    "single_token_id",
]
