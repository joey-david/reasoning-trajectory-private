"""Token generation and forced-prefix handling."""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.models.generation_types import GeneratedSequence, GenerationRequest
from src.models.generation_utils import GeneratedTextRegexStop, LiveGenerationProgress


def generate_sequence(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    encoded: dict[str, torch.Tensor],
    prompt_len: int,
    request: GenerationRequest,
) -> GeneratedSequence:
    """Run autoregressive generation, including forced prefixes and cap fallback.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer paired with ``model``.
        encoded: Tokenized prompt tensors on the model input device.
        prompt_len: Prompt length before any forced generation prefix.
        request: Sampling, stopping, fallback, and progress options.

    Returns:
        Full token IDs, generated suffix IDs, and decoded generated text.
    """
    do_sample = request.temperature > 0.0
    forced_prefix_ids = encode_forced_prefix(tokenizer, request.forced_prefix)
    encoded_for_generation = append_forced_prefix(encoded, forced_prefix_ids)
    continuation_tokens = int(request.max_new_tokens) - len(forced_prefix_ids)
    if continuation_tokens < 1:
        raise ValueError(
            "generation.forced_prefix must use fewer tokens than max_new_tokens"
        )
    kwargs: dict[str, Any] = {
        **encoded_for_generation,
        "max_new_tokens": continuation_tokens,
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        kwargs["temperature"] = float(request.temperature)
        if request.top_p is not None:
            kwargs["top_p"] = float(request.top_p)
        if request.top_k is not None:
            kwargs["top_k"] = int(request.top_k)

    stopping_criteria: list[StoppingCriteria] = []
    if request.stop_regex:
        stopping_criteria.append(
            GeneratedTextRegexStop(tokenizer, prompt_len, request.stop_regex)
        )
    tracker = None
    if request.progress is not None:
        request.progress.set_description(f"generation {request.progress_label}".strip())
        tracker = LiveGenerationProgress(
            request.progress, prompt_len, request.progress_label
        )
        tracker.update(prompt_len, force=True)
        stopping_criteria.append(tracker)
    if stopping_criteria:
        kwargs["stopping_criteria"] = StoppingCriteriaList(stopping_criteria)

    generated = model.generate(**kwargs)
    if tracker is not None:
        tracker.update(int(generated.shape[-1]), force=True)

    if (
        request.cap_fallback_prefix
        and int(generated.shape[-1]) - prompt_len >= request.max_new_tokens
    ):
        fallback = append_forced_prefix(
            {
                "input_ids": generated,
                "attention_mask": torch.ones_like(generated),
            },
            encode_forced_prefix(tokenizer, request.cap_fallback_prefix),
        )
        generated = model.generate(
            **fallback,
            max_new_tokens=request.cap_fallback_max_new_tokens,
            min_new_tokens=request.cap_fallback_min_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **{
                key: kwargs[key]
                for key in ("temperature", "top_p", "top_k")
                if key in kwargs
            },
        )

    full_ids = generated[0].detach().cpu().tolist()
    token_ids = full_ids[prompt_len:]
    return GeneratedSequence(
        full_ids=full_ids,
        token_ids=token_ids,
        text=tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
    )


def encode_forced_prefix(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> list[int]:
    """Tokenize text for forced insertion without special tokens.

    Args:
        tokenizer: Tokenizer used by generation.
        text: Prefix text to encode; an empty string disables the prefix.

    Returns:
        Prefix token IDs, or an empty list.
    """
    if not text:
        return []
    return tokenizer.encode(text, add_special_tokens=False)


def append_forced_prefix(
    encoded: dict[str, torch.Tensor],
    forced_prefix_ids: list[int],
) -> dict[str, torch.Tensor]:
    """Append fixed token IDs and corresponding attention positions to inputs.

    Args:
        encoded: Tokenizer output containing ``input_ids`` and optional mask.
        forced_prefix_ids: Token IDs to append before model generation.

    Returns:
        The original mapping when no prefix exists, otherwise a shallow copy
        with extended tensors.
    """
    if not forced_prefix_ids:
        return encoded

    input_ids = encoded["input_ids"]
    prefix = torch.tensor(
        [forced_prefix_ids],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    updated = dict(encoded)
    updated["input_ids"] = torch.cat([input_ids, prefix], dim=1)

    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        prefix_mask = torch.ones(
            (attention_mask.shape[0], len(forced_prefix_ids)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        updated["attention_mask"] = torch.cat([attention_mask, prefix_mask], dim=1)

    return updated
