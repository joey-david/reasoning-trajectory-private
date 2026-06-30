"""Orchestrate generation followed by a teacher-forced pass that captures selected decoder-layer states. It also derives optional token diagnostics and persists completed outputs through the artifact store."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src.runtime.artifact_store import save_generation_output
from src.runtime.config import RunConfig
from src.features.logit_lens import (
    ce_for_token,
    entropy_from_logits,
    prob_for_token,
    rank_for_token,
)
from src.runtime.generation_output import (
    HIDDEN_STATE_CONVENTION,
    CompleteGenerationOutput,
    TimestepArtifacts,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.models.introspection import (
    assert_unique_layers,
    get_base_model,
    get_decoder_layers,
    get_final_norm,
    get_hidden_size,
    get_input_device,
    get_lm_head,
    module_device,
    resolve_layer_indices,
)
from src.prompting.templates import build_prompt
from src.runtime.run_io import load_generation_index


@dataclass(slots=True)
class GenerationRequest:
    """Hold all resolved inputs needed to generate and analyze one rollout."""

    prompt: str
    sample_id: str
    seed: int
    temperature: float
    max_new_tokens: int
    forced_prefix: str
    stop_regex: str | None
    cap_fallback_prefix: str
    cap_fallback_min_new_tokens: int
    cap_fallback_max_new_tokens: int
    layer_indices: list[int]
    model_name: str
    gold_answer: str | None
    gold_token_id: int | None
    capture_diagnostics: bool
    top_p: float | None
    top_k: int | None
    progress: Any | None
    progress_label: str


@dataclass(slots=True)
class GeneratedSequence:
    """Hold the full sequence, generated suffix, and decoded generated text."""

    full_ids: list[int]
    token_ids: list[int]
    text: str


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
    layer_indices = list(capture_cfg.get("layers", [-1])) if capture_enabled else []
    gold_answer = gold_answer_from_sample(sample)
    output, hidden_states = generate_one_twopass(
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
    )
    return output


@torch.inference_mode()
def generate_one_twopass(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    request: GenerationRequest,
) -> tuple[CompleteGenerationOutput, torch.Tensor | None]:
    """Generate one rollout, then optionally capture and diagnose its hidden states.

    Args:
        model: Loaded causal language model.
        tokenizer: Tokenizer paired with ``model``.
        request: Fully resolved prompt, sampling, capture, and progress options.

    Returns:
        The JSON-facing output and an optional CPU float32 tensor shaped
        ``[generated_tokens, selected_layers, hidden_size]``. Each generated
        token uses the hidden state at its preceding prediction position.
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
    if request.layer_indices:
        if request.progress is not None:
            request.progress.set_postfix({}, refresh=True)
            request.progress.set_description(
                f"activation capture {request.progress_label}".strip()
            )
        hidden_states = capture_selected_hidden_states(
            model=model,
            full_seq_ids=sequence.full_ids,
            prompt_len=prompt_len,
            num_generated=len(sequence.token_ids),
            layer_indices=request.layer_indices,
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

    return output, hidden_states


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


@torch.inference_mode()
def capture_selected_hidden_states(
    *,
    model: PreTrainedModel,
    full_seq_ids: list[int],
    prompt_len: int,
    num_generated: int,
    layer_indices: list[int],
) -> torch.Tensor:
    """Capture selected decoder outputs at positions that predict generated tokens.

    Args:
        model: Loaded causal language model.
        full_seq_ids: Prompt and generated token IDs from pass one.
        prompt_len: Number of prompt tokens.
        num_generated: Number of generated suffix tokens.
        layer_indices: Requested positive or negative decoder-layer IDs.

    Returns:
        CPU float32 hidden states shaped
        ``[generated_tokens, selected_layers, hidden_size]``.
    """
    input_device = get_input_device(model)
    full_seq = torch.tensor([full_seq_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(full_seq)

    decoder_layers = get_decoder_layers(model)
    resolved_layers = resolve_layer_indices(layer_indices, len(decoder_layers))

    base_model = get_base_model(model)

    with SelectedLayerCapture(
        decoder_layers=decoder_layers,
        requested_layers=layer_indices,
        resolved_layers=resolved_layers,
    ) as capture:
        _ = base_model(
            input_ids=full_seq,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    if num_generated == 0:
        hidden_size = get_hidden_size(model)
        return torch.empty(
            (0, len(layer_indices), hidden_size),
            dtype=torch.float32,
            device="cpu",
        )

    start = prompt_len - 1
    stop = start + num_generated
    selected = [
        capture.outputs[layer][0, start:stop, :].float().cpu()
        for layer in layer_indices
    ]
    return torch.stack(selected, dim=1)  # [T, L, H]


@torch.inference_mode()
def compute_timestep_artifacts(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    hidden_states: torch.Tensor,
    generated_token_ids: list[int],
    prompt_len: int,
    gold_token_id: int | None,
) -> list[TimestepArtifacts]:
    """Compute per-layer scalar diagnostics for each generated token.

    Args:
        model: Model providing the final norm and language-model head.
        tokenizer: Tokenizer used to decode tokens and identify EOS.
        hidden_states: Tensor shaped ``[tokens, layers, hidden]``.
        generated_token_ids: Target generated IDs aligned with the token axis.
        prompt_len: Prompt length used to calculate absolute token positions.
        gold_token_id: Optional single-token gold answer to diagnose.

    Returns:
        One populated :class:`TimestepArtifacts` record per generated token.
    """
    lm_head = get_lm_head(model)
    final_norm = get_final_norm(model)

    eos_token_id = tokenizer.eos_token_id

    artifacts: list[TimestepArtifacts] = []

    T, L, _ = hidden_states.shape

    for t in range(T):
        token_id = int(generated_token_ids[t])
        token_str = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        token_pos = prompt_len + t
        artifact = TimestepArtifacts.from_token(
            token_id=token_id,
            token_str=token_str,
            token_pos=token_pos,
        )

        entropy: list[float] = []
        ce_next_token: list[float] = []
        rank_next_token: list[int] = []

        ce_gold_answer: list[float] | None = [] if gold_token_id is not None else None
        rank_gold_answer: list[int] | None = [] if gold_token_id is not None else None
        prob_gold_answer: list[float] | None = [] if gold_token_id is not None else None

        prob_eos: list[float] | None = [] if eos_token_id is not None else None
        rank_eos: list[int] | None = [] if eos_token_id is not None else None

        for layer_col in range(L):
            h = hidden_states[t, layer_col, :].unsqueeze(0)  # [1, H]
            logits = project_hidden_state(
                h,
                lm_head=lm_head,
                final_norm=final_norm,
            )  # [1, vocab]

            entropy.append(float(entropy_from_logits(logits)[0].detach().cpu()))
            ce_next_token.append(
                float(ce_for_token(logits, token_id)[0].detach().cpu())
            )
            rank_next_token.append(
                int(rank_for_token(logits, token_id)[0].detach().cpu())
            )

            if gold_token_id is not None:
                ce_gold_answer.append(
                    float(ce_for_token(logits, gold_token_id)[0].detach().cpu())
                )
                rank_gold_answer.append(
                    int(rank_for_token(logits, gold_token_id)[0].detach().cpu())
                )
                prob_gold_answer.append(
                    float(prob_for_token(logits, gold_token_id)[0].detach().cpu())
                )

            if eos_token_id is not None:
                prob_eos.append(
                    float(prob_for_token(logits, eos_token_id)[0].detach().cpu())
                )
                rank_eos.append(
                    int(rank_for_token(logits, eos_token_id)[0].detach().cpu())
                )

        artifact.entropy = entropy
        artifact.ce_next_token = ce_next_token
        artifact.rank_next_token = rank_next_token
        artifact.ce_gold_answer = ce_gold_answer
        artifact.rank_gold_answer = rank_gold_answer
        artifact.prob_gold_answer = prob_gold_answer
        artifact.prob_eos = prob_eos
        artifact.rank_eos = rank_eos
        artifacts.append(artifact)

    return artifacts


@torch.inference_mode()
def project_hidden_state(
    hidden_state: torch.Tensor,
    *,
    lm_head: torch.nn.Module,
    final_norm: torch.nn.Module | None,
) -> torch.Tensor:
    """Project hidden states into vocabulary logits using the model output stack.

    Args:
        hidden_state: Tensor shaped ``[batch, hidden_size]``.
        lm_head: Hidden-state-to-vocabulary projection.
        final_norm: Optional final decoder normalization applied before projection.

    Returns:
        Float32 logits shaped ``[batch, vocabulary_size]``.
    """
    h = hidden_state.float()

    if final_norm is not None:
        h = h.to(module_device(final_norm))
        h = final_norm(h)

    h = h.to(module_device(lm_head))
    logits = lm_head(h).float()

    if not torch.isfinite(logits).all():
        raise ValueError("NaN/Inf in projected logits")

    return logits


class SelectedLayerCapture:
    """Forward-hook capture for selected decoder block outputs."""

    def __init__(
        self,
        *,
        decoder_layers: torch.nn.ModuleList,
        requested_layers: list[int],
        resolved_layers: list[int],
    ) -> None:
        """Configure a hook context for requested decoder layers.

        Args:
            decoder_layers: Ordered decoder-block modules.
            requested_layers: User-facing layer IDs used as output keys.
            resolved_layers: Non-negative module indices aligned with requested IDs.

        Returns:
            None.
        """
        self.decoder_layers = decoder_layers
        self.requested_layers = requested_layers
        self.resolved_layers = resolved_layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> SelectedLayerCapture:
        """Register forward hooks for all selected layers.

        Returns:
            This capture object, whose ``outputs`` fill during a model forward pass.
        """
        for requested, resolved in zip(self.requested_layers, self.resolved_layers):
            layer = self.decoder_layers[resolved]

            def make_hook(key: int):
                """Build a forward hook that records one requested layer.

                Args:
                    key: User-facing layer ID used in the output mapping.

                Returns:
                    A PyTorch forward-hook callback.
                """

                def hook(_module, _inputs, output):
                    """Detach and retain the decoder block's hidden-state output.

                    Args:
                        _module: Decoder module supplied by PyTorch's hook API.
                        _inputs: Positional module inputs supplied by the hook API.
                        output: Tensor or tuple whose first item is the hidden state.

                    Returns:
                        None.
                    """
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.outputs[key] = hidden.detach()

                return hook

            self.handles.append(layer.register_forward_hook(make_hook(requested)))

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all registered hooks when leaving the capture context.

        Args:
            exc_type: Exception type raised in the context, if any.
            exc: Exception instance raised in the context, if any.
            tb: Associated traceback, if any.

        Returns:
            None; exceptions are not suppressed.
        """
        for handle in self.handles:
            handle.remove()


def set_seed(seed: int) -> None:
    """Seed PyTorch CPU and available CUDA random generators.

    Args:
        seed: Deterministic generation seed.

    Returns:
        None.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiveGenerationProgress(StoppingCriteria):
    """Update a tqdm progress description while always allowing generation to continue."""

    def __init__(self, progress: Any, prompt_len: int, label: str) -> None:
        """Initialize throttled token-rate reporting.

        Args:
            progress: tqdm-compatible progress object.
            prompt_len: Input length excluded from generated-token counts.
            label: Human-readable rollout identity appended to the description.

        Returns:
            None.
        """
        self.progress = progress
        self.prompt_len = prompt_len
        self.label = label
        self.started_at = time.monotonic()
        self.last_update = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs) -> bool:
        """Receive a Transformers stopping callback and update progress.

        Args:
            input_ids: Current prompt-plus-generation token IDs.
            scores: Current generation scores, unused by this tracker.
            **kwargs: Additional Transformers callback values, ignored.

        Returns:
            Always ``False`` so this tracker never stops generation.
        """
        self.update(int(input_ids.shape[-1]))
        return False

    def update(self, seq_len: int, *, force: bool = False) -> None:
        """Refresh the displayed token count and throughput when due.

        Args:
            seq_len: Current total prompt-plus-generation sequence length.
            force: Bypass the one-second display throttle.

        Returns:
            None.
        """
        now = time.monotonic()
        if not force and now - self.last_update < 1.0:
            return
        generated_tokens = max(seq_len - self.prompt_len, 0)
        tok_s = generated_tokens / max(now - self.started_at, 1e-9)
        self.progress.set_description(
            f"generation {generated_tokens} tok {tok_s:.1f} tok/s {self.label}"
        )
        self.last_update = now


class GeneratedTextRegexStop(StoppingCriteria):
    """Stop generation after a regex match has been followed by more decoded text."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_len: int,
        pattern: str,
    ) -> None:
        """Initialize incremental decoded-text matching.

        Args:
            tokenizer: Tokenizer used to decode newly generated IDs.
            prompt_len: Input length marking the start of generated text.
            pattern: Regular expression compiled with dot matching newlines.

        Returns:
            None.
        """
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.pattern = re.compile(pattern, re.S)
        self.last_len = prompt_len
        self.text = ""

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs) -> bool:
        """Decode new IDs and report whether the completed match can stop generation.

        Args:
            input_ids: Current prompt-plus-generation token IDs.
            scores: Current generation scores, unused by this criterion.
            **kwargs: Additional Transformers callback values, ignored.

        Returns:
            ``True`` once at least one character follows a regex match.
        """
        seq_len = int(input_ids.shape[-1])
        self.text += self.tokenizer.decode(
            input_ids[0, self.last_len : seq_len],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_len = seq_len
        match = self.pattern.search(self.text)
        return match is not None and match.end() < len(self.text)


def sample_id_from_sample(sample: dict[str, Any]) -> str:
    """Resolve a normalized sample identifier through supported fallback fields.

    Args:
        sample: Dataset sample containing one of the supported identity fields.

    Returns:
        String sample ID, falling back to ``"sample"``.
    """
    return str(
        sample.get("id")
        or sample.get("problem_id")
        or sample.get("sample_id")
        or "sample"
    )


def gold_answer_from_sample(sample: dict[str, Any]) -> str | None:
    """Resolve a gold answer through supported dataset field names.

    Args:
        sample: Dataset sample with an optional expected answer.

    Returns:
        String answer or ``None`` when no answer field is populated.
    """
    answer = (
        sample.get("expected_answer")
        or sample.get("correct_letter")
        or sample.get("answer")
        or sample.get("gold_answer")
    )
    return None if answer is None else str(answer)


def single_token_id(
    tokenizer: PreTrainedTokenizerBase,
    text: str | None,
) -> int | None:
    """Resolve text to one tokenizer ID, retrying with a leading space.

    Args:
        tokenizer: Tokenizer used by the model.
        text: Candidate answer string.

    Returns:
        The sole token ID when either encoding is one token, otherwise ``None``.
    """
    if text is None:
        return None

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        token_ids = tokenizer.encode(" " + text, add_special_tokens=False)

    return int(token_ids[0]) if len(token_ids) == 1 else None
