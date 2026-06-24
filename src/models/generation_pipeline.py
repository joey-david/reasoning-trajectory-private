"""Two-pass generation with selected-layer artifact capture.

Pass 1:
    Generate normally with model.generate(...).

Pass 2:
    Re-run the realized sequence once, teacher-forced, while forward hooks capture
    selected decoder-layer hidden states.

Storage is intentionally not handled here. This module returns:
    CompleteGenerationOutput
    hidden_states tensor with shape [T, L, H]

where:
    T = number of generated tokens
    L = number of selected layers
    H = hidden size
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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

from src.artifact_store import save_generation_output
from src.config import RunConfig
from src.features.logit_lens import (
    ce_for_token,
    entropy_from_logits,
    prob_for_token,
    rank_for_token,
)
from src.generation_output import (
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
from src.run_io import load_generation_index


@dataclass(slots=True)
class GenerationRequest:
    prompt: str
    sample_id: str
    seed: int
    temperature: float
    max_new_tokens: int
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
    full_ids: list[int]
    token_ids: list[int]
    text: str


def generate_run(
    run_path: str | Path,
    config: RunConfig | Mapping[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    """Generate and store outputs for a run folder."""
    run_path = Path(run_path)
    cfg = (
        config
        if isinstance(config, RunConfig)
        else RunConfig.from_dict(run_path, dict(config))
    )

    model_cfg = cfg["model"]
    generation_cfg = cfg["generation"]
    capture_cfg = cfg.get("capture", {})
    prompt_cfg = cfg.get("prompt", {})

    if model_cfg.get("backend", "hf") != "hf":
        raise ValueError(
            f"Unsupported generation backend: {model_cfg.get('backend')!r}"
        )

    model, tokenizer = load_hf_model_and_tokenizer(model_cfg)

    base_seed = int(generation_cfg.get("base_seed", 0))
    num_samples_per_item = int(generation_cfg.get("num_samples_per_item", 1))
    temperature = float(generation_cfg.get("temperature", 0.0))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 1024))
    top_p = generation_cfg.get("top_p")
    top_k = generation_cfg.get("top_k")

    layer_indices = list(capture_cfg.get("layers", [-1]))
    capture_enabled = bool(capture_cfg.get("enabled", True))
    if not capture_enabled:
        layer_indices = []
    capture_diagnostics = bool(capture_cfg.get("diagnostics", False))
    storage_dtype = str(capture_cfg.get("activation_storage_dtype", "float16"))
    existing_generations = load_generation_index(run_path)

    with tqdm(
        total=len(samples) * num_samples_per_item,
        desc="generation",
        unit="iter",
    ) as progress:
        for sample_index, sample in enumerate(samples):
            prompt = build_prompt(sample, prompt_cfg, tokenizer)
            sample_id = sample_id_from_sample(sample)
            gold_answer = gold_answer_from_sample(sample)
            gold_token_id = single_token_id(tokenizer, gold_answer)

            for sample_iter in range(num_samples_per_item):
                seed = base_seed + sample_index * 10_000 + sample_iter
                progress_label = (
                    f"item {sample_index + 1}/{len(samples)} {sample_id} "
                    f"iter {sample_iter + 1}/{num_samples_per_item}"
                )

                generation_key = (sample_id, seed, temperature)
                if generation_key in existing_generations:
                    progress.set_description(f"skipping {progress_label}")
                    progress.update(1)
                    continue

                output, hidden_states = generate_one_twopass(
                    model=model,
                    tokenizer=tokenizer,
                    request=GenerationRequest(
                        prompt=prompt,
                        sample_id=sample_id,
                        seed=seed,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        layer_indices=layer_indices,
                        model_name=model_cfg["name"],
                        gold_answer=gold_answer,
                        gold_token_id=gold_token_id,
                        capture_diagnostics=capture_diagnostics,
                        top_p=top_p,
                        top_k=top_k,
                        progress=progress,
                        progress_label=progress_label,
                    ),
                )

                save_generation_output(
                    run_path=run_path,
                    output=output,
                    hidden_states=hidden_states if capture_enabled else None,
                    storage_dtype=storage_dtype,
                )
                progress.update(1)
                existing_generations.add(generation_key)


@torch.inference_mode()
def generate_one_twopass(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    request: GenerationRequest,
) -> tuple[CompleteGenerationOutput, torch.Tensor | None]:
    """Generate one sample and capture selected-layer hidden states.

    Convention:
        For generated token at position `pos`, the predicting hidden state is
        at `pos - 1`.

    Returns:
        output:
            JSON-facing generation output with scalar timestep artifacts.
        hidden_states:
            Tensor [T, L, H], CPU, float32, or None when no layers are requested.
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
    do_sample = request.temperature > 0.0
    kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": int(request.max_new_tokens),
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

    tracker = None
    if request.progress is not None:
        request.progress.set_description(f"generation {request.progress_label}".strip())
        tracker = LiveGenerationProgress(
            request.progress, prompt_len, request.progress_label
        )
        tracker.update(prompt_len, force=True)
        kwargs["stopping_criteria"] = StoppingCriteriaList([tracker])

    generated = model.generate(**kwargs)
    if tracker is not None:
        tracker.update(int(generated.shape[-1]), force=True)

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


@torch.inference_mode()
def capture_selected_hidden_states(
    *,
    model: PreTrainedModel,
    full_seq_ids: list[int],
    prompt_len: int,
    num_generated: int,
    layer_indices: list[int],
) -> torch.Tensor:
    """Capture selected decoder block outputs for generated-token decisions.

    Returns:
        hidden_states: [T, L, H], CPU, float32.
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
    """Compute scalar per-token artifacts from selected hidden states.

    hidden_states:
        [T, L, H], CPU or GPU.
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
    """Apply final norm + lm_head to one hidden state.

    Args:
        hidden_state: [batch, hidden_dim]

    Returns:
        logits: [batch, vocab_size]
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
        self.decoder_layers = decoder_layers
        self.requested_layers = requested_layers
        self.resolved_layers = resolved_layers
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> SelectedLayerCapture:
        for requested, resolved in zip(self.requested_layers, self.resolved_layers):
            layer = self.decoder_layers[resolved]

            def make_hook(key: int):
                def hook(_module, _inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.outputs[key] = hidden.detach()

                return hook

            self.handles.append(layer.register_forward_hook(make_hook(requested)))

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiveGenerationProgress(StoppingCriteria):
    def __init__(self, progress: Any, prompt_len: int, label: str) -> None:
        self.progress = progress
        self.prompt_len = prompt_len
        self.label = label
        self.started_at = time.monotonic()
        self.last_update = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs) -> bool:
        self.update(int(input_ids.shape[-1]))
        return False

    def update(self, seq_len: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_update < 1.0:
            return
        generated_tokens = max(seq_len - self.prompt_len, 0)
        tok_s = generated_tokens / max(now - self.started_at, 1e-9)
        self.progress.set_description(
            f"generation {generated_tokens} tok {tok_s:.1f} tok/s {self.label}"
        )
        self.last_update = now


def sample_id_from_sample(sample: dict[str, Any]) -> str:
    return str(
        sample.get("id")
        or sample.get("problem_id")
        or sample.get("sample_id")
        or "sample"
    )


def gold_answer_from_sample(sample: dict[str, Any]) -> str | None:
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
    if text is None:
        return None

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        token_ids = tokenizer.encode(" " + text, add_special_tokens=False)

    return int(token_ids[0]) if len(token_ids) == 1 else None
