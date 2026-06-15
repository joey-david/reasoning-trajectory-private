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
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.artifact_store import save_generation_output
from src.config import RunConfig
from src.data import prompt_from_sample
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


def generate_run(
    run_path: str | Path,
    config: RunConfig | Mapping[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    """Generate and store outputs for a run folder."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_path = Path(run_path)
    cfg = (
        config
        if isinstance(config, RunConfig)
        else RunConfig.from_dict(run_path, dict(config))
    )

    if cfg.backend != "hf":
        raise ValueError(f"Unsupported generation backend: {cfg.backend!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map=cfg.device_map,
        torch_dtype=torch_dtype_from_config(cfg.torch_dtype),
        trust_remote_code=cfg.trust_remote_code,
    ).eval()

    for sample in samples:
        prompt = prompt_from_sample(sample, cfg)
        sample_id = sample_id_from_sample(sample)
        gold_answer = gold_answer_from_sample(sample)
        gold_token_id = single_token_id(tokenizer, gold_answer)

        for seed in cfg.seeds:
            for temperature in cfg.temperatures:
                output, hidden_states = generate_one_twopass(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    sample_id=sample_id,
                    seed=seed,
                    temperature=temperature,
                    max_new_tokens=cfg.max_new_tokens,
                    layer_indices=cfg.layers,
                    model_name=cfg.model_name,
                    gold_answer=gold_answer,
                    gold_token_id=gold_token_id,
                    capture_diagnostics=cfg.capture_diagnostics,
                    top_p=cfg.top_p,
                )
                save_generation_output(
                    run_path=run_path,
                    output=output,
                    hidden_states=hidden_states,
                    storage_dtype=cfg.activation_storage_dtype,
                )


@torch.inference_mode()
def generate_one_twopass(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    sample_id: str,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    layer_indices: list[int],
    model_name: str,
    gold_answer: str | None = None,
    gold_token_id: int | None = None,
    capture_diagnostics: bool = True,
    top_p: float | None = None,
) -> tuple[CompleteGenerationOutput, torch.Tensor]:
    """Generate one sample and capture selected-layer hidden states.

    Convention:
        For generated token at position `pos`, the predicting hidden state is
        at `pos - 1`.

    Returns:
        output:
            JSON-facing generation output with scalar timestep artifacts.
        hidden_states:
            Tensor [T, L, H], CPU, float32.
    """
    if not layer_indices:
        raise ValueError("layer_indices must contain at least one layer index")

    assert_unique_layers(layer_indices)
    set_seed(seed)

    input_device = get_input_device(model)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(input_device) for key, value in encoded.items()}

    prompt_token_ids = encoded["input_ids"][0].detach().cpu().tolist()
    prompt_len = len(prompt_token_ids)

    # -------------------------------------------------------------------------
    # Pass 1: generation
    # -------------------------------------------------------------------------
    do_sample = temperature > 0.0

    generate_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)

    generated = model.generate(**generate_kwargs)

    full_seq_ids = generated[0].detach().cpu().tolist()
    generated_token_ids = full_seq_ids[prompt_len:]

    produced_text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # -------------------------------------------------------------------------
    # Pass 2: teacher-forced selected-layer capture
    # -------------------------------------------------------------------------
    hidden_states = capture_selected_hidden_states(
        model=model,
        full_seq_ids=full_seq_ids,
        prompt_len=prompt_len,
        num_generated=len(generated_token_ids),
        layer_indices=layer_indices,
    )

    timestep_artifacts: list[TimestepArtifacts] = []

    if capture_diagnostics and len(generated_token_ids) > 0:
        timestep_artifacts = compute_timestep_artifacts(
            model=model,
            tokenizer=tokenizer,
            hidden_states=hidden_states,
            generated_token_ids=generated_token_ids,
            prompt_len=prompt_len,
            gold_token_id=gold_token_id,
        )
    else:
        timestep_artifacts = minimal_timestep_artifacts(
            tokenizer=tokenizer,
            generated_token_ids=generated_token_ids,
            prompt_len=prompt_len,
        )

    output = CompleteGenerationOutput(
        sample_id=sample_id,
        seed=seed,
        temperature=temperature,
        model_name=model_name,
        layer_indices=layer_indices,
        hidden_state_convention=HIDDEN_STATE_CONVENTION,
        prompt=prompt,
        input_ids=prompt_token_ids,
        generated_token_ids=generated_token_ids,
        full_seq_ids=full_seq_ids,
        dp1_idx=prompt_len,
        dp2_idx=None,
        reasoning_length=None,
        produced_text=produced_text,
        produced_answer=None,
        gold_answer=gold_answer,
        is_correct=None,
        timestep_artifacts=timestep_artifacts,
        hidden_states_file=None,
        metadata={
            "prompt_length": prompt_len,
            "generated_length": len(generated_token_ids),
            "do_sample": do_sample,
            "top_p": top_p,
        },
    )

    return output, hidden_states


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

    rows: list[torch.Tensor] = []

    for step in range(num_generated):
        token_pos = prompt_len + step
        predict_from_pos = token_pos - 1

        selected_layers = []

        for layer in layer_indices:
            h = capture.outputs[layer]  # [1, seq_len, hidden_dim]
            selected_layers.append(h[0, predict_from_pos, :].detach().float().cpu())

        rows.append(torch.stack(selected_layers, dim=0))  # [L, H]

    return torch.stack(rows, dim=0)  # [T, L, H]


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

        artifacts.append(
            artifact_with_diagnostics(
                artifact,
                entropy=entropy,
                ce_next_token=ce_next_token,
                rank_next_token=rank_next_token,
                ce_gold_answer=ce_gold_answer,
                rank_gold_answer=rank_gold_answer,
                prob_gold_answer=prob_gold_answer,
                prob_eos=prob_eos,
                rank_eos=rank_eos,
            )
        )

    return artifacts


def minimal_timestep_artifacts(
    *,
    tokenizer: PreTrainedTokenizerBase,
    generated_token_ids: list[int],
    prompt_len: int,
) -> list[TimestepArtifacts]:
    artifacts = []

    for t, token_id in enumerate(generated_token_ids):
        token_pos = prompt_len + t
        artifacts.append(
            TimestepArtifacts.from_token(
                token_id=int(token_id),
                token_str=tokenizer.decode(
                    [int(token_id)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                token_pos=token_pos,
            )
        )

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
                def hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    self.outputs[key] = hidden.detach()

                return hook

            self.handles.append(layer.register_forward_hook(make_hook(requested)))

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()


def get_base_model(model: PreTrainedModel) -> torch.nn.Module:
    """Return the decoder/base model, avoiding full-vocab logits in pass 2."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox

    raise TypeError(f"Could not find base decoder model for {type(model).__name__}")


def get_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers

    raise TypeError(f"Could not find decoder layers for {type(model).__name__}")


def get_lm_head(model: PreTrainedModel) -> torch.nn.Module:
    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(model, "embed_out"):
        return model.embed_out

    raise TypeError(f"Could not find lm_head for {type(model).__name__}")


def get_final_norm(model: PreTrainedModel) -> torch.nn.Module | None:
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm

    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "final_layer_norm"):
        return model.gpt_neox.final_layer_norm

    return None


def get_input_device(model: PreTrainedModel) -> torch.device:
    return model.get_input_embeddings().weight.device


def module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return get_input_device(module)  # type: ignore[arg-type]


def get_hidden_size(model: PreTrainedModel) -> int:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)

    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)

    return int(model.get_input_embeddings().weight.shape[1])


def resolve_layer_indices(layer_indices: list[int], num_layers: int) -> list[int]:
    resolved = []

    for layer in layer_indices:
        idx = layer if layer >= 0 else num_layers + layer

        if idx < 0 or idx >= num_layers:
            raise IndexError(
                f"Layer index {layer} resolves to {idx}, "
                f"but model has {num_layers} decoder layers"
            )

        resolved.append(idx)

    return resolved


def assert_unique_layers(layer_indices: list[int]) -> None:
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"Duplicate layer indices are not allowed: {layer_indices}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def artifact_with_diagnostics(
    artifact: TimestepArtifacts,
    *,
    entropy: list[float],
    ce_next_token: list[float],
    rank_next_token: list[int],
    ce_gold_answer: list[float] | None,
    rank_gold_answer: list[int] | None,
    prob_gold_answer: list[float] | None,
    prob_eos: list[float] | None,
    rank_eos: list[int] | None,
) -> TimestepArtifacts:
    artifact.entropy = entropy
    artifact.ce_next_token = ce_next_token
    artifact.rank_next_token = rank_next_token
    artifact.ce_gold_answer = ce_gold_answer
    artifact.rank_gold_answer = rank_gold_answer
    artifact.prob_gold_answer = prob_gold_answer
    artifact.prob_eos = prob_eos
    artifact.rank_eos = rank_eos
    return artifact


def torch_dtype_from_config(dtype: str | None) -> torch.dtype | str | None:
    if dtype in (None, "auto"):
        return dtype

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    try:
        return aliases[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch_dtype config value: {dtype!r}") from exc


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
