from __future__ import annotations

import torch
from transformers import PreTrainedModel


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
