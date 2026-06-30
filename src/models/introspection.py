"""Locate common decoder components across supported Hugging Face causal-LM architectures."""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


DecoderLayout = tuple[
    torch.nn.Module,
    torch.nn.ModuleList,
    torch.nn.Module | None,
]


def _decoder_layout(model: PreTrainedModel) -> DecoderLayout | None:
    """Resolve the decoder body, layers, and final norm for a supported model.

    Args:
        model: Hugging Face causal language model.

    Returns:
        The aligned decoder components, or ``None`` for an unknown architecture.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model, model.model.layers, getattr(model.model, "norm", None)

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return (
            model.transformer,
            model.transformer.h,
            getattr(model.transformer, "ln_f", None),
        )

    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return (
            model.gpt_neox,
            model.gpt_neox.layers,
            getattr(model.gpt_neox, "final_layer_norm", None),
        )

    return None


def get_base_model(model: PreTrainedModel) -> torch.nn.Module:
    """Return the decoder body without its vocabulary projection.

    Args:
        model: Hugging Face causal language model.

    Returns:
        The architecture's base decoder module.
    """
    layout = _decoder_layout(model)
    if layout is None:
        raise TypeError(f"Could not find base decoder model for {type(model).__name__}")
    return layout[0]


def get_decoder_layers(model: PreTrainedModel) -> torch.nn.ModuleList:
    """Locate the model's ordered decoder-block collection.

    Args:
        model: Hugging Face causal language model.

    Returns:
        The decoder layers used for activation hooks.
    """
    layout = _decoder_layout(model)
    if layout is None:
        raise TypeError(f"Could not find decoder layers for {type(model).__name__}")
    return layout[1]


def get_lm_head(model: PreTrainedModel) -> torch.nn.Module:
    """Locate the model's hidden-state-to-vocabulary projection.

    Args:
        model: Hugging Face causal language model.

    Returns:
        The language-model head module.
    """
    if hasattr(model, "lm_head"):
        return model.lm_head

    if hasattr(model, "embed_out"):
        return model.embed_out

    raise TypeError(f"Could not find lm_head for {type(model).__name__}")


def get_final_norm(model: PreTrainedModel) -> torch.nn.Module | None:
    """Locate the final decoder normalization module when exposed.

    Args:
        model: Hugging Face causal language model.

    Returns:
        The final normalization module, or ``None`` when not found.
    """
    layout = _decoder_layout(model)
    return None if layout is None else layout[2]


def get_input_device(model: PreTrainedModel) -> torch.device:
    """Read the device hosting the model's input embeddings.

    Args:
        model: Hugging Face model exposing input embeddings.

    Returns:
        Device of the input-embedding weights.
    """
    return model.get_input_embeddings().weight.device


def module_device(module: torch.nn.Module) -> torch.device:
    """Read a module's parameter device.

    Args:
        module: PyTorch module to inspect.

    Returns:
        Device of its first parameter, falling back to input embeddings.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return get_input_device(module)  # type: ignore[arg-type]


def get_hidden_size(model: PreTrainedModel) -> int:
    """Resolve the model hidden width from configuration or embeddings.

    Args:
        model: Hugging Face causal language model.

    Returns:
        Decoder hidden-state width.
    """
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)

    if hasattr(model.config, "n_embd"):
        return int(model.config.n_embd)

    return int(model.get_input_embeddings().weight.shape[1])


def resolve_layer_indices(layer_indices: list[int], num_layers: int) -> list[int]:
    """Resolve Python-style negative decoder-layer indices.

    Args:
        layer_indices: Requested positive or negative layer IDs.
        num_layers: Total decoder-layer count.

    Returns:
        Equivalent non-negative layer indices in requested order.
    """
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
    """Reject duplicate requested layer identifiers.

    Args:
        layer_indices: Layer IDs to validate.

    Returns:
        None.
    """
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError(f"Duplicate layer indices are not allowed: {layer_indices}")
