"""Teacher-forced residual, component, and token-diagnostic capture."""

from __future__ import annotations

from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.introspection import (
    get_base_model,
    get_decoder_layers,
    get_final_norm,
    get_hidden_size,
    get_input_device,
    get_lm_head,
    module_device,
    resolve_layer_indices,
)
from src.runtime.generation_output import TimestepArtifacts


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
    hidden_states, _ = capture_selected_activations(
        model=model,
        full_seq_ids=full_seq_ids,
        prompt_len=prompt_len,
        num_generated=num_generated,
        layer_indices=layer_indices,
        components=[],
    )
    return hidden_states


def capture_selected_activations(
    *,
    model: PreTrainedModel,
    full_seq_ids: list[int],
    prompt_len: int,
    num_generated: int,
    layer_indices: list[int],
    components: list[str],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Capture residual, MLP, and attention outputs in one teacher-forced pass.

    Args:
        model: Loaded model used for inference or transformation.
        full_seq_ids: Prompt and generated token IDs in one batch.
        prompt_len: Number of prompt tokens before generation.
        num_generated: Number of generated tokens to retain.
        layer_indices: Model layer indices to capture.
        components: Activation component names to capture.

    Returns:
        The computed aligned values described above.
    """
    supported = {"mlp_output", "attention_output"}
    unknown = set(components) - supported
    if unknown:
        raise ValueError(f"Unsupported capture components: {sorted(unknown)}")

    input_device = get_input_device(model)
    full_seq = torch.tensor([full_seq_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(full_seq)

    decoder_layers = get_decoder_layers(model)
    resolved_layers = resolve_layer_indices(layer_indices, len(decoder_layers))

    base_model = get_base_model(model)

    with (
        SelectedLayerCapture(
            decoder_layers=decoder_layers,
            requested_layers=layer_indices,
            resolved_layers=resolved_layers,
        ) as residual_capture,
        SelectedComponentCapture(
            decoder_layers=decoder_layers,
            requested_layers=layer_indices,
            resolved_layers=resolved_layers,
            components=components,
        ) as component_capture,
    ):
        _ = base_model(
            input_ids=full_seq,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )

    if num_generated == 0:
        hidden_size = get_hidden_size(model)
        empty = torch.empty(
            (0, len(layer_indices), hidden_size),
            dtype=torch.float32,
            device="cpu",
        )
        return empty, {component: empty.clone() for component in components}

    start = prompt_len - 1
    stop = start + num_generated
    selected = [
        residual_capture.outputs[layer][0, start:stop, :].float().cpu()
        for layer in layer_indices
    ]
    hidden_states = torch.stack(selected, dim=1)
    component_states = {
        component: torch.stack(
            [
                component_capture.outputs[component][layer][0, start:stop, :]
                .float()
                .cpu()
                for layer in layer_indices
            ],
            dim=1,
        )
        for component in components
    }
    return hidden_states, component_states


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

    T, L, _ = hidden_states.shape
    metrics: dict[str, torch.Tensor] = {
        "entropy": torch.empty((T, L), dtype=torch.float32),
        "ce_next_token": torch.empty((T, L), dtype=torch.float32),
        "rank_next_token": torch.empty((T, L), dtype=torch.int64),
    }
    if gold_token_id is not None:
        metrics.update(
            {
                "ce_gold_answer": torch.empty((T, L), dtype=torch.float32),
                "rank_gold_answer": torch.empty((T, L), dtype=torch.int64),
                "prob_gold_answer": torch.empty((T, L), dtype=torch.float32),
            }
        )
    if eos_token_id is not None:
        metrics.update(
            {
                "prob_eos": torch.empty((T, L), dtype=torch.float32),
                "rank_eos": torch.empty((T, L), dtype=torch.int64),
            }
        )

    targets = torch.tensor(generated_token_ids, dtype=torch.long)
    batch_size = 64
    for layer_col in range(L):
        for start in range(0, T, batch_size):
            stop = min(start + batch_size, T)
            logits = project_hidden_state(
                hidden_states[start:stop, layer_col, :],
                lm_head=lm_head,
                final_norm=final_norm,
            )
            log_probs = torch.log_softmax(logits, dim=-1)
            probabilities = log_probs.exp()
            target_ids = targets[start:stop].to(logits.device)
            target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
            metrics["entropy"][start:stop, layer_col] = (
                -(probabilities * log_probs).sum(dim=-1).cpu()
            )
            metrics["ce_next_token"][start:stop, layer_col] = (
                -log_probs.gather(1, target_ids[:, None]).squeeze(1).cpu()
            )
            metrics["rank_next_token"][start:stop, layer_col] = (
                (logits > target_logits[:, None]).sum(dim=-1).add(1).cpu()
            )
            for token_id, ce_key, probability_key, rank_key in (
                (
                    gold_token_id,
                    "ce_gold_answer",
                    "prob_gold_answer",
                    "rank_gold_answer",
                ),
                (eos_token_id, None, "prob_eos", "rank_eos"),
            ):
                if token_id is None:
                    continue
                token_logits = logits[:, int(token_id)]
                token_log_probs = log_probs[:, int(token_id)]
                if ce_key is not None:
                    metrics[ce_key][start:stop, layer_col] = -token_log_probs.cpu()
                metrics[probability_key][start:stop, layer_col] = (
                    token_log_probs.exp().cpu()
                )
                metrics[rank_key][start:stop, layer_col] = (
                    (logits > token_logits[:, None]).sum(dim=-1).add(1).cpu()
                )

    artifacts: list[TimestepArtifacts] = []
    for token_index, token_id in enumerate(generated_token_ids):
        artifact = TimestepArtifacts.from_token(
            token_id=int(token_id),
            token_str=tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            token_pos=prompt_len + token_index,
        )
        artifact.entropy = metrics["entropy"][token_index].tolist()
        artifact.ce_next_token = metrics["ce_next_token"][token_index].tolist()
        artifact.rank_next_token = metrics["rank_next_token"][token_index].tolist()
        if gold_token_id is not None:
            artifact.ce_gold_answer = metrics["ce_gold_answer"][token_index].tolist()
            artifact.rank_gold_answer = metrics["rank_gold_answer"][
                token_index
            ].tolist()
            artifact.prob_gold_answer = metrics["prob_gold_answer"][
                token_index
            ].tolist()
        if eos_token_id is not None:
            artifact.prob_eos = metrics["prob_eos"][token_index].tolist()
            artifact.rank_eos = metrics["rank_eos"][token_index].tolist()
        artifacts.append(artifact)
    return artifacts


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
    h = hidden_state

    if final_norm is not None:
        # Safely extract the dtype of the norm layer's parameters
        norm_dtype = next(final_norm.parameters()).dtype
        h = h.to(device=module_device(final_norm), dtype=norm_dtype)
        h = final_norm(h)

    # Safely extract the dtype of the lm_head's parameters
    head_dtype = next(lm_head.parameters()).dtype
    h = h.to(device=module_device(lm_head), dtype=head_dtype)

    # Project and immediately cast back to float32 for downstream analysis
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

        Args:
            None.
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


class SelectedComponentCapture:
    """Forward-hook capture for MLP and attention outputs inside decoder blocks."""

    def __init__(
        self,
        *,
        decoder_layers: torch.nn.ModuleList,
        requested_layers: list[int],
        resolved_layers: list[int],
        components: list[str],
    ) -> None:
        """Initialize the helper state.

        Args:
            decoder_layers: Ordered transformer decoder blocks.
            requested_layers: Layer indices requested by the run configuration.
            resolved_layers: Non-negative decoder indices corresponding to requested layers.
            components: Activation component names to capture.

        Returns:
            None.
        """
        self.decoder_layers = decoder_layers
        self.requested_layers = requested_layers
        self.resolved_layers = resolved_layers
        self.components = components
        self.outputs: dict[str, dict[int, torch.Tensor]] = {
            component: {} for component in components
        }
        self.handles: list[Any] = []

    def __enter__(self) -> SelectedComponentCapture:
        """Register capture hooks and enter the context manager.

        Args:
            None.

        Returns:
            This active context manager.
        """
        for requested, resolved in zip(self.requested_layers, self.resolved_layers):
            layer = self.decoder_layers[resolved]
            for component in self.components:
                attribute = "mlp" if component == "mlp_output" else "self_attn"
                module = getattr(layer, attribute, None)
                if module is None:
                    raise TypeError(
                        f"{type(layer).__name__} has no {attribute!r} module"
                    )
                self.handles.append(
                    module.register_forward_hook(self._make_hook(component, requested))
                )
        return self

    def _make_hook(self, component: str, layer: int):
        """Build a forward hook that stores one component output.

        Args:
            component: Activation component name.
            layer: Model layer index.

        Returns:
            A forward-hook callback for the requested component and layer.
        """

        def hook(_module, _inputs, output):
            """Capture and detach the hooked component output.

            Args:
                _module: Hooked PyTorch module; unused by the callback.
                _inputs: Hook input tuple; unused by the callback.
                output: Output produced by the hooked module.

            Returns:
                None.
            """
            hidden = output[0] if isinstance(output, tuple) else output
            self.outputs[component][layer] = hidden.detach()

        return hook

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove all registered hooks when leaving the context.

        Args:
            exc_type: Exception type raised inside the context, if any.
            exc: Exception raised inside the context, if any.
            tb: Traceback associated with the exception, if any.

        Returns:
            None.
        """
        for handle in self.handles:
            handle.remove()
