"""Hidden-state and logit utils"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

@torch.no_grad()
def compute_logit_lens(
    hidden_states: List[torch.Tensor],
    lm_head: torch.nn.Module,
    final_norm: torch.nn.Module = None,
) -> list[torch.Tensor]:
    """Compute logit_lens - the lm_head proj of each wanted layer's hidden state.
    Args:
        hidden_states: list of [batch, hidden_dim] tensors, one per layer
        unembed_matrix: [vocab_size, hidden_dim]
        precision: target precision of torch.Tensor encoding the logits for precision/size tradeoff.
    Returns:
        logits_per_layer: list of [batch, vocab_size] logits at precision precision
        probs_per_layer: list of [batch, vocab_size] probabilities"""

    logits_per_layer: List[torch.Tensor] = []
    probs_per_layer: List[torch.Tensor] = []

    # project to fp32 for stability
    WU32 = lm_head.float()
    for i, h in enumerate(hidden_states):
        h32 = h.float()
        if final_norm is not None:
            h32 = final_norm(h32)
        logits = h32 @ WU32.T

        # catch NaNs early
        if not torch.isfinite(logits).all():
            raise ValueError(f"NaN/Inf in logits at layer {i}")

        logits_per_layer.append(logits)
    
    return logits_per_layer

@torch.no_grad
