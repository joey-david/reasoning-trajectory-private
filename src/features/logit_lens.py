"""Hidden-state and logit utils"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


@torch.inference_mode()
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

    # project to fp32 for stability
    for i, h in enumerate(hidden_states):
        h32 = h.float()
        if final_norm is not None:
            h32 = final_norm(h32)
        logits = lm_head(h32).float()

        # catch NaNs early
        if not torch.isfinite(logits).all():
            raise ValueError(f"NaN/Inf in logits at layer {i}")

        logits_per_layer.append(logits)

    return logits_per_layer


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def ce_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs[:, token_id]


def prob_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs[:, token_id].exp()


def rank_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    target = logits[:, token_id]
    return (logits > target[:, None]).sum(dim=-1) + 1
