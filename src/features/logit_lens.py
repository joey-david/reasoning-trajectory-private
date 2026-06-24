"""Scalar diagnostics from projected hidden-state logits."""

import torch
import torch.nn.functional as F


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
