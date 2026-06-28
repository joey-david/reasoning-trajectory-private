"""Compute scalar token diagnostics from batches of projected hidden-state logits."""

import torch
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy for each logits row.

    Args:
        logits: Tensor whose final dimension contains vocabulary logits.

    Returns:
        Entropy values with the vocabulary dimension removed.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def ce_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """Compute cross-entropy against one target token for each logits row.

    Args:
        logits: Two-dimensional batch-by-vocabulary logits.
        token_id: Vocabulary ID to score.

    Returns:
        One negative log-probability per batch row.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -log_probs[:, token_id]


def prob_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """Extract one target token's probability from each logits row.

    Args:
        logits: Two-dimensional batch-by-vocabulary logits.
        token_id: Vocabulary ID to score.

    Returns:
        One probability per batch row.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs[:, token_id].exp()


def rank_for_token(logits: torch.Tensor, token_id: int) -> torch.Tensor:
    """Compute the one-based vocabulary rank of a target token.

    Args:
        logits: Two-dimensional batch-by-vocabulary logits.
        token_id: Vocabulary ID whose rank should be measured.

    Returns:
        One integer rank per batch row.
    """
    target = logits[:, token_id]
    return (logits > target[:, None]).sum(dim=-1) + 1
