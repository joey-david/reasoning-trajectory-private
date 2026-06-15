"""Generation output dataclass with activations.
Used to store the first pass of generation, on which activations are
computed during the second pass"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union

from yaml import tokens
import numpy as np

@dataclass
class TimestepArtifacts:
    """Per-token (later, timestep) artifacts, optimized for GPU storage (speedup)."""
    tokens_ids: List[int]
    tokens_str: str

    next_token_id: int

    # entropy per selected layer
    entropy: Optional[List[float]] = None

    # cross_entropy for the following targets for selected layers
    CE_next_tok: Optional[List[float]] = None
    CE_gold_ans: Optional[List[float]] = None
    # CE for produced_answer (retroactive)
    CE_produced_ans: Optional[List[float]] = None

    # prob and rank of eos token for selected layers
    prob_eos: Optional[List[float]] = None
    rank_eos: Optional[List[int]] = None

    # and what we came here for: full hidden state arrays per layer
    # TODO: check quantization for efficient storage
    hidden_states: Optional[List[Union["torch.Tensor", np.ndarray]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, including hidden states if present.
        This allows the caller to control whether hidden_states are stored by setting them to None"""

        result= {
            "tokens_ids": self.tokens_ids,
            "tokens_str": self.tokens_str,
            "next_token_id": self.next_token_id,
            "entropy": self.entropy,
            "cross_entropy_next_tok": self.CE_next_tok,
            "cross_entropy_gold_ans": self.CE_gold_ans,
            "cross_entropy_produced_ans": self.CE_gold_ans,
            "prob_eos": self.prob_eos,
            "rank_eos": self.rank_eos,
        }

        if self.hidden_states is not None:
            if isinstance(self.hidden_states, list):
                result["hidden_states"] = [
                    h.tolist() if hasattr(h, 'tolist') else h
                    for h in self.hidden_states
                ]
            else:
                result["hidden_states"] = self.hidden_states

        return result

@dataclass
class CompleteGenerationOutput:
    """Complete generation output with activations and artifacts"""
    # the layers we want to analyze
    layer_indices: list[int]

    input_ids = List[int]
    full_seq_ids = List[int]

    # Reasoning markers
    dp1_idx: int  # first reasoning token
    dp2_idx: Optional[int] = None  # start of final ans (if not cutoff)
    reasoning_length: Optional[int] = None

    # Text outputs
    produced_text: str = ""
    produced_ans: str = ""
    # correct answer according to the dataset
    gold_ans: Optional[str] = None

    # artifacts dataclass for each selected generation timestep
    # will be every token at first, should help identify reasoning
    # step demarcations
    timestep_artifacts: list[TimestepArtifacts] = field(default_factory=list)

    # TODO: windows, numsteps once clear steps demarcations are identified
    # group tokens/timesteps by step window, and keep track of the number of steps

    metadata: Dict[str, Any] = field(default_factory=list)

    def to_dict(self, minimal: bool = False) -> Dict[str, Any]
        """Convert self to dict for storage/serialization
        Args:
            minimal: if True, only include core fields:
        (produced_text, full_seq_ids, dp1_idx, gold_ans"""
        if minimal:
            return {
                "produced_text": self.produced_text,
                "input_ids": self.input_ids,
                "full_seq_ids": self.full_seq_ids,
                "dp1_idx": self.dp1_idx,
                "dp2_idx": self.dp2_idx,
                "gold_ans": self.gold_ans,
                "produced_ans": self.produced_ans,
                "reasoning_length": self.reasoning_length,
            }

        return {
            "produced_text": self.produced_text,
            "input_ids": self.input_ids,
            "full_seq_ids": self.full_seq_ids,
            "dp1_idx": self.dp1_idx,
            "dp2_idx": self.dp2_idx,
            "gold_ans": self.gold_ans,
            "produced_ans": self.produced_ans,
            "reasoning_length": self.reasoning_length,
            "timesteps": [t.to_dict() for t in self.timestep_artifacts],
            "metadata": self.metadata,
        }
