"""Define JSON-facing generation records and per-token diagnostics. Heavy tensors remain external and are referenced by artifact paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HIDDEN_STATE_CONVENTION = (
    "decoder_block_output_pre_final_norm; token_pos predicted by "
    "predict_from_pos=token_pos-1"
)


@dataclass(slots=True)
class TimestepArtifacts:
    """Scalar artifacts for one generated token.

    Convention:
        token_id is the generated token at token_pos.
        Its probability/CE/rank are computed from hidden state at predict_from_pos.

    For autoregressive LMs:
        predict_from_pos = token_pos - 1
    """

    token_id: int
    token_str: str
    token_pos: int
    predict_from_pos: int

    # All per-layer lists follow CompleteGenerationOutput.layer_indices order.
    entropy: list[float] | None = None
    ce_next_token: list[float] | None = None
    rank_next_token: list[int] | None = None

    ce_gold_answer: list[float] | None = None
    rank_gold_answer: list[int] | None = None
    prob_gold_answer: list[float] | None = None

    prob_eos: list[float] | None = None
    rank_eos: list[int] | None = None

    @classmethod
    def from_token(
        cls,
        *,
        token_id: int,
        token_str: str,
        token_pos: int,
    ) -> "TimestepArtifacts":
        """Initialize diagnostics for one generated token.

        Args:
            token_id: Tokenizer ID of the generated token.
            token_str: Decoded representation of the token.
            token_pos: Absolute token position in the prompt-plus-generation sequence.

        Returns:
            A diagnostic record with the autoregressive prediction position set.
        """
        return cls(
            token_id=int(token_id),
            token_str=token_str,
            token_pos=token_pos,
            predict_from_pos=token_pos - 1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the token diagnostics into JSON-compatible values.

        Returns:
            A dictionary containing token identity, positions, and scalar metrics.
        """
        return {
            "token_id": self.token_id,
            "token_str": self.token_str,
            "token_pos": self.token_pos,
            "predict_from_pos": self.predict_from_pos,
            "entropy": self.entropy,
            "ce_next_token": self.ce_next_token,
            "rank_next_token": self.rank_next_token,
            "ce_gold_answer": self.ce_gold_answer,
            "rank_gold_answer": self.rank_gold_answer,
            "prob_gold_answer": self.prob_gold_answer,
            "prob_eos": self.prob_eos,
            "rank_eos": self.rank_eos,
        }


@dataclass(slots=True)
class CompleteGenerationOutput:
    """Complete generation output for one sample/seed/temperature."""

    # Identity
    sample_id: str
    seed: int
    temperature: float
    model_name: str

    # Layer convention
    layer_indices: list[int]
    hidden_state_convention: str

    # Core sequence data
    prompt: str
    input_ids: list[int]
    generated_token_ids: list[int]

    # Decision-point indices
    dp1_idx: int
    dp2_idx: int | None = None
    reasoning_length: int | None = None

    # Text/eval
    produced_text: str = ""
    produced_answer: str | None = None
    gold_answer: str | None = None
    is_correct: bool | None = None

    # Per-token scalar artifacts
    timestep_artifacts: list[TimestepArtifacts] = field(default_factory=list)

    # Heavy artifacts live in separate binary files
    hidden_states_file: str | None = None
