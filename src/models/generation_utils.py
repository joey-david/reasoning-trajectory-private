"""Generation seeds, stopping criteria, and sample field resolution."""

from __future__ import annotations

import re
import time
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, StoppingCriteria


def set_seed(seed: int) -> None:
    """Seed PyTorch CPU and available CUDA random generators.

    Args:
        seed: Deterministic generation seed.

    Returns:
        None.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiveGenerationProgress(StoppingCriteria):
    """Update a tqdm progress description while always allowing generation to continue."""

    def __init__(self, progress: Any, prompt_len: int, label: str) -> None:
        """Initialize throttled token-rate reporting.

        Args:
            progress: tqdm-compatible progress object.
            prompt_len: Input length excluded from generated-token counts.
            label: Human-readable rollout identity appended to the description.

        Returns:
            None.
        """
        self.progress = progress
        self.prompt_len = prompt_len
        self.label = label
        self.started_at = time.monotonic()
        self.last_update = 0.0

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs) -> bool:
        """Receive a Transformers stopping callback and update progress.

        Args:
            input_ids: Current prompt-plus-generation token IDs.
            scores: Current generation scores, unused by this tracker.
            **kwargs: Additional Transformers callback values, ignored.

        Returns:
            Always ``False`` so this tracker never stops generation.
        """
        self.update(int(input_ids.shape[-1]))
        return False

    def update(self, seq_len: int, *, force: bool = False) -> None:
        """Refresh the displayed token count and throughput when due.

        Args:
            seq_len: Current total prompt-plus-generation sequence length.
            force: Bypass the one-second display throttle.

        Returns:
            None.
        """
        now = time.monotonic()
        if not force and now - self.last_update < 1.0:
            return
        generated_tokens = max(seq_len - self.prompt_len, 0)
        tok_s = generated_tokens / max(now - self.started_at, 1e-9)
        self.progress.set_description(
            f"generation {generated_tokens} tok {tok_s:.1f} tok/s {self.label}"
        )
        self.last_update = now


class GeneratedTextRegexStop(StoppingCriteria):
    """Stop generation after a regex match has been followed by more decoded text."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_len: int,
        pattern: str,
    ) -> None:
        """Initialize incremental decoded-text matching.

        Args:
            tokenizer: Tokenizer used to decode newly generated IDs.
            prompt_len: Input length marking the start of generated text.
            pattern: Regular expression compiled with dot matching newlines.

        Returns:
            None.
        """
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.pattern = re.compile(pattern, re.S)
        self.last_len = prompt_len
        self.text = ""

    def __call__(self, input_ids: torch.LongTensor, scores: Any, **kwargs) -> bool:
        """Decode new IDs and report whether the completed match can stop generation.

        Args:
            input_ids: Current prompt-plus-generation token IDs.
            scores: Current generation scores, unused by this criterion.
            **kwargs: Additional Transformers callback values, ignored.

        Returns:
            ``True`` once at least one character follows a regex match.
        """
        seq_len = int(input_ids.shape[-1])
        self.text += self.tokenizer.decode(
            input_ids[0, self.last_len : seq_len],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.last_len = seq_len
        match = self.pattern.search(self.text)
        return match is not None and match.end() < len(self.text)


def sample_id_from_sample(sample: dict[str, Any]) -> str:
    """Resolve a normalized sample identifier through supported fallback fields.

    Args:
        sample: Dataset sample containing one of the supported identity fields.

    Returns:
        String sample ID, falling back to ``"sample"``.
    """
    return str(
        sample.get("id")
        or sample.get("problem_id")
        or sample.get("sample_id")
        or "sample"
    )


def gold_answer_from_sample(sample: dict[str, Any]) -> str | None:
    """Resolve a gold answer through supported dataset field names.

    Args:
        sample: Dataset sample with an optional expected answer.

    Returns:
        String answer or ``None`` when no answer field is populated.
    """
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
    """Resolve text to one tokenizer ID, retrying with a leading space.

    Args:
        tokenizer: Tokenizer used by the model.
        text: Candidate answer string.

    Returns:
        The sole token ID when either encoding is one token, otherwise ``None``.
    """
    if text is None:
        return None

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != 1:
        token_ids = tokenizer.encode(" " + text, add_special_tokens=False)

    return int(token_ids[0]) if len(token_ids) == 1 else None
