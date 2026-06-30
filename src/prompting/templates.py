"""Render normalized questions as plain or tokenizer-specific chat prompts."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def build_prompt(
    sample: dict,
    prompt_cfg: dict,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
    """Build the text passed to the model for one sample.

    Args:
        sample: Normalized sample containing a ``question`` field.
        prompt_cfg: System text, instruction, mode, and chat-template options.
        tokenizer: Optional tokenizer used to render chat messages.

    Returns:
        A plain concatenated prompt or rendered chat-template prompt.
    """
    question = sample["question"]

    system = prompt_cfg.get("system", "")
    instruction = prompt_cfg.get("instruction", "")
    mode = prompt_cfg.get("mode", "plain")

    user_text = "\n\n".join(part for part in [instruction, question] if part)

    if mode == "plain":
        return "\n\n".join(part for part in [system, user_text] if part)

    if mode == "chat":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return "\n\n".join(part for part in [system, user_text] if part)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **prompt_cfg.get("chat_template_kwargs", {}),
        )

    raise ValueError(f"Unknown prompt mode: {mode!r}")
