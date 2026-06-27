from __future__ import annotations

from transformers import PreTrainedTokenizerBase


def build_prompt(
    sample: dict,
    prompt_cfg: dict,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> str:
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
