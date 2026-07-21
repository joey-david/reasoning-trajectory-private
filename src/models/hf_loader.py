"""Load Hugging Face causal language models and tokenizers from run configuration."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_dtype_from_config(dtype: str | None):
    """Resolve a configured precision alias to a PyTorch dtype.

    Args:
        dtype: Precision alias, ``"auto"``, or ``None``.

    Returns:
        A PyTorch dtype, ``"auto"``, or ``None`` for ``from_pretrained``.
    """
    if dtype in (None, "auto"):
        return dtype

    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype not in aliases:
        raise ValueError(f"Unsupported torch_dtype: {dtype!r}")

    return aliases[dtype]


def load_hf_model_and_tokenizer(model_cfg: dict):
    """Load and configure a Hugging Face causal LM and tokenizer.

    Args:
        model_cfg: Model name, device, precision, revision, trust, and attention options.

    Returns:
        The evaluation-mode model and tokenizer, with padding configured when possible.
    """
    model_kwargs = {
        "device_map": model_cfg.get("device_map", "auto"),
        "dtype": torch_dtype_from_config(model_cfg.get("dtype")),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("revision"):
        model_kwargs["revision"] = model_cfg["revision"]
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    tokenizer = load_hf_tokenizer(model_cfg)

    try:
        model = load_model(model_cfg["name"], model_kwargs)
    except ImportError:
        if model_kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        print("flash_attention_2 unavailable; retrying with attn_implementation=sdpa")
        model_kwargs["attn_implementation"] = "sdpa"
        model = load_model(model_cfg["name"], model_kwargs)

    return model, tokenizer


def load_hf_tokenizer(model_cfg: dict):
    """Load and configure only the tokenizer described by a model config.

    Args:
        model_cfg: Model name, revision, and remote-code trust options.

    Returns:
        The configured Hugging Face tokenizer.
    """
    tokenizer_kwargs = dict(model_cfg.get("tokenizer_kwargs", {}))
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        revision=model_cfg.get("revision"),
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(name: str, kwargs: dict):
    """Load one pretrained causal language model in evaluation mode.

    Args:
        name: Hugging Face model identifier or local model path.
        kwargs: Keyword arguments forwarded to ``from_pretrained``.

    Returns:
        The loaded evaluation-mode causal language model.
    """
    return AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
