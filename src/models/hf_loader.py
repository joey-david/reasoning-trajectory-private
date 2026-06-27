from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def torch_dtype_from_config(dtype: str | None):
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
    model_kwargs = {
        "device_map": model_cfg.get("device_map", "auto"),
        "torch_dtype": torch_dtype_from_config(model_cfg.get("dtype")),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    if model_cfg.get("revision"):
        model_kwargs["revision"] = model_cfg["revision"]
    if model_cfg.get("attn_implementation"):
        model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    trust_remote_code = model_kwargs["trust_remote_code"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=trust_remote_code,
        revision=model_cfg.get("revision"),
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = load_model(model_cfg["name"], model_kwargs)
    except ImportError:
        if model_kwargs.get("attn_implementation") != "flash_attention_2":
            raise
        print("flash_attention_2 unavailable; retrying with attn_implementation=sdpa")
        model_kwargs["attn_implementation"] = "sdpa"
        model = load_model(model_cfg["name"], model_kwargs)

    return model, tokenizer


def load_model(name: str, kwargs: dict):
    return AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
