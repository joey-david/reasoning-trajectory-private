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
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        device_map=model_cfg.get("device_map", "auto"),
        torch_dtype=torch_dtype_from_config(model_cfg.get("torch_dtype")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    ).eval()

    return model, tokenizer
