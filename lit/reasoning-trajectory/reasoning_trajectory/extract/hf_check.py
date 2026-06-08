from __future__ import annotations

import json
import os
import time
from pathlib import Path

from reasoning_trajectory.core.storage import load_config
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.extract.generations import HFLoadConfig, load_hf_causal_lm


@tool(
    "hf-check",
    "extract",
    "Run a small HuggingFace generation and hidden-state check.",
    "rt extract hf-check --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_hf_check.json",
    "reasoning_trajectory.extract.hf_check.hf_inference_check",
    "toolkit/docs/tools/hf-check.md",
)
def hf_inference_check(config_path: str | Path | None = None, out: str | Path | None = None) -> dict:
    cfg = load_config(config_path)
    model_id = cfg.get("model_name", "sshleifer/tiny-gpt2")
    prompt = cfg.get("prompt", "Step 1: compute 2 + 2.\nStep 2:")
    t0 = time.time()
    model, tok = load_hf_causal_lm(
        HFLoadConfig(
            model_id,
            token_env=cfg.get("hf_token_env", "HF_TOKEN"),
            cache_dir=cfg.get("cache_dir"),
            device=cfg.get("device"),
            local_files_only=cfg.get("local_files_only", False),
            use_safetensors=cfg.get("use_safetensors"),
        )
    )
    try:
        import torch
    except ImportError as exc:
        raise ImportError("hf-check requires torch") from exc
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=cfg.get("max_new_tokens", 8), do_sample=False, pad_token_id=tok.eos_token_id)
        forward = model(generated, output_hidden_states=True, use_cache=False)
    text = tok.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    row = {
        "model_id": model_id,
        "generated_preview": text[:120],
        "sequence_tokens": int(generated.shape[1]),
        "hidden_layers": len(forward.hidden_states),
        "hidden_shape_last": list(forward.hidden_states[-1].shape),
        "device": str(next(model.parameters()).device),
        "cuda": bool(torch.cuda.is_available()),
        "elapsed_sec": round(time.time() - t0, 3),
        "token_present": bool(os.environ.get(cfg.get("hf_token_env", "HF_TOKEN"))),
    }
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row
