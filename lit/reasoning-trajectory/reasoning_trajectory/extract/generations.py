from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_trajectory.core.storage import config_hash, git_commit, load_config
from reasoning_trajectory.core.storage import save_jsonl
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.schema import Trajectory
from reasoning_trajectory.core.utils import set_seed
from reasoning_trajectory.extract.answers import answer_correct
from reasoning_trajectory.extract.activations import (
    hf_token_hidden_states,
    mock_token_hidden_states,
)
from reasoning_trajectory.extract.token_steps import steps_from_text


def _mock_generate(prompt: str) -> str:
    return "Step 1: identify the quantity.\nStep 2: compute it carefully.\nStep 3: report the result.\n#### 4"


@dataclass
class HFLoadConfig:
    model_name: str
    token_env: str = "HF_TOKEN"
    cache_dir: str | None = None
    device: str | None = None
    device_map: str | dict[str, Any] | None = None
    dtype: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    use_safetensors: bool | None = None
    attn_implementation: str | None = None


def load_hf_causal_lm(cfg: HFLoadConfig):
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "HuggingFace extraction requires torch and transformers"
        ) from exc
    token = os.environ.get(cfg.token_env)
    kwargs = {
        "token": token,
        "cache_dir": cfg.cache_dir,
        "trust_remote_code": cfg.trust_remote_code,
        "local_files_only": cfg.local_files_only,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **kwargs)
    model_kwargs = dict(kwargs)
    if cfg.use_safetensors is not None:
        model_kwargs["use_safetensors"] = cfg.use_safetensors
    if cfg.dtype:
        model_kwargs["dtype"] = getattr(torch, cfg.dtype)
    if cfg.device_map is not None:
        model_kwargs["device_map"] = cfg.device_map
    if cfg.attn_implementation:
        model_kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs).eval()
    if cfg.device and cfg.device_map is None:
        model = model.to(cfg.device)
    return model, tokenizer


def _hf_generate(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    cfg: dict[str, Any],
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> tuple[str, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("HuggingFace extraction requires torch") from exc
    set_seed(seed)
    if model is None or tokenizer is None:
        model, tokenizer = load_hf_causal_lm(
            HFLoadConfig(
                model_name=model_name,
                token_env=cfg.get("hf_token_env", "HF_TOKEN"),
                cache_dir=cfg.get("cache_dir"),
                device=cfg.get("device"),
                device_map=cfg.get("device_map"),
                dtype=cfg.get("dtype"),
                trust_remote_code=cfg.get("trust_remote_code", False),
                local_files_only=cfg.get("local_files_only", False),
                use_safetensors=cfg.get("use_safetensors"),
                attn_implementation=cfg.get("attn_implementation"),
            )
        )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 0
    with torch.no_grad():
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            for name in ["top_p", "top_k", "min_p", "repetition_penalty"]:
                if name in cfg:
                    gen_kwargs[name] = cfg[name]
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return text, model, tokenizer


@tool(
    "extract",
    "extract",
    "Generate text and save token/step hidden-state trajectories in the shared schema.",
    "rt extract --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30",
    "reasoning_trajectory.extract.generations.extract_from_config",
    "toolkit/docs/tools/extract.md",
    dashboard=True,
)
def extract_from_config(
    config_path: str | Path | None, out: str | Path | None = None
) -> list[Trajectory]:
    cfg = load_config(config_path)
    prompts = cfg.get("prompts") or [
        {"problem_id": "example", "prompt": cfg.get("prompt", "What is 2 squared?")}
    ]
    model_name = cfg.get("model_name", "mock")
    seeds = cfg.get("seeds", [0])
    temperatures = cfg.get("temperatures", [0.0])
    layers = cfg.get("layers")
    pooling = cfg.get("pooling", "mean")
    trajectories: list[Trajectory] = []
    hf_model = None
    hf_tokenizer = None
    if model_name != "mock":
        hf_model, hf_tokenizer = load_hf_causal_lm(
            HFLoadConfig(
                model_name=model_name,
                token_env=cfg.get("hf_token_env", "HF_TOKEN"),
                cache_dir=cfg.get("cache_dir"),
                device=cfg.get("device"),
                device_map=cfg.get("device_map"),
                dtype=cfg.get("dtype"),
                trust_remote_code=cfg.get("trust_remote_code", False),
                local_files_only=cfg.get("local_files_only", False),
                use_safetensors=cfg.get("use_safetensors"),
                attn_implementation=cfg.get("attn_implementation"),
            )
        )
    for item in prompts:
        for seed in seeds:
            for temperature in temperatures:
                if model_name == "mock":
                    final_text = _mock_generate(item["prompt"])
                    token_hidden = mock_token_hidden_states(
                        max(1, len(final_text.split())),
                        cfg.get("mock_layers", 4),
                        cfg.get("mock_hidden", 8),
                        seed,
                    )
                else:
                    final_text, model, tokenizer = _hf_generate(
                        model_name,
                        item["prompt"],
                        cfg.get("max_new_tokens", 128),
                        temperature,
                        seed,
                        cfg,
                        hf_model,
                        hf_tokenizer,
                    )
                    token_hidden = hf_token_hidden_states(
                        model, tokenizer, "", final_text
                    )
                steps = steps_from_text(
                    final_text, token_hidden, layers=layers, pooling=pooling
                )
                expected_answer = item.get(
                    "expected_answer",
                    item.get(
                        "final_answer",
                        cfg.get("expected_answer", cfg.get("final_answer")),
                    ),
                )
                predicted_answer, inferred_correct = answer_correct(
                    final_text, expected_answer
                )
                final_answer = item.get(
                    "final_answer", predicted_answer or cfg.get("final_answer")
                )
                final_correct = item.get(
                    "final_correct", cfg.get("final_correct", inferred_correct)
                )
                trajectories.append(
                    Trajectory(
                        trajectory_id=f"{item.get('problem_id', 'problem')}-{seed}-{temperature}",
                        problem_id=item.get("problem_id", "problem"),
                        dataset=cfg.get("dataset", "manual"),
                        model_name=model_name,
                        prompt=item["prompt"],
                        seed=seed,
                        temperature=temperature,
                        decoding_method="sample" if temperature > 0 else "greedy",
                        final_text=final_text,
                        final_answer=final_answer,
                        final_correct=final_correct,
                        steps=steps,
                        metadata={
                            "created_at": cfg.get("created_at", "config"),
                            "repo_commit": git_commit(),
                            "config_hash": config_hash(cfg),
                            "expected_answer": str(expected_answer)
                            if expected_answer is not None
                            else None,
                            "predicted_answer": predicted_answer,
                        },
                    )
                )
    if out:
        out_path = Path(out)
        target = out_path / "trajectories.jsonl" if out_path.suffix == "" else out_path
        save_jsonl(trajectories, target)
    return trajectories
