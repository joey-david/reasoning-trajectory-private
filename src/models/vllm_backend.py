"""Small vLLM adapter for persistent multi-GPU text generation."""

from __future__ import annotations

from typing import Any


def load_vllm_model(model_cfg: dict[str, Any]) -> tuple[Any, Any]:
    """Load one offline vLLM engine and its tokenizer."""
    import torch
    import vllm
    from vllm import LLM

    if not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError(
            f"PyTorch {torch.__version__} lacks float8_e8m0fnu; "
            "install vLLM with its current Torch backend"
        )
    if torch.cuda.device_count() != int(model_cfg.get("required_gpus", 1)):
        raise RuntimeError(
            f"Expected {model_cfg.get('required_gpus', 1)} visible GPUs, "
            f"found {torch.cuda.device_count()}"
        )
    unsupported = [
        torch.cuda.get_device_name(index)
        for index in range(torch.cuda.device_count())
        if torch.cuda.get_device_capability(index) < (7, 5)
    ]
    if unsupported:
        raise RuntimeError(f"FP8 Marlin requires compute capability >=7.5: {unsupported}")

    print(
        f"vLLM {vllm.__version__}, Torch {torch.__version__}, "
        f"GPUs {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )
    engine = LLM(
        model=str(model_cfg["name"]),
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        dtype=str(model_cfg.get("dtype", "auto")),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        max_model_len=int(model_cfg.get("max_model_len", 4096)),
        gpu_memory_utilization=float(model_cfg.get("gpu_memory_utilization", 0.92)),
        language_model_only=bool(model_cfg.get("language_model_only", True)),
        enforce_eager=bool(model_cfg.get("enforce_eager", False)),
    )
    return engine, engine.get_tokenizer()


def generate_vllm(
    engine: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> tuple[str, int]:
    """Render one chat and generate a deterministic vLLM completion."""
    from vllm import SamplingParams

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    outputs = engine.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=max_tokens),
        use_tqdm=False,
    )
    completion = outputs[0].outputs[0]
    return completion.text, len(completion.token_ids)
