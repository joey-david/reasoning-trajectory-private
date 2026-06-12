from __future__ import annotations

from pathlib import Path
from typing import Any


# TODO: Set the model of choice that will generate, the precision
# TODO:
# TODO:
# TODO:
# TODO:
def generate_run(
    run_path: Path, config: dict[str, Any], samples: list[dict[str, Any]]
) -> None:
    """Generate text for selected samples.



    Hugging Face functions you will probably use:
    - `AutoTokenizer.from_pretrained(config["model_name"])`
    - `AutoModelForCausalLM.from_pretrained(config["model_name"], device_map="auto")`
    - `tokenizer(prompt, return_tensors="pt")`
    - `model.generate(...)`
    - `tokenizer.decode(token_ids, skip_special_tokens=True)`

    Torch functions/patterns:
    - `torch.manual_seed(seed)` for reproducibility.
    - `with torch.no_grad():` for inference.
    - `tensor.to(model.device)` to move inputs to the GPU.
    """
    # TODO: import torch and transformers here, not at module import time.
    # Heavy imports should happen only when generation actually runs.
    #
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # import torch
    #
    # tokenizer = AutoTokenizer.from_pretrained(...)
    # model = AutoModelForCausalLM.from_pretrained(...).eval()

    # TODO: create:
    # generation_dir = run_path / "generation"
    # activation_dir = generation_dir / "activations"
    # output_path = generation_dir / "generations.jsonl"
    #
    # TODO: loop over:
    # for sample in samples:
    #     prompt = prompt_from_sample(sample, config)
    #     for seed in config.get("seeds", [0]):
    #         for temperature in config.get("temperatures", [0.0]):
    #             ...
    #
    # TODO: write JSONL rows using `json.dumps(row, ensure_ascii=False)`.
    raise NotImplementedError(
        "Generation is intentionally a teaching placeholder. "
        "Implement src/generation.py one small step at a time."
    )
