from __future__ import annotations

from typing import Any


class AppleMLXGenerator:
    """Apple-only MLX shim for local smoke generation.

    It intentionally lives outside the normal Hugging Face generator because it
    is a test backend for Apple Silicon, not the Linux/NVIDIA target path.
    """

    def __init__(self, config: dict[str, Any]):
        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
            import mlx.core as mx
        except Exception as exc:
            raise RuntimeError(
                "Apple MLX backend requested, but mlx-lm is not installed. "
                "Install it in an Apple Silicon environment or use backend: hf."
            ) from exc
        self.generate_text = generate
        self.make_sampler = make_sampler
        self.mx = mx
        self.model, self.tokenizer = load(config["model_name"])

    def generate(self, prompt: str, config: dict[str, Any], seed: int, temperature: float) -> dict[str, Any]:
        self.mx.random.seed(seed)
        text = self.generate_text(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=int(config.get("max_new_tokens", 256)),
            sampler=self.make_sampler(temp=float(temperature)),
            verbose=False,
        )
        token_ids = self.tokenizer.encode(text)
        token_texts = [str(token_id) for token_id in token_ids]
        return {
            "text": text,
            "token_ids": token_ids,
            "token_texts": token_texts,
            "logprobs": [],
            "activations": {},
        }
