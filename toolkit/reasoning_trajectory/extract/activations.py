from __future__ import annotations

import numpy as np
import os


def mock_token_hidden_states(tokens: int, layers: int = 4, hidden: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.0, 1.0, tokens)[:, None, None]
    layer = np.linspace(0.0, 0.2, layers)[None, :, None]
    return base + layer + rng.normal(0, 0.01, size=(tokens, layers, hidden))


def hf_token_hidden_states(model, tokenizer, prompt: str, generated_text: str | None = None) -> np.ndarray:
    """Extract token hidden states from a HuggingFace causal LM forward pass."""
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    os.environ.setdefault("TRANSFORMERS_NO_VISION", "1")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("HF activation extraction requires torch") from exc
    text = prompt + (generated_text or "")
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    states = torch.stack([h[0] for h in out.hidden_states], dim=1)
    return states.detach().float().cpu().numpy()
