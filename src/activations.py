from __future__ import annotations

from pathlib import Path
from typing import Any


def activation_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Return activation settings from config.

    Prefer the new shape:

        activations:
          layers: [-1]
          precision: float16

    You can also support old configs with top-level `layers`.
    """
    settings = dict(config.get("activations") or {})
    if "layers" not in settings and "layers" in config:
        settings["layers"] = config["layers"]
    if "precision" not in settings and "activation_storage_dtype" in config:
        settings["precision"] = config["activation_storage_dtype"]
    return settings


def save_activations(
    *,
    activation_dir: Path,
    sample: dict[str, Any],
    seed: int,
    temperature: float,
    hidden_states: Any,
    config: dict[str, Any],
) -> str | None:
    """Save hidden states for selected layers.

    Implement this after text generation works.

    Transformers pattern:
    - Call the model with `output_hidden_states=True`.
    - Read `outputs.hidden_states`, a tuple of tensors.
    - Layer `-1` means the last item in that tuple.

    NumPy/Torch functions:
    - `tensor.detach()` breaks the computation graph.
    - `tensor.cpu()` moves data off GPU.
    - `tensor.to(torch.float16)` changes precision.
    - `numpy.savez_compressed(path, layer_0=array, layer_1=array)`

    For int8 storage later, save a scale next to the quantized array. Do not
    start there; make float16 work first.
    """
    settings = activation_settings(config)
    layers = settings.get("layers", [])
    if not layers or hidden_states is None:
        return None

    sample_id = str(sample.get("id") or sample.get("problem_id") or "sample")
    safe_temp = str(temperature).replace(".", "p")
    path = activation_dir / f"{sample_id}_seed{seed}_temp{safe_temp}.npz"

    # TODO: build a dict like {"layer_-1": numpy_array} and save it.
    # import numpy as np
    # np.savez_compressed(path, **arrays)
    return str(path)
