"""Artifact storage for generation outputs.

JSON/JSONL:
    metadata, token ids, text, scalar diagnostics, paths.

NPZ:
    heavy numeric arrays such as hidden states.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.generation_output import CompleteGenerationOutput


def save_generation_output(
    *,
    run_path: Path,
    output: CompleteGenerationOutput,
    hidden_states: torch.Tensor | np.ndarray | None,
    storage_dtype: str,
    write_full_json: bool = True,
) -> CompleteGenerationOutput:
    """Save one generation output and return the updated object.

    Directory layout:
        generation/
          generations.jsonl
          outputs/<sample>__seed...json
          hidden_states/<sample>__seed....npz
    """
    generation_dir = run_path / "generation"
    outputs_dir = generation_dir / "outputs"
    hidden_dir = generation_dir / "hidden_states"

    generation_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(output.sample_id, output.seed, output.temperature)

    if hidden_states is not None:
        hidden_path = hidden_dir / f"{stem}.npz"
        save_hidden_states_npz(
            path=hidden_path,
            hidden_states=hidden_states,
            layer_indices=output.layer_indices,
            storage_dtype=storage_dtype,
        )
        output.hidden_states_file = hidden_path.relative_to(run_path).as_posix()

    if write_full_json:
        json_path = outputs_dir / f"{stem}.json"
        write_json(json_path, output.to_dict(minimal=False))

    append_jsonl(
        generation_dir / "generations.jsonl",
        output.to_dict(minimal=True),
    )

    return output


def save_hidden_states_npz(
    *,
    path: Path,
    hidden_states: torch.Tensor | np.ndarray,
    layer_indices: list[int],
    storage_dtype: str,
) -> None:
    """Save hidden states.

    Shape invariant:
        hidden_states: [T, L, H]
        layer_indices: [L]
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    hidden_np = to_numpy(hidden_states)

    arrays: dict[str, np.ndarray] = {
        "layer_indices": np.asarray(layer_indices, dtype=np.int32),
    }

    if storage_dtype == "float16":
        arrays["hidden_states"] = hidden_np.astype(np.float16)

    elif storage_dtype == "float32":
        arrays["hidden_states"] = hidden_np.astype(np.float32)

    elif storage_dtype == "int8_scaled":
        q, scale = quantize_int8_symmetric(hidden_np)
        arrays["hidden_states_q"] = q
        arrays["hidden_states_scale"] = scale

    else:
        raise ValueError(f"Unsupported hidden-state storage dtype: {storage_dtype!r}")

    np.savez_compressed(path, **arrays)


def load_hidden_states_npz(path: str | Path) -> tuple[np.ndarray, list[int]]:
    data = np.load(path)

    layer_indices = data["layer_indices"].astype(int).tolist()

    if "hidden_states" in data:
        return data["hidden_states"], layer_indices

    if "hidden_states_q" in data and "hidden_states_scale" in data:
        x = data["hidden_states_q"].astype(np.float32)
        scale = data["hidden_states_scale"].astype(np.float32)
        return x * scale[..., None], layer_indices

    raise KeyError(f"No hidden states found in {path}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def quantize_int8_symmetric(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-token/per-layer symmetric int8 quantization.

    x:     [T, L, H]
    q:     [T, L, H]
    scale: [T, L]
    """
    x = x.astype(np.float32)

    max_abs = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = max_abs / 127.0
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)

    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)

    return q, scale.squeeze(-1)


def artifact_stem(sample_id: str, seed: int, temperature: float) -> str:
    safe_sample = sanitize_filename(sample_id)
    safe_temp = str(temperature).replace(".", "p")
    return f"{safe_sample}__seed{seed}__temp{safe_temp}"


def sanitize_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text[:160] or "sample"
