"""Persist generation metadata and scalar outputs as JSON/JSONL, with heavy hidden-state arrays in compressed NPZ files. File locks protect shared outputs during multi-process generation."""

from __future__ import annotations

import fcntl
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.runtime.generation_output import CompleteGenerationOutput


def save_generation_output(
    *,
    run_path: Path,
    output: CompleteGenerationOutput,
    hidden_states: Any | None,
    storage_dtype: str,
    component_states: dict[str, Any] | None = None,
) -> CompleteGenerationOutput:
    """Persist one generation and return its artifact-linked output object.

    Args:
        run_path: Run folder that owns the ``generation`` directory.
        output: Completed in-memory generation record.
        hidden_states: Optional ``[tokens, layers, hidden]`` tensor or array.
        storage_dtype: Hidden-state encoding, such as ``float16`` or
            ``int8_scaled``.
        component_states: Optional named attention or MLP activation tensors.

    Returns:
        The same output object, updated with its hidden-state path when saved.
    """
    generation_dir = run_path / "generation"
    hidden_dir = generation_dir / "hidden_states"
    samples_dir = generation_dir / "samples"

    generation_dir.mkdir(parents=True, exist_ok=True)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    stem = artifact_stem(output.sample_id, output.seed, output.temperature)

    if hidden_states is not None:
        hidden_path = hidden_dir / f"{stem}.npz"
        save_hidden_states_npz(
            path=hidden_path,
            hidden_states=hidden_states,
            layer_indices=output.layer_indices,
            storage_dtype=storage_dtype,
            component_states=component_states,
        )
        output.hidden_states_file = hidden_path.relative_to(run_path).as_posix()

    write_json(
        generation_dir / "metadata.json",
        generation_metadata(
            output,
            storage_dtype,
            components=sorted((component_states or {}).keys()),
        ),
    )

    sample_path = samples_dir / f"{sanitize_filename(output.sample_id)}.json"
    if not sample_path.exists():
        write_json(sample_path, sample_record(output))

    row = compact_generation_record(output)
    append_jsonl(
        generation_dir / "generations.jsonl",
        row,
    )

    return output


def generation_metadata(
    output: CompleteGenerationOutput,
    storage_dtype: str,
    *,
    components: list[str] | None = None,
) -> dict[str, Any]:
    """Build run-level metadata for a stored generation.

    Args:
        output: Generation providing model, layer, and convention metadata.
        storage_dtype: Encoding used for persisted activations.
        components: Optional activation component names stored with the run.

    Returns:
        JSON-compatible schema and activation metadata.
    """
    return {
        "schema_version": 2,
        "model_name": output.model_name,
        "layer_indices": output.layer_indices,
        "hidden_state_convention": output.hidden_state_convention,
        "activation_storage_dtype": storage_dtype,
        "components": components or [],
    }


def sample_record(output: CompleteGenerationOutput) -> dict[str, Any]:
    """Build the stable per-sample prompt record.

    Args:
        output: Generation containing the sample's prompt and input metadata.

    Returns:
        JSON-compatible sample identity, prompt, input IDs, answer, and DP1 index.
    """
    return {
        "sample_id": output.sample_id,
        "prompt": output.prompt,
        "input_ids": output.input_ids,
        "gold_answer": output.gold_answer,
        "dp1_idx": output.dp1_idx,
    }


def compact_generation_record(output: CompleteGenerationOutput) -> dict[str, Any]:
    """Build one append-only generation-index row.

    Args:
        output: Completed generation with optional token diagnostics.

    Returns:
        JSON-compatible rollout data, omitting timesteps when none were captured.
    """
    row = {
        "sample_id": output.sample_id,
        "seed": output.seed,
        "temperature": output.temperature,
        "generated_token_ids": output.generated_token_ids,
        "produced_text": output.produced_text,
        "produced_answer": output.produced_answer,
        "is_correct": output.is_correct,
        "dp2_idx": output.dp2_idx,
        "reasoning_length": output.reasoning_length,
        "hidden_states_file": output.hidden_states_file,
    }
    if any(t.entropy is not None for t in output.timestep_artifacts):
        row["timesteps"] = [t.to_dict() for t in output.timestep_artifacts]
    return row


def save_hidden_states_npz(
    *,
    path: Path,
    hidden_states: Any,
    layer_indices: list[int],
    storage_dtype: str,
    component_states: dict[str, Any] | None = None,
) -> None:
    """Store selected hidden states in a compressed NPZ artifact.

    Args:
        path: Destination NPZ path.
        hidden_states: Tensor or array shaped ``[tokens, layers, hidden]``.
        layer_indices: Decoder-layer IDs corresponding to the layer axis.
        storage_dtype: ``float16``, ``float32``, or symmetric ``int8_scaled``.
        component_states: Optional named component tensors stored beside the
            residual states.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    hidden_np = to_numpy(hidden_states)

    arrays: dict[str, np.ndarray] = {
        "layer_indices": np.asarray(layer_indices, dtype=np.int32),
    }

    store_array(arrays, "hidden_states", hidden_np, storage_dtype)
    for component, values in (component_states or {}).items():
        store_array(
            arrays,
            f"component_{component}",
            to_numpy(values),
            storage_dtype,
        )

    np.savez_compressed(path, **arrays)


def load_hidden_states_npz(path: str | Path) -> tuple[np.ndarray, list[int]]:
    """Load and, when necessary, dequantize a hidden-state artifact.

    Args:
        path: NPZ artifact written by :func:`save_hidden_states_npz`.

    Returns:
        A hidden-state array and its ordered decoder-layer IDs.
    """
    with np.load(path) as data:
        layer_indices = data["layer_indices"].astype(int).tolist()

        if "hidden_states" in data:
            return data["hidden_states"].copy(), layer_indices

        if "hidden_states_q" in data and "hidden_states_scale" in data:
            x = data["hidden_states_q"].astype(np.float32)
            scale = data["hidden_states_scale"].astype(np.float32)
            return x * scale[..., None], layer_indices

    raise KeyError(f"No hidden states found in {path}")


def load_component_states_npz(
    path: str | Path,
    component: str,
) -> tuple[np.ndarray, list[int]]:
    """Load and dequantize one captured decoder component.

    Args:
        path: Filesystem path to read from or write to.
        component: Activation component name.

    Returns:
        The computed aligned values described above.
    """
    prefix = f"component_{component}"
    with np.load(path) as data:
        layer_indices = data["layer_indices"].astype(int).tolist()
        if prefix in data:
            return data[prefix].copy(), layer_indices
        quantized = f"{prefix}_q"
        scale_key = f"{prefix}_scale"
        if quantized in data and scale_key in data:
            values = data[quantized].astype(np.float32)
            scale = data[scale_key].astype(np.float32)
            return values * scale[..., None], layer_indices
    raise KeyError(f"Component {component!r} not found in {path}")


def store_array(
    arrays: dict[str, np.ndarray],
    name: str,
    values: np.ndarray,
    storage_dtype: str,
) -> None:
    """Encode one activation tensor under a stable NPZ key prefix.

    Args:
        arrays: Mutable artifact array mapping.
        name: Base key under which to store the array and any scale.
        values: Values to summarize or transform.
        storage_dtype: NumPy dtype used when persisting activations.

    Returns:
        None.
    """
    if storage_dtype == "float16":
        arrays[name] = values.astype(np.float16)
    elif storage_dtype == "float32":
        arrays[name] = values.astype(np.float32)
    elif storage_dtype == "int8_scaled":
        quantized, scale = quantize_int8_symmetric(values)
        arrays[f"{name}_q"] = quantized
        arrays[f"{name}_scale"] = scale
    else:
        raise ValueError(f"Unsupported hidden-state storage dtype: {storage_dtype!r}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL record while holding an exclusive file lock.

    Args:
        path: Destination JSONL file.
        row: JSON-compatible record to append.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Atomically-with-respect-to-cooperating-writers replace a JSON document.

    Args:
        path: Destination JSON file.
        obj: JSON-compatible object to serialize.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def to_numpy(x: Any) -> np.ndarray:
    """Convert a NumPy array or detached tensor-like value to NumPy.

    Args:
        x: Existing array or tensor exposing ``detach``, ``cpu``, and ``numpy``.

    Returns:
        A NumPy view or copied CPU representation of ``x``.
    """
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def quantize_int8_symmetric(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize hidden states symmetrically per token and layer.

    Args:
        x: Floating-point array shaped ``[tokens, layers, hidden]``.

    Returns:
        The int8 array with the same shape and scales shaped ``[tokens, layers]``.
    """
    x = x.astype(np.float32)

    max_abs = np.max(np.abs(x), axis=-1, keepdims=True)
    scale = max_abs / 127.0
    scale = np.where(scale == 0.0, 1.0, scale).astype(np.float32)

    q = np.round(x / scale)
    q = np.clip(q, -127, 127).astype(np.int8)

    return q, scale.squeeze(-1)


def artifact_stem(sample_id: str, seed: int, temperature: float) -> str:
    """Build a filesystem-safe identity for one generation.

    Args:
        sample_id: Source sample identifier.
        seed: Generation seed.
        temperature: Sampling temperature.

    Returns:
        A filename stem containing the sanitized identity fields.
    """
    safe_sample = sanitize_filename(sample_id)
    safe_temp = str(temperature).replace(".", "p")
    return f"{safe_sample}__seed{seed}__temp{safe_temp}"


def sanitize_filename(text: str) -> str:
    """Normalize arbitrary text into a bounded safe filename component.

    Args:
        text: Value to sanitize.

    Returns:
        At most 160 safe characters, or ``"sample"`` when none remain.
    """
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return text[:160] or "sample"
