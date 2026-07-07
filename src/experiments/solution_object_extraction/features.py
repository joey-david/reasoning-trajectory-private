"""Pool selected decoder states over controlled semantic anchors."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models.activation_capture import SelectedLayerCapture
from src.models.introspection import (
    get_base_model,
    get_decoder_layers,
    resolve_layer_indices,
)

from .anchors import anchor_token_range
from .storage import read_jsonl, write_npz


def resolve_model_name(model_cfg: dict[str, Any]) -> str:
    """Resolve an explicit override or the cached local BF16 checkpoint."""
    override = os.environ.get("SOLUTION_OBJECT_MODEL")
    if override:
        return override
    configured = str(model_cfg["name"])
    if configured != "HuggingFaceTB/SmolLM3-3B":
        return configured
    snapshots = sorted(
        Path.home().glob(
            ".cache/huggingface/hub/"
            "models--mlx-community--SmolLM3-3B-bf16/snapshots/*"
        )
    )
    return str(snapshots[-1]) if snapshots else configured


def load_activation_model(
    model_cfg: dict[str, Any],
) -> tuple[Any, Any, torch.device]:
    """Load the configured causal LM on one local or remote device."""
    model_name = resolve_model_name(model_cfg)
    requested = str(model_cfg.get("device", "auto"))
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    dtype_name = str(model_cfg.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=False,
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    ).eval()
    model.to(device)
    return model, tokenizer, device


def capture_bank_features(
    *,
    bank_path: Path,
    output_path: Path,
    model_cfg: dict[str, Any],
    layers: list[int],
    batch_size: int,
) -> dict[str, Any]:
    """Capture and mean-pool selected residual streams for every bank row."""
    rows = read_jsonl(bank_path)
    model, tokenizer, device = load_activation_model(model_cfg)
    decoder_layers = get_decoder_layers(model)
    resolved = resolve_layer_indices(layers, len(decoder_layers))
    vectors: list[np.ndarray] = []
    last_vectors: list[np.ndarray] = []
    last_two_vectors: list[np.ndarray] = []
    pre_anchor_vectors: list[np.ndarray] = []
    delta_vectors: list[np.ndarray] = []
    text_vectors: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    base_model = get_base_model(model)
    for start in tqdm(
        range(0, len(rows), batch_size),
        desc="capture object anchors",
        unit="batch",
    ):
        batch = rows[start : start + batch_size]
        token_ranges = [
            anchor_token_range(tokenizer, row["text"], row["anchor_text"])
            for row in batch
        ]
        encoded = tokenizer(
            [row["text"] for row in batch],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with (
            torch.inference_mode(),
            SelectedLayerCapture(
                decoder_layers=decoder_layers,
                requested_layers=layers,
                resolved_layers=resolved,
            ) as capture,
        ):
            base_model(**encoded, use_cache=False, return_dict=True)
        for row_index, (token_start, token_end) in enumerate(token_ranges):
            pooled = torch.stack(
                [
                    capture.outputs[layer][
                        row_index, token_start : token_end + 1
                    ].float().mean(dim=0)
                    for layer in layers
                ]
            )
            vectors.append(pooled.cpu().numpy())
            last_vectors.append(
                torch.stack(
                    [
                        capture.outputs[layer][row_index, token_end].float()
                        for layer in layers
                    ]
                )
                .cpu()
                .numpy()
            )
            last_two_start = max(token_start, token_end - 1)
            last_two_vectors.append(
                torch.stack(
                    [
                        capture.outputs[layer][
                            row_index, last_two_start : token_end + 1
                        ]
                        .float()
                        .mean(dim=0)
                        for layer in layers
                    ]
                )
                .cpu()
                .numpy()
            )
            pre_anchor_index = max(token_start - 1, 0)
            pre_anchor = torch.stack(
                [
                    capture.outputs[layer][row_index, pre_anchor_index].float()
                    for layer in layers
                ]
            )
            pre_anchor_vectors.append(pre_anchor.cpu().numpy())
            delta_vectors.append((pooled - pre_anchor).cpu().numpy())
            sequence_length = int(encoded["attention_mask"][row_index].sum())
            text_vectors.append(
                torch.stack(
                    [
                        capture.outputs[layer][row_index, :sequence_length]
                        .float()
                        .mean(dim=0)
                        for layer in layers
                    ]
                )
                .cpu()
                .numpy()
            )
            ranges.append((token_start, token_end))
    array = np.stack(vectors).astype(np.float16)
    write_npz(
        output_path,
        h_pool=array,
        h_last=np.stack(last_vectors).astype(np.float16),
        h_last_two=np.stack(last_two_vectors).astype(np.float16),
        h_pre_anchor=np.stack(pre_anchor_vectors).astype(np.float16),
        h_delta=np.stack(delta_vectors).astype(np.float16),
        h_text_mean=np.stack(text_vectors).astype(np.float16),
        record_ids=np.asarray([row["record_id"] for row in rows], dtype=str),
        layers=np.asarray(layers, dtype=np.int32),
        token_ranges=np.asarray(ranges, dtype=np.int32),
    )
    return {
        "records": len(rows),
        "layers": layers,
        "hidden_size": int(array.shape[-1]),
        "model": resolve_model_name(model_cfg),
        "device": str(device),
    }


def load_captured_features(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load pooled vectors, record IDs, and layer identifiers."""
    with np.load(path) as data:
        return (
            data["h_pool"].astype(np.float32),
            data["record_ids"].astype(str),
            data["layers"].astype(int).tolist(),
        )
