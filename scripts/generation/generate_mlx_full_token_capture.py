#!/usr/bin/env python3
"""Generate a local MLX run with every-token final-layer activation deltas."""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import yaml
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.llama import create_attention_mask
from mlx_lm.sample_utils import make_sampler
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.prompting.templates import build_prompt
from src.runtime.artifact_store import append_jsonl, save_hidden_states_npz, write_json
from src.runtime.data import load_samples, write_jsonl
from src.runtime.generation_output import HIDDEN_STATE_CONVENTION
from src.runtime.paths import resolve_repo_path


NUMBER_RE = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"


MODEL_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--mlx-community--SmolLM3-3B-bf16/"
    "snapshots/7bd243308ac4462f1250777fac5c39fd85ab943a"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_path",
        nargs="?",
        default="runs/SmolLM3-3B/screening/frontier_identification/"
        "gsm_symb_40_60_full_token_mlx",
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--max-traces", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    run_path = Path(args.run_path)
    cfg = yaml.safe_load((run_path / "config.yaml").read_text()) or {}
    if args.max_new_tokens is not None:
        cfg["generation"]["max_new_tokens"] = args.max_new_tokens
        cfg["generation"].pop("cap_fallback", None)
    samples = load_samples(run_path / "dataset.jsonl")
    if args.max_items is not None:
        samples = samples[: args.max_items]
    generate_run(run_path, cfg, samples, max_traces=args.max_traces)
    return 0


def generate_run(
    run_path: Path,
    cfg: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    max_traces: int | None = None,
) -> None:
    generation_cfg = cfg["generation"]
    capture_cfg = cfg["capture"]
    analysis_cfg = cfg.get("analysis", {})
    model_path = Path(cfg["model"].get("path") or MODEL_SNAPSHOT)
    model, tokenizer = load(str(model_path), lazy=False)
    sampler = make_sampler(
        temp=float(generation_cfg.get("temperature", 0.0)),
        top_p=float(generation_cfg.get("top_p", 0.0)),
        top_k=int(generation_cfg.get("top_k", 0)),
    )
    existing = load_existing_keys(run_path)
    total = len(samples) * int(generation_cfg.get("num_samples_per_item", 1))
    with tqdm(total=total, desc="mlx full-token generation", unit="trace") as progress:
        written = 0
        for sample_index, sample in enumerate(samples):
            for sample_iter in range(int(generation_cfg.get("num_samples_per_item", 1))):
                if max_traces is not None and written >= max_traces:
                    return
                seed = int(generation_cfg.get("base_seed", 0)) + sample_index * 10_000 + sample_iter
                key = (sample["id"], seed, float(generation_cfg.get("temperature", 0.0)))
                if key in existing:
                    progress.update(1)
                    continue
                progress.set_description(f"generate {sample['id']} seed {seed}")
                row = generate_one(
                    run_path=run_path,
                    cfg=cfg,
                    model=model,
                    tokenizer=tokenizer,
                    sampler=sampler,
                    sample=sample,
                    seed=seed,
                    capture_cfg=capture_cfg,
                    analysis_cfg=analysis_cfg,
                )
                append_jsonl(run_path / "generation/generations.jsonl", row)
                existing.add(key)
                written += 1
                progress.update(1)


def generate_one(
    *,
    run_path: Path,
    cfg: dict[str, Any],
    model: Any,
    tokenizer: Any,
    sampler: Any,
    sample: dict[str, Any],
    seed: int,
    capture_cfg: dict[str, Any],
    analysis_cfg: dict[str, Any],
) -> dict[str, Any]:
    mx.random.seed(seed)
    generation_cfg = cfg["generation"]
    prompt = build_prompt(sample, cfg.get("prompt", {}), tokenizer)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    forced_ids = tokenizer.encode(generation_cfg.get("forced_prefix", ""), add_special_tokens=False)
    prompt_plus_forced = input_ids + forced_ids
    continuation_budget = int(generation_cfg["max_new_tokens"]) - len(forced_ids)
    generated_ids = list(forced_ids)
    generated_text = tokenizer.decode(generated_ids)
    stop_re = re.compile(analysis_cfg.get("produced_answer_regex", r"$^"), re.S)

    for token, _logprobs in generate_step(
        mx.array(prompt_plus_forced),
        model,
        max_tokens=max(1, continuation_budget),
        sampler=sampler,
    ):
        token_id = int(token.item() if hasattr(token, "item") else token)
        generated_ids.append(token_id)
        generated_text += tokenizer.decode([token_id])
        if stop_re.search(generated_text):
            break

    if len(generated_ids) >= int(generation_cfg["max_new_tokens"]):
        fallback = generation_cfg.get("cap_fallback", {})
        generated_ids, generated_text = append_cap_fallback(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            full_ids=input_ids + generated_ids,
            generated_ids=generated_ids,
            generated_text=generated_text,
            fallback=fallback,
        )

    full_ids = input_ids + generated_ids
    hidden_with_prompt = capture_final_layer(model, full_ids)
    prompt_len = len(input_ids)
    hidden_states = hidden_with_prompt[prompt_len:, None, :]
    deltas = activation_deltas(hidden_with_prompt, prompt_len, len(generated_ids))
    hidden_path = save_trace_npz(
        run_path=run_path,
        sample_id=sample["id"],
        seed=seed,
        temperature=float(generation_cfg.get("temperature", 0.0)),
        hidden_states=hidden_states,
        layer_indices=capture_cfg.get("layers", [-1]),
        storage_dtype=capture_cfg.get("activation_storage_dtype", "int8_scaled"),
    )
    write_sample_record(run_path, sample, prompt, input_ids)
    produced_answer = extract_answer(generated_text, analysis_cfg.get("produced_answer_regex"))
    gold_answer = extract_answer(str(sample.get("gold_answer", "")), analysis_cfg.get("gold_answer_regex"))
    think_end_id = int(analysis_cfg.get("think_end_token_id", 128003))
    reasoning_length = generated_ids.index(think_end_id) + 1 if think_end_id in generated_ids else None
    return {
        "sample_id": sample["id"],
        "seed": seed,
        "temperature": float(generation_cfg.get("temperature", 0.0)),
        "generated_token_ids": generated_ids,
        "produced_text": generated_text,
        "produced_answer": produced_answer,
        "is_correct": answers_match(produced_answer, gold_answer),
        "dp2_idx": prompt_len + reasoning_length if reasoning_length is not None else None,
        "reasoning_length": reasoning_length,
        "hidden_states_file": str(hidden_path.relative_to(run_path)),
        "timesteps": timestep_rows(tokenizer, generated_ids, prompt_len, deltas),
    }


def append_cap_fallback(
    *,
    model: Any,
    tokenizer: Any,
    sampler: Any,
    full_ids: list[int],
    generated_ids: list[int],
    generated_text: str,
    fallback: dict[str, Any],
) -> tuple[list[int], str]:
    prefix = str(fallback.get("prefix", ""))
    if not prefix:
        return generated_ids, generated_text
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    generated_ids = generated_ids + prefix_ids
    generated_text += tokenizer.decode(prefix_ids)
    prompt = mx.array(full_ids + prefix_ids)
    for token, _logprobs in generate_step(
        prompt,
        model,
        max_tokens=int(fallback.get("max_new_tokens", 8)),
        sampler=sampler,
    ):
        token_id = int(token.item() if hasattr(token, "item") else token)
        generated_ids.append(token_id)
        generated_text += tokenizer.decode([token_id])
    return generated_ids, generated_text


def capture_final_layer(model: Any, token_ids: list[int]) -> np.ndarray:
    h = model.model.embed_tokens(mx.array(token_ids)[None, :])
    fa_mask = create_attention_mask(h, None)
    swa_mask = (
        create_attention_mask(h, None, window_size=model.model.sliding_window)
        if model.model.swa_idx is not None
        else None
    )
    for layer in model.model.layers:
        mask = swa_mask if layer.use_sliding else fa_mask
        h = layer(h, mask, cache=None)
    h = h.astype(mx.float32)
    mx.eval(h)
    return np.asarray(h[0], dtype=np.float32)


def activation_deltas(hidden: np.ndarray, prompt_len: int, generated_count: int) -> list[float]:
    values: list[float] = []
    for i in range(generated_count):
        current = prompt_len + i
        previous = max(current - 1, 0)
        values.append(float(np.linalg.norm(hidden[current] - hidden[previous])))
    return values


def timestep_rows(tokenizer: Any, token_ids: list[int], prompt_len: int, deltas: list[float]) -> list[dict[str, Any]]:
    rows = []
    for index, token_id in enumerate(token_ids):
        rows.append(
            {
                "token_id": int(token_id),
                "token_idx": index,
                "token_str": tokenizer.decode([int(token_id)]),
                "token_pos": prompt_len + index,
                "predict_from_pos": prompt_len + index - 1,
                "activation_delta": deltas[index],
            }
        )
    return rows


def save_trace_npz(
    *,
    run_path: Path,
    sample_id: str,
    seed: int,
    temperature: float,
    hidden_states: np.ndarray,
    layer_indices: list[int],
    storage_dtype: str,
) -> Path:
    stem = f"{sample_id}__seed{seed}__temp{str(temperature).replace('.', 'p')}"
    path = run_path / "generation/hidden_states" / f"{stem}.npz"
    save_hidden_states_npz(
        path=path,
        hidden_states=hidden_states,
        layer_indices=layer_indices,
        storage_dtype=storage_dtype,
    )
    metadata = {
        "schema_version": 2,
        "model_name": "mlx-community/SmolLM3-3B-bf16",
        "layer_indices": layer_indices,
        "hidden_state_convention": HIDDEN_STATE_CONVENTION,
        "activation_storage_dtype": storage_dtype,
        "components": [],
        "timestep_metrics": ["activation_delta"],
    }
    write_json(run_path / "generation/metadata.json", metadata)
    return path


def write_sample_record(run_path: Path, sample: dict[str, Any], prompt: str, input_ids: list[int]) -> None:
    path = run_path / "generation/samples" / f"{sample['id']}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("input_ids"):
            return
    write_json(
        path,
        {
            "sample_id": sample["id"],
            "prompt": prompt,
            "input_ids": input_ids,
            "gold_answer": sample.get("gold_answer"),
            "dp1_idx": len(input_ids),
        },
    )


def load_existing_keys(run_path: Path) -> set[tuple[str, int, float]]:
    path = run_path / "generation/generations.jsonl"
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((str(row["sample_id"]), int(row["seed"]), float(row["temperature"])))
    return keys


def extract_answer(text: str, pattern: str | None) -> str | None:
    if pattern:
        match = re.search(pattern, text, re.S)
        if match:
            groups = [group for group in match.groups() if group is not None]
            return (groups[-1] if groups else match.group(0)).strip()
    nums = re.findall(NUMBER_RE, text.replace(",", ""))
    return nums[-1] if nums else None


def answers_match(a: str | None, b: str | None) -> bool | None:
    if a is None or b is None:
        return None
    try:
        return Decimal(a.replace(",", "")) == Decimal(b.replace(",", ""))
    except InvalidOperation:
        return a.strip().lower() == b.strip().lower()


if __name__ == "__main__":
    raise SystemExit(main())
