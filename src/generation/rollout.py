from __future__ import annotations

from pathlib import Path

from src.config import run_path
from src.data import load_samples
from src.generation.artifacts import append_jsonl
from src.generation.hf import HFGenerator


def run_generation(config: dict) -> Path:
    import numpy as np

    out_dir = run_path(config) / "generation"
    output_path = out_dir / "generations.jsonl"
    activation_dir = out_dir / "activations"
    generator = HFGenerator(config)
    rows = []

    for sample in load_samples(config):
        for temperature in config.get("temperatures", [0.0]):
            for seed in config.get("seeds", [0]):
                result = generator.generate(sample.prompt, config, int(seed), float(temperature))
                activation_file = activation_dir / f"{sample.id}_seed{seed}_temp{temperature}.npz"
                if result["activations"]:
                    np.savez_compressed(activation_file, **result["activations"])
                rows.append({
                    "sample_id": sample.id,
                    "seed": int(seed),
                    "temperature": float(temperature),
                    "prompt": sample.prompt,
                    "expected_answer": sample.expected_answer,
                    "text": result["text"],
                    "token_ids": result["token_ids"],
                    "logprobs": result["logprobs"],
                    "activation_file": str(activation_file.relative_to(run_path(config))) if result["activations"] else "",
                    "metadata": sample.metadata or {},
                })

    append_jsonl(output_path, rows)
    return output_path
