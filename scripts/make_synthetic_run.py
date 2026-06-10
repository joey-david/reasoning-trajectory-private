#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.common import extract_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small 50/50 synthetic run for UI and analysis smoke tests.")
    parser.add_argument("--run-path", default="runs/synthetic/half_right_half_wrong")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--tokens", type=int, default=36)
    args = parser.parse_args()
    write_run(Path(args.run_path), args.samples, args.tokens)
    print(args.run_path)


def write_run(run_path: Path, samples: int, tokens: int) -> None:
    import numpy as np

    rng = np.random.default_rng(7)
    run_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": "synthetic/half-right-half-wrong",
        "dataset_path": "datasets/sheep.jsonl",
        "max_new_tokens": tokens,
        "seeds": list(range(samples)),
        "temperatures": [0.7],
        "layers": [0, 4, 8],
    }
    (run_path / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    gen_dir = run_path / "generation"
    act_dir = gen_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(samples):
        success = idx % 2 == 0
        answer = "9" if success else str(7 + (idx % 5))
        text = f"We keep the animals that did not run away. Therefore \\\\boxed{{{answer}}}."
        arrays = {}
        attractor = np.array([2.0, -1.0, 0.7]) if success else np.array([-1.8, 1.3, -0.4])
        for layer in config["layers"]:
            steps = np.linspace(0, 1, tokens)[:, None]
            noise = rng.normal(0, 0.25, size=(tokens, 24))
            path = np.zeros((tokens, 24))
            path[:, :3] = steps * attractor * (1 + layer / 10)
            arrays[str(layer)] = path + noise
        file_name = f"sample_{idx}_seed{idx}_temp0.7.npz"
        np.savez_compressed(act_dir / file_name, **arrays)
        rows.append({
            "sample_id": f"sample_{idx}",
            "seed": idx,
            "temperature": 0.7,
            "prompt": "Solve step by step: A farmer has 17 sheep. All but 9 run away. How many sheep are left?",
            "expected_answer": "9",
            "text": text,
            "token_ids": list(range(tokens)),
            "token_texts": text.split(),
            "logprobs": (-rng.random(tokens) * 2).tolist(),
            "predicted_answer": extract_answer(text),
            "success": success,
            "activation_file": str((act_dir / file_name).relative_to(run_path)),
            "metadata": {},
        })
    with (gen_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
