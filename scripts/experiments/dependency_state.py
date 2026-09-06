#!/usr/bin/env python3
"""Prepare and score dependency-state pilots; generation uses the existing runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reasoning_trajectory.artifacts import read_generation_rows  # noqa: E402
from src.experiments.dependency_state import load_cases, revision_cases, summarize  # noqa: E402
from src.runtime.artifact_store import write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402


def prepare(args) -> dict:
    run = args.run_path
    if (run / "config.yaml").exists() or (run / "dataset.jsonl").exists():
        raise ValueError("run already prepared; use a new run path")
    if args.count < 1:
        raise ValueError("count must be positive")
    build = load_cases if args.experiment == "load" else revision_cases
    cases = build(count=args.count, seed=args.seed)
    model = dict(load_config(args.model_config)["model"])
    if args.model_name:
        model["name"] = args.model_name
    model.update(device_map={"": 0}, attn_implementation="sdpa")
    config = {
        "model": model,
        "dataset": {
            "source": "jsonl",
            "path": str(run / "dataset.jsonl"),
            "adapter": "plain_question",
        },
        "generation": {
            "max_new_tokens": 2048 if args.experiment == "load" else 1024,
            "num_samples_per_item": 1,
            "base_seed": args.seed,
            "temperature": 0.0,
            "eos_token_id": args.eos_token_id,
        },
        "capture": {"enabled": False, "layers": [], "diagnostics": False},
        "prompt": {
            "mode": "chat",
            "instruction": "Work through the calculation, then end with one line: "
            "Answer: [integers in result, in order]. Use integers, not expressions, in that final list.",
        },
        "dependency_state": {
            "experiment": args.experiment,
            "count": args.count,
            "seed": args.seed,
            "status": "diagnostic_pilot",
        },
    }
    write_jsonl(run / "dataset.jsonl", cases)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {
        "cases": len(cases),
        "instances": args.count,
        "seed": args.seed,
        "dataset_sha256": hashlib.sha256(
            (run / "dataset.jsonl").read_bytes()
        ).hexdigest(),
        "experiment_source_sha256": hashlib.sha256(
            (ROOT / "src/experiments/dependency_state.py").read_bytes()
        ).hexdigest(),
        "generation_source_sha256": hashlib.sha256(
            (ROOT / "src/models/generation_pipeline.py").read_bytes()
        ).hexdigest(),
    }
    write_json(run / "evaluation/manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "reduce"))
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--experiment", choices=("load", "revision"))
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--model-name", help="Override the copied model checkpoint path")
    parser.add_argument("--eos-token-id", nargs="+", type=int)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=26090501)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.model_config is None or args.experiment is None:
            parser.error("prepare requires --model-config and --experiment")
        result = prepare(args)
    else:
        config = load_config(args.run_path)
        result = summarize(
            load_samples(args.run_path / "dataset.jsonl"),
            read_generation_rows(args.run_path),
            max_new_tokens=config["generation"]["max_new_tokens"],
        )
        write_json(args.run_path / "evaluation/summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "measurements"}, indent=2)
    )


if __name__ == "__main__":
    main()
