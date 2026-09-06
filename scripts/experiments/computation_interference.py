#!/usr/bin/env python3
"""Prepare and run the prior-computation interference pilot."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.experiments.computation_interference import (  # noqa: E402
    build_cases,
    freeze_heads,
    measure_case,
    screen_case,
    summarize,
    token_contract,
)
from src.models.hf_loader import load_hf_model_and_tokenizer  # noqa: E402
from src.runtime.artifact_store import append_jsonl, write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402


def prepare(args):
    run = args.run_path
    if run.exists():
        raise ValueError("use a fresh run path")
    source = load_config(args.model_config)
    settings = {
        "screen_count": args.screen_count,
        "test_count": args.test_count,
        "seed": args.seed,
        "head_count": 4,
        "head_batch_size": 8,
    }
    rows = build_cases(
        **{key: settings[key] for key in ("screen_count", "test_count", "seed")}
    )
    write_jsonl(run / "dataset.jsonl", rows)
    config = {
        "model": {
            **source["model"],
            "attn_implementation": "eager",
            "device_map": {"": 0},
        },
        "generation": {
            "max_new_tokens": 96,
            "eos_token_id": source["generation"].get("eos_token_id"),
        },
        "computation_interference": settings,
    }
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(
        run / "manifest.json",
        {
            "dataset_sha256": hashlib.sha256(
                (run / "dataset.jsonl").read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(
                (ROOT / "src/experiments/computation_interference.py").read_bytes()
            ).hexdigest(),
            "status": "prospective_mechanism_discovery",
            "families": len(rows),
            "selection": "all screen families, no failure filtering, four heads by mean edge-ablation effect",
            "test": "new operands and longer correct histories; all four arms and four interventions",
            "primary": "operator-lure final accuracy gain exceeds both matched controls; also compare the value-matched arm",
            "limits": "controlled supplied arithmetic, not natural self-generated reasoning; screen uses first-token margins only",
        },
    )
    return {"families": len(rows)}


def run_experiment(run):
    config = load_config(run)
    manifest = json.loads((run / "manifest.json").read_text())
    for path, key in (
        (run / "dataset.jsonl", "dataset_sha256"),
        (ROOT / "src/experiments/computation_interference.py", "source_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"changed locked input: {path}")
    rows = load_samples(run / "dataset.jsonl")
    settings = config["computation_interference"]
    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    contracts = []
    for row in rows:
        family = [token_contract(tokenizer, row, arm) for arm in row["arms"]]
        if len({len(c["input_ids"]) for c in family}) != 1:
            raise ValueError(f"unmatched context lengths in {row['id']}")
        if len({tuple(c["source_positions"]) for c in family}) != 1:
            raise ValueError("source positions differ across arms")
        contracts.extend(family)
    write_json(
        run / "evaluation/token_contract.json",
        {
            "contexts": len(contracts),
            "max_tokens": max(len(c["input_ids"]) for c in contracts),
            "matched_lengths": True,
            "matched_source_positions": True,
            "number_token_lengths": sorted({len(c["answer_ids"]) for c in contracts}),
        },
    )
    screen_path = run / "evaluation/screen.jsonl"
    screened = load_samples(screen_path) if screen_path.exists() else []
    done = {r["id"] for r in screened}
    for row in rows:
        if row["split"] != "screen" or row["id"] in done:
            continue
        result = screen_case(
            model, tokenizer, row, head_batch_size=settings["head_batch_size"]
        )
        append_jsonl(screen_path, result)
        screened.append(result)
        print(f"screen {row['id']}", flush=True)
    heads = freeze_heads(screened, count=settings["head_count"], seed=settings["seed"])
    write_json(run / "evaluation/heads.json", heads)
    path = run / "evaluation/measurements.jsonl"
    measurements = load_samples(path) if path.exists() else []
    done = {(r["id"], r["arm"], r["condition"]) for r in measurements}
    eos = (
        config["generation"]["eos_token_id"]
        or model.generation_config.eos_token_id
        or tokenizer.eos_token_id
    )
    for row in rows:
        if row["split"] != "test":
            continue
        for arm in row["arms"]:
            for condition in ("baseline", "source", "control_heads", "control_source"):
                if (row["id"], arm, condition) in done:
                    continue
                result = measure_case(
                    model,
                    tokenizer,
                    row,
                    arm,
                    condition,
                    heads,
                    max_new_tokens=config["generation"]["max_new_tokens"],
                    eos_token_id=eos,
                )
                append_jsonl(path, result)
                measurements.append(result)
        print(f"measure {row['id']}", flush=True)
    result = summarize(measurements)
    result["expected_measurements"] = settings["test_count"] * 16
    result["completed_measurements"] = len(measurements)
    write_json(run / "evaluation/summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "reduce"))
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--screen-count", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=26090611)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.model_config is None or min(args.screen_count, args.test_count) < 1:
            parser.error("prepare needs a model config and positive counts")
        result = prepare(args)
    elif args.action == "run":
        result = run_experiment(args.run_path)
    else:
        result = summarize(
            load_samples(args.run_path / "evaluation/measurements.jsonl")
        )
        write_json(args.run_path / "evaluation/summary.json", result)
    print(json.dumps(result, indent=2))
