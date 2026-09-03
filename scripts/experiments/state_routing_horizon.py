#!/usr/bin/env python3
"""Prepare, run, and reduce the state-routing premise pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.state_routing_horizon import (  # noqa: E402
    build_cases,
    evaluate_case,
    select_shard,
    summarize,
    validate_cases,
    validate_token_contract,
)
from src.models.hf_loader import (  # noqa: E402
    load_hf_model_and_tokenizer,
    load_hf_tokenizer,
)
from src.runtime.artifact_store import append_jsonl, write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402


def _settings(run_path: Path) -> dict:
    return dict(load_config(run_path)["state_routing_horizon"])


def _result_path(run_path: Path) -> Path:
    return run_path / "evaluation" / "premise.jsonl"


def prepare(run_path: Path) -> dict:
    settings = _settings(run_path)
    rows = build_cases(count=int(settings["case_count"]), seed=int(settings["seed"]))
    manifest = validate_cases(rows, expected=int(settings["case_count"]))
    data_path = run_path / "dataset.jsonl"
    if data_path.exists():
        existing = load_samples(data_path)
        if existing != rows:
            raise ValueError(f"{data_path} differs from deterministic regeneration")
    else:
        write_jsonl(data_path, rows)
    write_json(run_path / "dataset_manifest.json", manifest)
    return manifest


def validate_tokens(run_path: Path) -> dict:
    config = load_config(run_path)
    tokenizer = load_hf_tokenizer(config["model"])
    rows = load_samples(run_path / "dataset.jsonl")
    contracts = [validate_token_contract(tokenizer, row) for row in rows]
    result = {
        "case_count": len(rows),
        "max_tokens": max(int(row["token_count"]) for row in contracts),
        "min_changed_tokens": min(
            int(row["changed_token_count"]) for row in contracts
        ),
        "max_changed_tokens": max(
            int(row["changed_token_count"]) for row in contracts
        ),
        "single_token_spans": True,
    }
    write_json(run_path / "token_contract.json", result)
    return result


def run(
    run_path: Path, *, shard: int, shards: int, max_cases: int | None
) -> dict:
    config = load_config(run_path)
    settings = dict(config["state_routing_horizon"])
    raw_model = dict(config["model"])
    raw_model["attn_implementation"] = "eager"
    raw_model["device_map"] = {"": 0}
    model, tokenizer = load_hf_model_and_tokenizer(raw_model)
    output = _result_path(run_path)
    completed = {
        str(row["id"]) for row in load_samples(output)
    } if output.exists() else set()
    selected = select_shard(
        load_samples(run_path / "dataset.jsonl"),
        shard=shard,
        shards=shards,
        max_cases=max_cases,
    )
    written = 0
    for row in selected:
        if str(row["id"]) in completed:
            continue
        result = evaluate_case(
            model=model,
            tokenizer=tokenizer,
            row=row,
            layer_stride=int(settings.get("layer_stride", 2)),
            max_new_tokens=int(settings.get("max_new_tokens", 192)),
        )
        append_jsonl(output, result)
        written += 1
    return {"shard": shard, "selected": len(selected), "written": written}


def reduce(run_path: Path) -> dict:
    rows = load_samples(_result_path(run_path))
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("premise results contain duplicate case ids")
    result = summarize(rows)
    write_json(run_path / "evaluation" / "summary.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate-tokens", "run", "reduce"))
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        result = prepare(args.run_path)
    elif args.action == "validate-tokens":
        result = validate_tokens(args.run_path)
    elif args.action == "run":
        result = run(
            args.run_path,
            shard=args.shard,
            shards=args.shards,
            max_cases=args.max_cases,
        )
    else:
        result = reduce(args.run_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
