#!/usr/bin/env python3
"""Prepare, screen, select, measure, and reduce routing heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.state_routing_heads import (  # noqa: E402
    build_head_cases,
    measure_case,
    measure_free_trace_case,
    screen_case,
    select_heads,
    select_shard,
    summarize_measurements,
    summarize_free_trace,
    token_contract,
    validate_head_cases,
)
from src.models.hf_loader import (  # noqa: E402
    load_hf_model_and_tokenizer,
    load_hf_tokenizer,
)
from src.runtime.artifact_store import append_jsonl, write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402


def _settings(run_path: Path) -> dict:
    return dict(load_config(run_path)["state_routing_heads"])


def prepare(run_path: Path) -> dict:
    settings = _settings(run_path)
    rows = build_head_cases(
        screen_count=int(settings["screen_count"]),
        development_count=int(settings["development_count"]),
        test_count=int(settings["test_count"]),
        seed=int(settings["seed"]),
    )
    expected = {
        "screen": int(settings["screen_count"]),
        "development": int(settings["development_count"]),
        "test": int(settings["test_count"]),
    }
    result = validate_head_cases(rows, expected)
    write_jsonl(run_path / "dataset.jsonl", rows)
    write_json(run_path / "dataset_manifest.json", result)
    return result


def validate_tokens(run_path: Path) -> dict:
    config = load_config(run_path)
    tokenizer = load_hf_tokenizer(config["model"])
    contracts = [
        token_contract(tokenizer, row)
        for row in load_samples(run_path / "dataset.jsonl")
    ]
    result = {
        "case_count": len(contracts),
        "max_tokens": max(row["token_count"] for row in contracts),
        "changed_token_counts": sorted(
            {row["changed_token_count"] for row in contracts}
        ),
        "single_token_spans": True,
    }
    write_json(run_path / "token_contract.json", result)
    return result


def _load_model(run_path: Path):
    config = load_config(run_path)
    raw = dict(config["model"])
    raw["attn_implementation"] = "eager"
    raw["device_map"] = {"": 0}
    return load_hf_model_and_tokenizer(raw)


def screen(run_path: Path, *, shard: int, shards: int) -> dict:
    model, tokenizer = _load_model(run_path)
    settings = _settings(run_path)
    path = run_path / "evaluation/screen.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    selected = select_shard(
        load_samples(run_path / "dataset.jsonl"),
        split="screen",
        shard=shard,
        shards=shards,
    )
    written = 0
    for row in selected:
        if str(row["id"]) in done:
            continue
        append_jsonl(
            path,
            screen_case(
                model=model,
                tokenizer=tokenizer,
                row=row,
                head_batch_size=int(settings["head_batch_size"]),
            ),
        )
        written += 1
    return {"selected": len(selected), "written": written}


def select(run_path: Path) -> dict:
    settings = _settings(run_path)
    result = select_heads(
        load_samples(run_path / "evaluation/screen.jsonl"),
        count=int(settings["selected_head_count"]),
        seed=int(settings["seed"]),
    )
    write_json(run_path / "evaluation/heads.json", result)
    return result


def measure(run_path: Path, *, split: str, shard: int, shards: int) -> dict:
    model, tokenizer = _load_model(run_path)
    settings = _settings(run_path)
    path = run_path / "evaluation/measurements.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    rows = select_shard(
        load_samples(run_path / "dataset.jsonl"),
        split=split,
        shard=shard,
        shards=shards,
    )
    head_spec = json.loads((run_path / "evaluation/heads.json").read_text())
    written = 0
    for row in rows:
        if str(row["id"]) in done:
            continue
        append_jsonl(
            path,
            measure_case(
                model=model,
                tokenizer=tokenizer,
                row=row,
                head_spec=head_spec,
                max_new_tokens=int(settings["max_new_tokens"]),
            ),
        )
        written += 1
    return {"split": split, "selected": len(rows), "written": written}


def reduce(run_path: Path) -> dict:
    rows = load_samples(run_path / "evaluation/measurements.jsonl")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate measurement ids")
    result = summarize_measurements(rows)
    write_json(run_path / "evaluation/summary.json", result)
    return result


def free_route(run_path: Path, *, split: str, shard: int, shards: int) -> dict:
    model, tokenizer = _load_model(run_path)
    path = run_path / "evaluation/free_routing.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    rows = select_shard(
        load_samples(run_path / "dataset.jsonl"),
        split=split,
        shard=shard,
        shards=shards,
    )
    measurements = {
        str(row["id"]): row
        for row in load_samples(run_path / "evaluation/measurements.jsonl")
    }
    head_spec = json.loads((run_path / "evaluation/heads.json").read_text())
    written = 0
    for row in rows:
        if str(row["id"]) in done:
            continue
        append_jsonl(
            path,
            measure_free_trace_case(
                model=model,
                tokenizer=tokenizer,
                row=row,
                measurement=measurements[str(row["id"])],
                head_spec=head_spec,
            ),
        )
        written += 1
    return {"split": split, "selected": len(rows), "written": written}


def reduce_free(run_path: Path) -> dict:
    rows = load_samples(run_path / "evaluation/free_routing.jsonl")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate free-routing ids")
    result = summarize_free_trace(rows)
    write_json(run_path / "evaluation/free_routing_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "validate-tokens",
            "screen",
            "select",
            "measure",
            "reduce",
            "free-route",
            "reduce-free",
        ),
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--split", choices=("development", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.run_path)
    elif args.action == "validate-tokens":
        result = validate_tokens(args.run_path)
    elif args.action == "screen":
        result = screen(args.run_path, shard=args.shard, shards=args.shards)
    elif args.action == "select":
        result = select(args.run_path)
    elif args.action == "measure":
        if args.split is None:
            parser.error("measure requires --split")
        result = measure(
            args.run_path, split=args.split, shard=args.shard, shards=args.shards
        )
    elif args.action == "reduce":
        result = reduce(args.run_path)
    elif args.action == "free-route":
        if args.split is None:
            parser.error("free-route requires --split")
        result = free_route(
            args.run_path, split=args.split, shard=args.shard, shards=args.shards
        )
    else:
        result = reduce_free(args.run_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
