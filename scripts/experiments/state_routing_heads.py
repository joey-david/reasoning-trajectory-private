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
from src.experiments.state_routing_repair import (  # noqa: E402
    STEERING_ALPHAS,
    intermediate_read_sites,
    intermediate_steering_case,
    steering_case,
    steering_eligible,
    summarize_intermediate_steering,
    summarize_steering,
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


def steer(run_path: Path, *, shard: int, shards: int) -> dict:
    model, tokenizer = _load_model(run_path)
    path = run_path / "evaluation/steering.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    dataset = {str(row["id"]): row for row in load_samples(run_path / "dataset.jsonl")}
    measurements = {
        str(row["id"]): row
        for row in load_samples(run_path / "evaluation/measurements.jsonl")
    }
    free_rows = load_samples(run_path / "evaluation/free_routing.jsonl")
    targets = [
        row
        for row in free_rows
        if steering_eligible(
            dataset[str(row["id"])],
            measurements[str(row["id"])],
            row,
        )
    ]
    targets = [row for index, row in enumerate(targets) if index % shards == shard]
    heads = json.loads((run_path / "evaluation/heads.json").read_text())
    written = 0
    for target in targets:
        target_id = str(target["id"])
        if target_id in done:
            continue
        append_jsonl(
            path,
            steering_case(
                model=model,
                tokenizer=tokenizer,
                row=dataset[target_id],
                measurement=measurements[target_id],
                free_row=target,
                head_spec=heads,
                alphas=list(STEERING_ALPHAS),
            ),
        )
        written += 1
    return {"selected": len(targets), "written": written}


def reduce_steering(run_path: Path, *, alpha: float | None) -> dict:
    rows = load_samples(run_path / "evaluation/steering.jsonl")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate steering ids")
    result = summarize_steering(rows, list(STEERING_ALPHAS), selected_alpha=alpha)
    write_json(run_path / "evaluation/steering_summary.json", result)
    return result


def steer_intermediate(
    run_path: Path, *, shard: int, shards: int, alpha: float
) -> dict:
    model, tokenizer = _load_model(run_path)
    path = run_path / "evaluation/intermediate_steering.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    dataset = {str(row["id"]): row for row in load_samples(run_path / "dataset.jsonl")}
    measurements = {
        str(row["id"]): row
        for row in load_samples(run_path / "evaluation/measurements.jsonl")
    }
    free = {
        str(row["id"]): row
        for row in load_samples(run_path / "evaluation/free_routing.jsonl")
    }
    all_sites = [
        (dataset[row_id], site)
        for row_id in free
        for site in intermediate_read_sites(
            dataset[row_id], measurements[row_id], free[row_id]
        )
        if site["split"] == "test"
    ]
    failed = [
        value for value in all_sites if value[1]["saved_operand"] != value[1]["answer"]
    ]
    correct = [
        value for value in all_sites if value[1]["saved_operand"] == value[1]["answer"]
    ]
    sites = failed + sorted(correct, key=lambda value: value[1]["id"])[:100]
    sites = [value for index, value in enumerate(sites) if index % shards == shard]
    heads = json.loads((run_path / "evaluation/heads.json").read_text())
    written = 0
    for row, site in sites:
        if str(site["id"]) in done:
            continue
        append_jsonl(
            path,
            intermediate_steering_case(
                model=model,
                tokenizer=tokenizer,
                row=row,
                measurement=measurements[str(row["id"])],
                site=site,
                head_spec=heads,
                alpha=alpha,
            ),
        )
        written += 1
    return {"selected": len(sites), "written": written}


def reduce_intermediate(run_path: Path) -> dict:
    rows = load_samples(run_path / "evaluation/intermediate_steering.jsonl")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate intermediate-steering ids")
    result = summarize_intermediate_steering(rows)
    write_json(run_path / "evaluation/intermediate_steering_summary.json", result)
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
            "steer",
            "reduce-steering",
            "steer-intermediate",
            "reduce-intermediate",
        ),
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--split", choices=("development", "test"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--alpha", type=float)
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
    elif args.action == "reduce-free":
        result = reduce_free(args.run_path)
    elif args.action == "steer":
        result = steer(args.run_path, shard=args.shard, shards=args.shards)
    elif args.action == "reduce-steering":
        result = reduce_steering(args.run_path, alpha=args.alpha)
    elif args.action == "steer-intermediate":
        if args.alpha is None:
            parser.error("steer-intermediate requires --alpha")
        result = steer_intermediate(
            args.run_path, shard=args.shard, shards=args.shards, alpha=args.alpha
        )
    else:
        result = reduce_intermediate(args.run_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
