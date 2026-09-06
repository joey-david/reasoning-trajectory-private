#!/usr/bin/env python3
"""Mine and measure natural computed-state routing sites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning_trajectory.artifacts import (  # noqa: E402
    read_generation_rows,
    read_sample_records,
)
from src.datasets.loaders import load_run_samples  # noqa: E402
from src.experiments.natural_state_routing import (  # noqa: E402
    build_natural_cases,
    measure_natural_case,
    natural_token_contract,
    summarize_natural_measurements,
)
from src.models.hf_loader import (  # noqa: E402
    load_hf_model_and_tokenizer,
    load_hf_tokenizer,
)
from src.runtime.artifact_store import append_jsonl, write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402
from src.runtime.paths import resolve_repo_path  # noqa: E402


def _artifacts(run_path: Path):
    generations = read_generation_rows(run_path)
    return (
        generations,
        {str(row["sample_id"]): row for row in generations},
        read_sample_records(run_path),
    )


def prepare(run_path: Path) -> dict:
    config = load_config(run_path)
    generations, _, _ = _artifacts(run_path)
    rows = build_natural_cases(
        generations,
        load_run_samples(run_path, config["dataset"]),
        produced_answer_regex=str(config["analysis"]["produced_answer_regex"]),
        gold_answer_regex=str(config["analysis"]["gold_answer_regex"]),
    )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate natural-routing case IDs")
    write_jsonl(run_path / "evaluation/natural_routing_cases.jsonl", rows)
    result = {
        "schema_version": 1,
        "generation_count": len(generations),
        "case_count": len(rows),
        "final_correct_cases": sum(row["final_correct"] for row in rows),
    }
    write_json(run_path / "evaluation/natural_routing_manifest.json", result)
    return result


def validate_tokens(run_path: Path) -> dict:
    config = load_config(run_path)
    tokenizer = load_hf_tokenizer(config["model"])
    _, generations, samples = _artifacts(run_path)
    valid = []
    failures = {}
    for row in load_samples(run_path / "evaluation/natural_routing_cases.jsonl"):
        sample_id = str(row["sample_id"])
        try:
            contract = natural_token_contract(
                tokenizer,
                samples[sample_id],
                generations[sample_id],
                row,
            )
        except ValueError as error:
            message = str(error)
            failures[message] = failures.get(message, 0) + 1
            continue
        valid.append({**row, "token_contract": contract})
    write_jsonl(run_path / "evaluation/natural_routing_token_cases.jsonl", valid)
    result = {
        "schema_version": 1,
        "candidate_count": len(valid) + sum(failures.values()),
        "valid_count": len(valid),
        "final_correct_valid": sum(row["final_correct"] for row in valid),
        "failures": failures,
    }
    write_json(run_path / "evaluation/natural_routing_token_manifest.json", result)
    return result


def _load_model(run_path: Path):
    config = load_config(run_path)
    model = dict(config["model"])
    model["attn_implementation"] = "eager"
    model["device_map"] = {"": 0}
    return load_hf_model_and_tokenizer(model)


def measure(run_path: Path, *, shard: int, shards: int, limit: int | None) -> dict:
    config = load_config(run_path)
    settings = config["natural_state_routing"]
    rows = load_samples(run_path / "evaluation/natural_routing_token_cases.jsonl")
    rows = sorted(
        rows,
        key=lambda row: (
            not row["final_correct"],
            row["source_equation_index"] - row["target_equation_index"],
            row["id"],
        ),
    )
    if limit is not None:
        rows = rows[:limit]
    rows = [row for index, row in enumerate(rows) if index % shards == shard]
    path = run_path / "evaluation/natural_routing_measurements.jsonl"
    done = {str(row["id"]) for row in load_samples(path)} if path.exists() else set()
    _, generations, samples = _artifacts(run_path)
    heads = json.loads(resolve_repo_path(settings["frozen_heads"]).read_text())
    model, tokenizer = _load_model(run_path)
    written = 0
    for row in rows:
        if str(row["id"]) in done:
            continue
        sample_id = str(row["sample_id"])
        append_jsonl(
            path,
            measure_natural_case(
                model=model,
                tokenizer=tokenizer,
                sample=samples[sample_id],
                generation=generations[sample_id],
                case=row,
                heads=heads,
                alpha=float(settings["alpha"]),
            ),
        )
        written += 1
    return {"selected": len(rows), "written": written}


def reduce(run_path: Path) -> dict:
    result = summarize_natural_measurements(
        load_samples(run_path / "evaluation/natural_routing_measurements.jsonl")
    )
    write_json(run_path / "evaluation/natural_routing_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("prepare", "validate-tokens", "measure", "reduce")
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.run_path)
    elif args.action == "validate-tokens":
        result = validate_tokens(args.run_path)
    elif args.action == "measure":
        result = measure(
            args.run_path, shard=args.shard, shards=args.shards, limit=args.limit
        )
    else:
        result = reduce(args.run_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
