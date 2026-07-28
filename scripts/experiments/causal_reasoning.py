#!/usr/bin/env python3
"""Prepare, inspect, validate, and reduce causal reasoning experiment suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.causal_reasoning.datasets import (  # noqa: E402
    build_experiment_cases,
    validate_experiment_cases,
)
from src.experiments.causal_reasoning.reporting import (  # noqa: E402
    reduce_suite,
)
from src.experiments.depth_relief.benchmark import (  # noqa: E402
    PromptSpec,
    candidate_token_ids,
    format_prompt_spec,
)
from src.models.hf_loader import load_hf_tokenizer  # noqa: E402
from src.orchestration.jobs.causal_reasoning import pending_tasks  # noqa: E402
from src.runtime.artifact_store import write_json  # noqa: E402
from src.runtime.config import load_config  # noqa: E402
from src.runtime.data import load_samples, write_jsonl  # noqa: E402


def _child_runs(suite_path: Path) -> list[Path]:
    config = load_config(suite_path)
    return [
        Path(str(value))
        for value in config["causal_reasoning_suite"]["runs"]
    ]


def prepare_suite(suite_path: Path) -> dict:
    manifests = {}
    for run_path in _child_runs(suite_path):
        config = load_config(run_path)["causal_reasoning"]
        rows = build_experiment_cases(
            str(config["experiment"]),
            count=int(config["case_count"]),
            seed=int(config["seed"]),
        )
        manifest = validate_experiment_cases(
            rows,
            experiment=str(config["experiment"]),
            expected_count=int(config["case_count"]),
        )
        data_path = run_path / "dataset.jsonl"
        if data_path.exists():
            existing = load_samples(data_path)
            existing_manifest = validate_experiment_cases(
                existing,
                experiment=str(config["experiment"]),
                expected_count=int(config["case_count"]),
            )
            if existing_manifest["sha256"] != manifest["sha256"]:
                raise ValueError(
                    f"{data_path} differs from deterministic regeneration"
                )
        else:
            write_jsonl(data_path, rows)
        write_json(run_path / "dataset_manifest.json", manifest)
        manifests[run_path.as_posix()] = manifest
    return {"suite": suite_path.as_posix(), "runs": manifests}


def validate_tokens(suite_path: Path) -> dict:
    suite = load_config(suite_path)
    tokenizer = load_hf_tokenizer(suite["model"])
    result = {}
    for run_path in _child_runs(suite_path):
        config = load_config(run_path)["causal_reasoning"]
        max_tokens = 0
        checkpoint_widths = set()
        for case in load_samples(run_path / "dataset.jsonl"):
            for raw in case["prompts"].values():
                prompt = format_prompt_spec(
                    tokenizer,
                    PromptSpec(
                        text=str(raw["text"]),
                        checkpoint_start=int(raw["checkpoint_start"]),
                        checkpoint_end=int(raw["checkpoint_end"]),
                    ),
                    {"prompt": config["prompt"]},
                )
                candidate_token_ids(
                    tokenizer, prompt.text, case["candidate_symbols"]
                )
                encoded = tokenizer(
                    prompt.text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                max_tokens = max(max_tokens, len(encoded["input_ids"]))
                checkpoint_widths.add(
                    sum(
                        end > prompt.checkpoint_start
                        and start < prompt.checkpoint_end
                        for start, end in encoded["offset_mapping"]
                    )
                )
        limit = int(config["max_sequence_length"])
        if max_tokens > limit:
            raise ValueError(
                f"{run_path} needs {max_tokens} tokens but allows {limit}"
            )
        result[run_path.as_posix()] = {
            "max_prompt_tokens": max_tokens,
            "checkpoint_token_widths": sorted(checkpoint_widths),
            "max_sequence_length": limit,
        }
    return {"model": suite["model"]["name"], "runs": result}


def status(suite_path: Path) -> dict:
    tasks, total, complete = pending_tasks(suite_path)
    return {
        "suite": suite_path.as_posix(),
        "total": total,
        "complete": complete,
        "pending": len(tasks),
        "next_tasks": tasks[:10],
    }


def inspect_prompts(suite_path: Path) -> dict:
    result = {}
    for run_path in _child_runs(suite_path):
        row = load_samples(run_path / "dataset.jsonl")[0]
        result[row["experiment"]] = {
            "id": row["id"],
            "labels": row["labels"],
            "evaluations": row["evaluations"],
            "prompts": {
                name: prompt["text"] for name, prompt in row["prompts"].items()
            },
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("prepare", "validate-tokens", "status", "inspect", "reduce"),
    )
    parser.add_argument("suite", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        result = prepare_suite(args.suite)
    elif args.action == "validate-tokens":
        result = validate_tokens(args.suite)
    elif args.action == "status":
        result = status(args.suite)
    elif args.action == "inspect":
        result = inspect_prompts(args.suite)
    else:
        result = reduce_suite(args.suite)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
