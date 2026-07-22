#!/usr/bin/env python3
"""Prepare, train, evaluate, and inspect the state-handoff LoRA pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.depth_relief.state_handoff_data import (
    TRAINING_CONDITIONS,
    prepare_state_handoff_datasets,
    validate_state_handoff_training_data,
)
from src.experiments.depth_relief.state_handoff_continuation import (
    continuation_status,
    evaluate_continuation_profile,
    prepare_continuation_programs,
)
from src.experiments.depth_relief.state_handoff_evaluation import (
    compare_state_handoff_conditions,
    evaluate_state_handoff_condition,
)
from src.experiments.depth_relief.state_handoff_training import (
    state_handoff_training_status,
    train_state_handoff_condition,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare-data",
            "validate-data",
            "train",
            "evaluate",
            "compare",
            "status",
            "smoke",
            "prepare-continuation",
            "evaluate-continuation",
            "status-continuation",
        ),
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument("--condition", choices=TRAINING_CONDITIONS)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--profile", default="probe")
    args = parser.parse_args()
    if args.command in {"train", "evaluate"} and args.condition is None:
        parser.error(f"{args.command} requires --condition")
    if args.command == "prepare-data":
        result = prepare_state_handoff_datasets(args.run_path)
    elif args.command == "validate-data":
        result = validate_state_handoff_training_data(args.run_path)
    elif args.command == "train":
        result = train_state_handoff_condition(
            args.run_path,
            args.condition,
            max_optimizer_steps=args.max_optimizer_steps,
        )
    elif args.command == "evaluate":
        result = evaluate_state_handoff_condition(
            args.run_path,
            args.condition,
            max_cases=args.max_cases,
        )
    elif args.command == "compare":
        result = compare_state_handoff_conditions(args.run_path)
    elif args.command == "smoke":
        from src.experiments.depth_relief.state_handoff_smoke import run_tiny_smoke

        result = run_tiny_smoke(args.run_path)
    elif args.command == "prepare-continuation":
        result = prepare_continuation_programs(args.run_path, args.profile)
    elif args.command == "evaluate-continuation":
        result = evaluate_continuation_profile(
            args.run_path,
            args.profile,
            max_cases=args.max_cases,
        )
    elif args.command == "status-continuation":
        result = continuation_status(args.run_path)
    else:
        result = state_handoff_training_status(args.run_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
