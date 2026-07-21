#!/usr/bin/env python3
"""Prepare, validate, and analyze the three ``notes.pdf`` replications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def parse_args() -> argparse.Namespace:
    """Parse one explicit replication operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare-robustness",
        "validate-robustness",
        "analyze-robustness",
        "prepare-symbolic",
        "validate-symbolic",
        "select-symbolic",
        "analyze-symbolic",
        "validate-rl",
        "analyze-rl",
    ):
        command = subparsers.add_parser(name)
        command.add_argument("run", type=Path)
        if name in {"validate-robustness", "validate-symbolic"}:
            command.add_argument("--no-dataset", action="store_true")
        if name == "analyze-rl":
            command.add_argument(
                "--layers",
                type=lambda value: tuple(int(item) for item in value.split(",")),
            )
    prepare_robustness = subparsers.choices["prepare-robustness"]
    prepare_robustness.add_argument("--token-limit", type=int)
    prepare_symbolic = subparsers.choices["prepare-symbolic"]
    prepare_symbolic.add_argument("--candidates-per-cell", type=int)

    status = subparsers.add_parser("checklist")
    status.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/layer_replication_checklist"),
    )

    train = subparsers.add_parser("train-rl")
    train.add_argument("run", type=Path)
    train_group = train.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--layer", type=int)
    train_group.add_argument("--full", action="store_true")
    train.add_argument("--no-resume", action="store_true")

    evaluate = subparsers.add_parser("evaluate-rl")
    evaluate.add_argument("run", type=Path)
    evaluate_group = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_group.add_argument("--layer", type=int)
    evaluate_group.add_argument("--full", action="store_true")
    evaluate_group.add_argument("--base", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute the selected operation and print its structured result."""
    args = parse_args()
    if args.command == "checklist":
        from src.experiments.layer_replications import checklist

        result = checklist.build(
            Path("runs/Qwen2.5-1.5B/replications/lad_layer_robustness"),
            Path("runs/Qwen2.5-7B/replications/yang_symbolic_mechanisms"),
            Path("runs/Qwen3-1.7B-Base/replications/zhang_single_layer_rl"),
            output=args.output,
        )
    elif args.command in {
        "prepare-robustness",
        "validate-robustness",
        "analyze-robustness",
    }:
        from src.experiments.layer_replications import robustness

        if args.command == "prepare-robustness":
            result = robustness.prepare_dataset(args.run, token_limit=args.token_limit)
        elif args.command == "validate-robustness":
            result = robustness.validate(args.run, require_dataset=not args.no_dataset)
        else:
            result = robustness.analyze(args.run)
    elif args.command in {
        "prepare-symbolic",
        "validate-symbolic",
        "select-symbolic",
        "analyze-symbolic",
    }:
        from src.experiments.layer_replications import symbolic

        if args.command == "prepare-symbolic":
            result = symbolic.prepare_dataset(
                args.run, candidates_per_cell=args.candidates_per_cell
            )
        elif args.command == "validate-symbolic":
            result = symbolic.validate(args.run, require_dataset=not args.no_dataset)
        elif args.command == "select-symbolic":
            result = symbolic.select_valid_pairs(args.run)
        else:
            from src.experiments.layer_replications import symbolic_analysis

            result = symbolic_analysis.analyze(args.run)
    else:
        from src.experiments.layer_replications import rl_analysis, single_layer_rl

        if args.command == "validate-rl":
            result = single_layer_rl.validate(args.run)
        elif args.command == "analyze-rl":
            result = rl_analysis.analyze(args.run, layers=args.layers)
        elif args.command == "train-rl":
            result = single_layer_rl.train(
                args.run,
                layer=args.layer,
                full=args.full,
                resume=not args.no_resume,
            )
        else:
            result = single_layer_rl.evaluate(
                args.run,
                layer=args.layer,
                full=args.full,
                base=args.base,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
