#!/usr/bin/env python3
"""Run the controlled latent solution-object extraction protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.solution_object_extraction.pipeline import (
    run_analysis,
    run_capture,
    run_causal,
    run_prepare,
    validate_run,
)
from src.experiments.solution_object_extraction.ablations import (
    run_ablation_grid,
    run_ablation_sweep,
)
from src.experiments.solution_object_extraction.nonlinear import (
    run_nonlinear_sweep,
)
from src.experiments.solution_object_extraction.sweeps import (
    run_causal_sweep,
    run_retrieval_sweep,
)
from src.experiments.solution_object_extraction.writer import (
    run_writer_experiment,
)
from src.experiments.solution_object_extraction.validation import (
    validate_improvement,
)


SMALL_RUN = Path("runs/SmolLM3-3B/interventions/solution_object_extraction_small")
MEDIUM_RUN = Path("runs/SmolLM3-3B/interventions/solution_object_extraction_medium")


def main() -> int:
    """Dispatch one explicit, restart-safe experiment stage."""
    parser = argparse.ArgumentParser(
        description="Prepare, capture, analyze, or causally test solution objects."
    )
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "capture",
            "analyze",
            "causal",
            "run",
            "validate",
            "validate-improvement",
            "status",
            "retrieval-sweep",
            "causal-sweep",
            "nonlinear",
            "writer",
            "ablation-grid",
            "ablation-sweep",
            "improve",
        ),
    )
    parser.add_argument("run_path", nargs="?", type=Path, default=SMALL_RUN)
    parser.add_argument(
        "--skip-causal",
        action="store_true",
        help="For analysis-only environments; leaves E/F explicitly incomplete.",
    )
    args = parser.parse_args()
    if args.command == "status":
        for path in (SMALL_RUN, MEDIUM_RUN):
            try:
                print(json.dumps(validate_run(path), indent=2))
            except (FileNotFoundError, ValueError) as error:
                print(f"{path}: {error}")
        return 0
    if args.command == "prepare":
        result = run_prepare(args.run_path)
    elif args.command == "capture":
        result = run_capture(args.run_path)
    elif args.command == "analyze":
        result = run_analysis(args.run_path)
    elif args.command == "causal":
        result = run_causal(args.run_path)
    elif args.command == "validate":
        result = validate_run(args.run_path)
    elif args.command == "validate-improvement":
        result = validate_improvement(args.run_path)
    elif args.command == "retrieval-sweep":
        result = run_retrieval_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "causal-sweep":
        result = run_causal_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "nonlinear":
        result = run_nonlinear_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "writer":
        result = run_writer_experiment(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "ablation-grid":
        result = run_ablation_grid(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "ablation-sweep":
        result = run_ablation_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
    elif args.command == "improve":
        run_retrieval_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
        run_causal_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
        run_nonlinear_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
        run_writer_experiment(
            args.run_path, local=is_local_scale(args.run_path)
        )
        run_ablation_sweep(
            args.run_path, local=is_local_scale(args.run_path)
        )
        result = run_ablation_grid(
            args.run_path, local=is_local_scale(args.run_path)
        )
    else:
        run_prepare(args.run_path)
        run_capture(args.run_path)
        result = run_analysis(args.run_path)
        if not args.skip_causal:
            result = run_causal(args.run_path)
    print(json.dumps(result, indent=2))
    return 0


def is_local_scale(run_path: Path) -> bool:
    """Use the bounded local sweep grid only for the small run."""
    return run_path.name.endswith("_small")


if __name__ == "__main__":
    raise SystemExit(main())
