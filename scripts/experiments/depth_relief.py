#!/usr/bin/env python3
"""Prepare, run, inspect, and aggregate causal depth-relief experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.depth_relief.pipeline import (
    analyze,
    analyze_calibration,
    analyze_factorization,
    analyze_routing,
    analyze_qualification,
    calibration_status,
    compare_factorization_runs,
    compare_runs,
    factorization_status,
    prepare,
    prepare_calibration,
    prepare_factorization,
    prepare_routing,
    prepare_qualification,
    qualification_status,
    routing_eligibility,
    routing_status,
    run_mlx,
    run_calibration_mlx,
    run_factorization_mlx,
    run_qualification_mlx,
    status,
    validate_qualification,
    validate_calibration,
    validate_factorization,
    validate_routing,
)
from src.runtime.artifact_store import write_json
from src.experiments.depth_relief.abstraction_pipeline import (
    analyze_interchange,
    analyze_state_abstraction_information,
    analyze_transfer_decoder_matrix,
    interchange_eligibility,
    prepare_interchange,
    prepare_state_abstraction,
    state_abstraction_status,
    validate_state_abstraction,
)
from src.experiments.depth_relief.transfer_pipeline import (
    analyze_handoff,
    analyze_localization,
    analyze_transfer,
    fit_transfer,
    handoff_eligibility,
    prepare_handoff,
    prepare_transfer,
    transfer_status,
    validate_transfer,
)
from src.experiments.depth_relief.explicit_handoff import (
    analyze_explicit_handoff,
    explicit_handoff_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=(
        "prepare",
        "local-mlx",
        "analyze",
        "status",
        "compare",
        "compare-factorization",
        "prepare-qualification",
        "local-mlx-qualification",
        "analyze-qualification",
        "status-qualification",
        "validate-qualification",
        "prepare-calibration",
        "local-mlx-calibration",
        "analyze-calibration",
        "status-calibration",
        "validate-calibration",
        "prepare-factorization",
        "local-mlx-factorization",
        "analyze-factorization",
        "status-factorization",
        "validate-factorization",
        "prepare-routing",
        "analyze-routing",
        "status-routing",
        "validate-routing",
        "routing-eligibility",
        "prepare-transfer",
        "validate-transfer",
        "fit-transfer",
        "analyze-transfer",
        "analyze-localization",
        "handoff-eligibility",
        "prepare-handoff",
        "analyze-handoff",
        "status-transfer",
        "prepare-abstraction",
        "validate-abstraction",
        "analyze-abstraction-information",
        "analyze-transfer-matrix",
        "abstraction-interchange-eligibility",
        "prepare-abstraction-interchange",
        "analyze-abstraction-interchange",
        "status-abstraction",
        "analyze-explicit-handoff",
        "status-explicit-handoff",
    ))
    parser.add_argument("run_paths", type=Path, nargs="+")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    exit_code = 0
    if args.command in {"compare", "compare-factorization"}:
        result = (
            compare_runs(args.run_paths)
            if args.command == "compare"
            else compare_factorization_runs(args.run_paths)
        )
        if args.output:
            write_json(args.output, result)
    elif len(args.run_paths) != 1:
        parser.error(f"{args.command} accepts exactly one run path")
    elif args.command == "prepare":
        result = prepare(args.run_paths[0])
    elif args.command == "prepare-qualification":
        result = prepare_qualification(args.run_paths[0])
    elif args.command == "prepare-calibration":
        result = prepare_calibration(args.run_paths[0])
    elif args.command == "prepare-factorization":
        result = prepare_factorization(args.run_paths[0])
    elif args.command == "prepare-routing":
        result = prepare_routing(args.run_paths[0])
    elif args.command == "routing-eligibility":
        result = routing_eligibility(args.run_paths[0])
        exit_code = 0 if result["eligible"] else 3
    elif args.command == "prepare-transfer":
        result = prepare_transfer(args.run_paths[0])
    elif args.command == "prepare-abstraction":
        result = prepare_state_abstraction(args.run_paths[0])
    elif args.command == "validate-abstraction":
        result = validate_state_abstraction(args.run_paths[0])
    elif args.command == "abstraction-interchange-eligibility":
        result = interchange_eligibility(args.run_paths[0])
        exit_code = 0 if result["eligible"] else 3
    elif args.command == "prepare-abstraction-interchange":
        result = prepare_interchange(args.run_paths[0])
    elif args.command == "prepare-handoff":
        result = prepare_handoff(args.run_paths[0])
    elif args.command == "handoff-eligibility":
        result = handoff_eligibility(args.run_paths[0])
        exit_code = 0 if result["eligible"] else 3
    elif args.command == "validate-qualification":
        result = validate_qualification(args.run_paths[0])
    elif args.command == "validate-calibration":
        result = validate_calibration(args.run_paths[0])
    elif args.command == "validate-factorization":
        result = validate_factorization(args.run_paths[0])
    elif args.command == "validate-routing":
        result = validate_routing(args.run_paths[0])
    elif args.command == "validate-transfer":
        result = validate_transfer(args.run_paths[0])
    elif args.command == "local-mlx":
        result = run_mlx(args.run_paths[0], max_cases=args.max_cases)
    elif args.command == "local-mlx-qualification":
        result = run_qualification_mlx(
            args.run_paths[0], max_cases=args.max_cases
        )
    elif args.command == "local-mlx-calibration":
        result = run_calibration_mlx(
            args.run_paths[0], max_cases=args.max_cases
        )
    elif args.command == "local-mlx-factorization":
        result = run_factorization_mlx(
            args.run_paths[0], max_cases=args.max_cases
        )
    elif args.command == "analyze":
        result = analyze(args.run_paths[0])
    elif args.command == "analyze-qualification":
        result = analyze_qualification(args.run_paths[0])
    elif args.command == "analyze-calibration":
        result = analyze_calibration(args.run_paths[0])
    elif args.command == "analyze-factorization":
        result = analyze_factorization(args.run_paths[0])
    elif args.command == "analyze-routing":
        result = analyze_routing(args.run_paths[0])
    elif args.command == "fit-transfer":
        result = fit_transfer(args.run_paths[0])
    elif args.command == "analyze-transfer":
        result = analyze_transfer(args.run_paths[0])
    elif args.command == "analyze-localization":
        result = analyze_localization(args.run_paths[0])
    elif args.command == "analyze-transfer-matrix":
        result = analyze_transfer_decoder_matrix(args.run_paths[0])
    elif args.command == "analyze-abstraction-information":
        result = analyze_state_abstraction_information(args.run_paths[0])
    elif args.command == "analyze-abstraction-interchange":
        result = analyze_interchange(args.run_paths[0])
    elif args.command == "analyze-handoff":
        result = analyze_handoff(args.run_paths[0])
    elif args.command == "analyze-explicit-handoff":
        result = analyze_explicit_handoff(args.run_paths[0])
    elif args.command == "status-qualification":
        result = qualification_status(args.run_paths[0])
    elif args.command == "status-calibration":
        result = calibration_status(args.run_paths[0])
    elif args.command == "status-factorization":
        result = factorization_status(args.run_paths[0])
    elif args.command == "status-routing":
        result = routing_status(args.run_paths[0])
    elif args.command == "status-transfer":
        result = transfer_status(args.run_paths[0])
    elif args.command == "status-abstraction":
        result = state_abstraction_status(args.run_paths[0])
    elif args.command == "status-explicit-handoff":
        result = explicit_handoff_status(args.run_paths[0])
    else:
        result = status(args.run_paths[0])
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
