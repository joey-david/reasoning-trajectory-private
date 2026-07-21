"""Run-folder preparation, local MLX execution, aggregation, and status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl

from .benchmark import build_benchmark, build_qualification_benchmark, state_symbols
from .calibration import (
    build_calibration_benchmark,
    evaluate_calibration_case_mlx,
    summarize_calibration_rows,
    summarize_history_execution,
    validate_calibration_case,
)
from .factorization import (
    DIAGNOSTIC_TARGETS,
    build_factorization_benchmark,
    evaluate_factorization_case_mlx,
    summarize_factorization_rows,
    validate_factorization_case,
)
from .metrics import bootstrap_mean_ci, summarize_rows
from .qualification import (
    evaluate_qualification_case_mlx,
    summarize_qualification_rows,
    validate_qualification_case,
)
from .routing import (
    select_routing_cases,
    summarize_routing_rows,
    validate_routing_case,
)


RESULT_PATH = Path("depth_relief/cases.jsonl")
QUALIFICATION_RESULT_PATH = Path("depth_relief/qualification_cases.jsonl")
CALIBRATION_RESULT_PATH = Path("depth_relief/calibration_cases.jsonl")
FACTORIZATION_RESULT_PATH = Path("depth_relief/factorization_cases.jsonl")
ROUTING_RESULT_PATH = Path("depth_relief/routing_cases.jsonl")


def output_path(run_path: Path) -> Path:
    """Return the append-only case result path for one run."""
    return run_path / RESULT_PATH


def qualification_output_path(run_path: Path) -> Path:
    """Return the append-only qualification result path."""
    return run_path / QUALIFICATION_RESULT_PATH


def calibration_output_path(run_path: Path) -> Path:
    """Return the append-only frontier-calibration result path."""
    return run_path / CALIBRATION_RESULT_PATH


def factorization_output_path(run_path: Path) -> Path:
    """Return the append-only state-factorization result path."""
    return run_path / FACTORIZATION_RESULT_PATH


def routing_output_path(run_path: Path) -> Path:
    """Return the append-only explicit-state routing result path."""
    return run_path / ROUTING_RESULT_PATH


def prepare(run_path: Path) -> dict[str, Any]:
    """Materialize the deterministic controlled benchmark inside a run folder."""
    config = load_config(run_path)
    experiment = config.get("depth_relief", {})
    cases = build_benchmark(experiment.get("benchmark", {}))
    write_jsonl(run_path / "dataset.jsonl", cases)
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "families": sorted({case["family"] for case in cases}),
        "bits": sorted({case["bits"] for case in cases}),
        "conditions": [
            "none",
            "partial_1..partial_b-1",
            "gold",
            "self",
            "counterfactual",
            "random",
        ],
    }
    write_json(run_path / "depth_relief/benchmark_manifest.json", manifest)
    return manifest


def prepare_qualification(run_path: Path) -> dict[str, Any]:
    """Materialize the held-out, workload-scaled qualification benchmark."""
    config = load_config(run_path)
    experiment = config.get("depth_relief_qualification", {})
    cases = build_qualification_benchmark(experiment.get("benchmark", {}))
    write_jsonl(run_path / "dataset.jsonl", cases)
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "family": "pointer",
        "bits": sorted({int(case["bits"]) for case in cases}),
        "history_steps": sorted(
            {int(case["history_steps"]) for case in cases}
        ),
        "conditions": ["direct", "none", "gold", "counterfactual", "invalid"],
    }
    write_json(run_path / "depth_relief/qualification_manifest.json", manifest)
    return manifest


def prepare_calibration(run_path: Path) -> dict[str, Any]:
    """Materialize the sentinel-based task-frontier discovery grid."""
    config = load_config(run_path)
    experiment = config.get("depth_relief_calibration", {})
    cases = build_calibration_benchmark(experiment.get("benchmark", {}))
    write_jsonl(run_path / "dataset.jsonl", cases)
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "history_families": sorted({case["history_family"] for case in cases}),
        "final_families": sorted({case["final_family"] for case in cases}),
        "bits": sorted({int(case["bits"]) for case in cases}),
        "history_steps": sorted({int(case["history_steps"]) for case in cases}),
        "sentinels": list(experiment.get("sentinels", ["unknown", "missing"])),
        "conditions": ["direct", "none", "none_alt", "gold", "counterfactual"],
    }
    write_json(run_path / "depth_relief/calibration_manifest.json", manifest)
    return manifest


def prepare_factorization(run_path: Path) -> dict[str, Any]:
    """Materialize the read/update/synthesize/compose behavioral assay."""
    config = load_config(run_path)
    experiment = config.get("state_materialization", {})
    cases = build_factorization_benchmark(experiment.get("benchmark", {}))
    write_jsonl(run_path / "dataset.jsonl", cases)
    manifest = {
        "schema_version": 1,
        "case_count": len(cases),
        "history_families": sorted({case["history_family"] for case in cases}),
        "final_families": sorted({case["final_family"] for case in cases}),
        "formats": sorted({case["format"] for case in cases}),
        "bits": sorted({int(case["bits"]) for case in cases}),
        "state_representations": sorted(
            {str(case.get("state_representation", "decimal")) for case in cases}
        ),
        "state_symbols": [
            list(value)
            for value in sorted(
                {state_symbols(case) for case in cases}
            )
        ],
        "history_steps": sorted({int(case["history_steps"]) for case in cases}),
        "assays": ["read", "update", "synthesize", "compose"],
        "constituent_controls": "one actual-input control per history transition",
        "diagnostic_targets": list(DIAGNOSTIC_TARGETS),
    }
    write_json(run_path / "depth_relief/factorization_manifest.json", manifest)
    return manifest


def prepare_routing(run_path: Path) -> dict[str, Any]:
    """Pin the factorization cases that already pass Read, Update, and Synthesize."""
    config = load_config(run_path)
    experiment = config.get("state_routing", {})
    source_run = Path(str(experiment["source_run"]))
    eligibility = routing_eligibility(source_run)
    if not eligibility["eligible"]:
        raise RuntimeError("Source factorization did not pass the routing gate")
    source_prompt = load_config(source_run).get("state_materialization", {}).get(
        "prompt", {}
    )
    if experiment.get("prompt", {}) != source_prompt:
        raise ValueError("Routing prompt config must exactly match factorization")
    cases = load_samples(source_run / "dataset.jsonl")
    rows = read_factorization_results(source_run)
    selected = select_routing_cases(cases, rows)
    write_jsonl(run_path / "dataset.jsonl", selected)
    manifest = {
        "schema_version": 1,
        "source_run": str(source_run),
        "source_case_count": len(cases),
        "case_count": len(selected),
        "selection": "read AND update AND synthesize unconstrained-correct",
        "conditions": ["materialized", "counterfactual"],
        "formats": sorted({case["format"] for case in selected}),
        "history_families": sorted({case["history_family"] for case in selected}),
        "history_steps": sorted({int(case["history_steps"]) for case in selected}),
        "state_representations": sorted(
            {str(case.get("state_representation", "decimal")) for case in selected}
        ),
    }
    write_json(run_path / "depth_relief/routing_manifest.json", manifest)
    return manifest


def routing_eligibility(run_path: Path) -> dict[str, Any]:
    """Expose the prespecified decision that admits a routing confirmation."""
    path = run_path / "depth_relief/factorization_summary.json"
    if not path.exists():
        raise ValueError(f"Missing factorization summary: {path}")
    summary = json.loads(path.read_text())
    decision = summary["decision"]["serial_integration_failure"]
    return {
        "run_path": str(run_path),
        "eligible": bool(decision["supported"]),
        "eligible_case_count": int(summary["routing_analysis"]["eligible_count"]),
        "checks": decision["checks"],
    }


def validate_qualification(run_path: Path) -> dict[str, Any]:
    """Validate every pinned prompt against its tokenizer without loading a model."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("depth_relief_qualification", {})
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    records = [
        validate_qualification_case(
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        for case in load_samples(run_path / "dataset.jsonl")
    ]
    return {
        "run_path": str(run_path),
        "case_count": len(records),
        "condition_count": sum(record["condition_count"] for record in records),
        "matched_token_count_range": [
            min(record["matched_token_count"] for record in records),
            max(record["matched_token_count"] for record in records),
        ],
        "validated": True,
    }


def validate_calibration(run_path: Path) -> dict[str, Any]:
    """Validate every frontier prompt and its one-token checkpoint substitution."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("depth_relief_calibration", {})
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    records = [
        validate_calibration_case(
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        for case in load_samples(run_path / "dataset.jsonl")
    ]
    return {
        "run_path": str(run_path),
        "case_count": len(records),
        "condition_count": sum(record["condition_count"] for record in records),
        "matched_token_count_range": [
            min(record["matched_token_count"] for record in records),
            max(record["matched_token_count"] for record in records),
        ],
        "checkpoint_token_index_range": [
            min(record["checkpoint_token_index"] for record in records),
            max(record["checkpoint_token_index"] for record in records),
        ],
        "validated": True,
    }


def validate_factorization(run_path: Path) -> dict[str, Any]:
    """Validate every assay prompt against the pinned one-token state contract."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("state_materialization", {})
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    records = [
        validate_factorization_case(
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        for case in load_samples(run_path / "dataset.jsonl")
    ]
    return {
        "run_path": str(run_path),
        "case_count": len(records),
        "condition_count": sum(record["condition_count"] for record in records),
        "token_count_range": [
            min(record["token_count_range"][0] for record in records),
            max(record["token_count_range"][1] for record in records),
        ],
        "validated": True,
    }


def validate_routing(run_path: Path) -> dict[str, Any]:
    """Validate exact state-token substitution and one-token output candidates."""
    from src.models.hf_loader import load_hf_tokenizer

    config = load_config(run_path)
    experiment = config.get("state_routing", {})
    tokenizer = load_hf_tokenizer(config.get("model", {}))
    records = [
        validate_routing_case(tokenizer=tokenizer, case=case, config=experiment)
        for case in load_samples(run_path / "dataset.jsonl")
    ]
    return {
        "run_path": str(run_path),
        "case_count": len(records),
        "condition_count": sum(record["condition_count"] for record in records),
        "token_count_range": [
            min(record["token_count"] for record in records),
            max(record["token_count"] for record in records),
        ],
        "validated": True,
    }


def read_results(run_path: Path) -> list[dict[str, Any]]:
    """Read case results while rejecting duplicate task identities."""
    path = output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate depth-relief case IDs in {path}")
    return rows


def read_qualification_results(run_path: Path) -> list[dict[str, Any]]:
    """Read qualification rows while rejecting duplicate case identities."""
    path = qualification_output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate depth-relief qualification IDs in {path}")
    return rows


def read_calibration_results(run_path: Path) -> list[dict[str, Any]]:
    """Read frontier rows while rejecting duplicate case identities."""
    path = calibration_output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate depth-relief calibration IDs in {path}")
    return rows


def read_factorization_results(run_path: Path) -> list[dict[str, Any]]:
    """Read state-factorization rows while rejecting duplicate case identities."""
    path = factorization_output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate state-factorization IDs in {path}")
    return rows


def read_routing_results(run_path: Path) -> list[dict[str, Any]]:
    """Read routing rows while rejecting duplicate semantic case identities."""
    path = routing_output_path(run_path)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate state-routing IDs in {path}")
    return rows


def analyze(run_path: Path) -> dict[str, Any]:
    """Aggregate completed cases into the durable summary report."""
    summary = summarize_rows(read_results(run_path))
    write_json(run_path / "depth_relief/summary.json", summary)
    return summary


def analyze_qualification(run_path: Path) -> dict[str, Any]:
    """Aggregate qualification behavior and evaluate its prespecified gate."""
    config = load_config(run_path)
    experiment = config.get("depth_relief_qualification", {})
    summary = summarize_qualification_rows(
        read_qualification_results(run_path), experiment.get("gate", {})
    )
    write_json(run_path / "depth_relief/qualification_summary.json", summary)
    return summary


def analyze_calibration(run_path: Path) -> dict[str, Any]:
    """Aggregate the discovery grid and select a held-out frontier candidate."""
    config = load_config(run_path)
    experiment = config.get("depth_relief_calibration", {})
    rows = read_calibration_results(run_path)
    cases = load_samples(run_path / "dataset.jsonl")
    summary = summarize_calibration_rows(rows, experiment.get("discovery_gate", {}))
    summary["history_execution_diagnostic"] = summarize_history_execution(
        rows, cases
    )
    write_json(run_path / "depth_relief/calibration_summary.json", summary)
    return summary


def analyze_factorization(run_path: Path) -> dict[str, Any]:
    """Aggregate the behavioral factorization and its prespecified decision checks."""
    config = load_config(run_path)
    experiment = config.get("state_materialization", {})
    summary = summarize_factorization_rows(
        read_factorization_results(run_path), experiment.get("decision", {})
    )
    write_json(run_path / "depth_relief/factorization_summary.json", summary)
    return summary


def analyze_routing(run_path: Path) -> dict[str, Any]:
    """Aggregate matched factual rescue and counterfactual state steering."""
    config = load_config(run_path)
    experiment = config.get("state_routing", {})
    summary = summarize_routing_rows(
        read_routing_results(run_path), experiment.get("gate", {})
    )
    write_json(run_path / "depth_relief/routing_summary.json", summary)
    return summary


def run_mlx(run_path: Path, *, max_cases: int | None = None) -> dict[str, Any]:
    """Run a resumable local MLX distributional screen without causal interventions."""
    from mlx_lm import load

    from .mlx import evaluate_case_mlx

    config = load_config(run_path)
    experiment = config.get("depth_relief", {})
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path") or model_cfg["name"]
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {row["id"] for row in read_results(run_path)}
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    model, tokenizer = load(str(model_path), lazy=False)
    for index, case in enumerate(pending, 1):
        print(f"mlx depth relief {index}/{len(pending)}: {case['id']}", flush=True)
        row = evaluate_case_mlx(
            model=model,
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        append_jsonl(output_path(run_path), row)
    return analyze(run_path)


def run_qualification_mlx(
    run_path: Path, *, max_cases: int | None = None
) -> dict[str, Any]:
    """Run the behavior-only qualification locally through MLX."""
    from mlx_lm import load

    config = load_config(run_path)
    experiment = config.get("depth_relief_qualification", {})
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path") or model_cfg["name"]
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {row["id"] for row in read_qualification_results(run_path)}
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    model, tokenizer = load(str(model_path), lazy=False)
    for index, case in enumerate(pending, 1):
        print(
            f"mlx depth qualification {index}/{len(pending)}: {case['id']}",
            flush=True,
        )
        row = evaluate_qualification_case_mlx(
            model=model,
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        append_jsonl(qualification_output_path(run_path), row)
    return analyze_qualification(run_path)


def run_calibration_mlx(
    run_path: Path, *, max_cases: int | None = None
) -> dict[str, Any]:
    """Run the frontier calibration locally through the shared MLX scorer."""
    from mlx_lm import load

    config = load_config(run_path)
    experiment = config.get("depth_relief_calibration", {})
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path") or model_cfg["name"]
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {row["id"] for row in read_calibration_results(run_path)}
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    model, tokenizer = load(str(model_path), lazy=False)
    for index, case in enumerate(pending, 1):
        print(
            f"mlx depth calibration {index}/{len(pending)}: {case['id']}",
            flush=True,
        )
        row = evaluate_calibration_case_mlx(
            model=model,
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        append_jsonl(calibration_output_path(run_path), row)
    return analyze_calibration(run_path)


def run_factorization_mlx(
    run_path: Path, *, max_cases: int | None = None
) -> dict[str, Any]:
    """Run the factorized behavioral assay locally through the shared MLX scorer."""
    from mlx_lm import load

    config = load_config(run_path)
    experiment = config.get("state_materialization", {})
    model_cfg = config.get("model", {})
    model_path = model_cfg.get("path") or model_cfg["name"]
    cases = load_samples(run_path / "dataset.jsonl")
    completed = {row["id"] for row in read_factorization_results(run_path)}
    pending = [case for case in cases if case["id"] not in completed]
    if max_cases is not None:
        pending = pending[:max_cases]
    model, tokenizer = load(str(model_path), lazy=False)
    for index, case in enumerate(pending, 1):
        print(
            f"mlx state factorization {index}/{len(pending)}: {case['id']}",
            flush=True,
        )
        row = evaluate_factorization_case_mlx(
            model=model,
            tokenizer=tokenizer,
            case=case,
            config=experiment,
        )
        append_jsonl(factorization_output_path(run_path), row)
    return analyze_factorization(run_path)


def status(run_path: Path) -> dict[str, Any]:
    """Return materialization and completion counts without loading a model."""
    dataset = run_path / "dataset.jsonl"
    cases = load_samples(dataset) if dataset.exists() else []
    results = read_results(run_path)
    return {
        "run_path": str(run_path),
        "prepared": dataset.exists(),
        "total_cases": len(cases),
        "completed_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "summary_exists": (run_path / "depth_relief/summary.json").exists(),
    }


def qualification_status(run_path: Path) -> dict[str, Any]:
    """Return qualification completion counts without loading a model."""
    dataset = run_path / "dataset.jsonl"
    cases = load_samples(dataset) if dataset.exists() else []
    results = read_qualification_results(run_path)
    return {
        "run_path": str(run_path),
        "prepared": dataset.exists(),
        "total_cases": len(cases),
        "completed_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "summary_exists": (
            run_path / "depth_relief/qualification_summary.json"
        ).exists(),
    }


def calibration_status(run_path: Path) -> dict[str, Any]:
    """Return frontier-calibration completion counts without loading a model."""
    dataset = run_path / "dataset.jsonl"
    cases = load_samples(dataset) if dataset.exists() else []
    results = read_calibration_results(run_path)
    return {
        "run_path": str(run_path),
        "prepared": dataset.exists(),
        "total_cases": len(cases),
        "completed_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "summary_exists": (
            run_path / "depth_relief/calibration_summary.json"
        ).exists(),
    }


def factorization_status(run_path: Path) -> dict[str, Any]:
    """Return state-factorization completion counts without loading a model."""
    dataset = run_path / "dataset.jsonl"
    cases = load_samples(dataset) if dataset.exists() else []
    results = read_factorization_results(run_path)
    return {
        "run_path": str(run_path),
        "prepared": dataset.exists(),
        "total_cases": len(cases),
        "completed_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "summary_exists": (
            run_path / "depth_relief/factorization_summary.json"
        ).exists(),
    }


def routing_status(run_path: Path) -> dict[str, Any]:
    """Return routing-confirmation completion counts without loading a model."""
    dataset = run_path / "dataset.jsonl"
    cases = load_samples(dataset) if dataset.exists() else []
    results = read_routing_results(run_path)
    return {
        "run_path": str(run_path),
        "prepared": dataset.exists(),
        "total_cases": len(cases),
        "completed_cases": len(results),
        "remaining_cases": len(cases) - len(results),
        "summary_exists": (run_path / "depth_relief/routing_summary.json").exists(),
    }


def compare_runs(run_paths: list[Path]) -> dict[str, Any]:
    """Compare model runs on their shared case IDs with paired depth relief."""
    if len({str(path) for path in run_paths}) < 2:
        raise ValueError("Model comparison requires at least two run paths")
    indexed = {
        str(path): {str(row["id"]): row for row in read_results(path)}
        for path in run_paths
    }
    shared = set.intersection(*(set(rows) for rows in indexed.values()))
    if not shared:
        raise ValueError("Model runs have no shared completed case IDs")

    def relief(row: dict[str, Any]) -> float:
        conditions = row["conditions"]
        return float(
            conditions["none"]["settling_depth"]
            - conditions["gold"]["settling_depth"]
        )

    def curve_relief(row: dict[str, Any]) -> float:
        conditions = row["conditions"]
        return float(
            conditions["none"]["dtr_jsd_auc"]
            - conditions["gold"]["dtr_jsd_auc"]
        )

    reference_name = str(run_paths[0])

    def compare_metric(metric: Any, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        per_model = {
            name: bootstrap_mean_ci(
                [metric(rows[case_id]) for case_id in sorted(shared)],
                seed=seed + index,
            )
            for index, (name, rows) in enumerate(indexed.items())
        }
        reference = indexed[reference_name]
        paired = {}
        for index, path in enumerate(run_paths[1:], 1):
            name = str(path)
            rows = indexed[name]
            paired[f"{name}_minus_{reference_name}"] = bootstrap_mean_ci(
                [
                    metric(rows[case_id]) - metric(reference[case_id])
                    for case_id in sorted(shared)
                ],
                seed=seed + 100 + index,
            )
        return per_model, paired

    per_model, paired_differences = compare_metric(relief, 100)
    curve_per_model, curve_paired_differences = compare_metric(curve_relief, 300)
    return {
        "schema_version": 1,
        "shared_case_count": len(shared),
        "reference": reference_name,
        "depth_relief_by_model": per_model,
        "paired_depth_relief_differences": paired_differences,
        "jsd_curve_area_relief_by_model": curve_per_model,
        "paired_jsd_curve_area_relief_differences": curve_paired_differences,
    }


def compare_factorization_runs(run_paths: list[Path]) -> dict[str, Any]:
    """Compare factorized behavior across models on exactly shared semantic cases."""
    if len({str(path) for path in run_paths}) < 2:
        raise ValueError("State-factorization comparison requires at least two runs")
    indexed = {
        str(path): {str(row["id"]): row for row in read_factorization_results(path)}
        for path in run_paths
    }
    shared = set.intersection(*(set(rows) for rows in indexed.values()))
    if not shared:
        raise ValueError("State-factorization runs have no shared completed case IDs")

    def correct(row: dict[str, Any], assay: str) -> int:
        return int(row["conditions"][assay]["is_expected_unconstrained"])

    assays = ("read", "update", "synthesize", "compose")
    per_model = {}
    for model_index, (name, rows) in enumerate(indexed.items()):
        shared_rows = [rows[case_id] for case_id in sorted(shared)]
        per_model[name] = {
            "accuracy": {
                assay: bootstrap_mean_ci(
                    [correct(row, assay) for row in shared_rows],
                    seed=800 + model_index * 20 + assay_index,
                )
                for assay_index, assay in enumerate(assays)
            },
            "update_minus_synthesize": bootstrap_mean_ci(
                [
                    correct(row, "update") - correct(row, "synthesize")
                    for row in shared_rows
                ],
                seed=810 + model_index * 20,
            ),
        }

    reference_name = str(run_paths[0])
    reference = indexed[reference_name]
    paired = {}
    for model_index, path in enumerate(run_paths[1:], 1):
        name = str(path)
        rows = indexed[name]
        paired[name] = {
            "accuracy_minus_reference": {
                assay: bootstrap_mean_ci(
                    [
                        correct(rows[case_id], assay)
                        - correct(reference[case_id], assay)
                        for case_id in sorted(shared)
                    ],
                    seed=900 + model_index * 20 + assay_index,
                )
                for assay_index, assay in enumerate(assays)
            },
            "dissociation_minus_reference": bootstrap_mean_ci(
                [
                    (
                        correct(rows[case_id], "update")
                        - correct(rows[case_id], "synthesize")
                    )
                    - (
                        correct(reference[case_id], "update")
                        - correct(reference[case_id], "synthesize")
                    )
                    for case_id in sorted(shared)
                ],
                seed=910 + model_index * 20,
            ),
        }
    return {
        "schema_version": 1,
        "shared_case_count": len(shared),
        "reference": reference_name,
        "per_model": per_model,
        "paired_differences": paired,
    }
