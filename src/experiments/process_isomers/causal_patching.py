"""Run and validate causal patches between symbolic process isomers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from reasoning_trajectory.artifacts import read_generation_rows
from src.experiments.process_isomers.causal_patch_generation import (
    generate_patched_continuation,
    load_activation_vector,
    select_control_donor,
)
from src.experiments.process_isomers.causal_patch_mechanics import (
    PATCH_MODES,
    FirstForwardComponentPatch,
    ProjectionSubspace,
    completion_state_index,
    load_completed_patches,
    load_projection_subspace,
    output_degeneration_reasons,
    resolve_patch_modes,
    resolve_patch_target,
    validate_pair_rows,
    validate_patch_target,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import append_jsonl, write_json
from src.runtime.config import load_config
from src.runtime.data import load_samples


def run_causal_patching(
    run_path: Path,
    *,
    patch_mode: str | None = None,
    max_pairs: int | None = None,
    continuations_per_condition: int | None = None,
) -> None:
    """Execute configured H3 cells over a prepared process-isomer manifest.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        patch_mode: Full-space or projected-subspace patch mode.
        max_pairs: Maximum number of pairs to retain.
        continuations_per_condition: Number of generated replicates for each patch cell.

    Returns:
        None.
    """
    config = load_config(run_path)
    patch_cfg = config["patching"]
    activation_run = Path(patch_cfg["activation_run"])
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())
    configured_max_pairs = int(patch_cfg.get("max_pairs", len(pairs)))
    pairs = pairs[: min(max_pairs or configured_max_pairs, configured_max_pairs)]
    component, layer = resolve_patch_target(patch_cfg)
    patch_modes = resolve_patch_modes(patch_cfg, patch_mode)
    projection = (
        load_projection_subspace(
            Path(patch_cfg["projection_path"]),
            component=component,
            layer=layer,
        )
        if "subspace" in patch_modes
        else None
    )
    rows = {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in read_generation_rows(activation_run)
    }
    validate_pair_rows(pairs, rows)

    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    output_path = run_path / "patching" / "continuations.jsonl"
    completed = load_completed_patches(output_path)
    vector_cache: dict[tuple[str, int, str, int, int], torch.Tensor] = {}
    continuation_count = continuations_per_condition or int(
        patch_cfg.get("continuations_per_condition", 5)
    )

    for pair_index, pair in enumerate(pairs):
        for mode in patch_modes:
            for condition in patch_cfg["conditions"]:
                for continuation in range(continuation_count):
                    key = (
                        int(pair["pair_id"]),
                        mode,
                        str(condition),
                        continuation,
                    )
                    if key in completed:
                        continue
                    seed = (
                        int(patch_cfg.get("base_seed", 0))
                        + pair_index * 100
                        + continuation
                    )
                    record = generate_patched_continuation(
                        model=model,
                        tokenizer=tokenizer,
                        activation_run=activation_run,
                        rows=rows,
                        pairs=pairs,
                        pair=pair,
                        patch_mode=mode,
                        condition=str(condition),
                        continuation=continuation,
                        seed=seed,
                        component=component,
                        layer=layer,
                        projection=projection,
                        patch_cfg=patch_cfg,
                        analysis_cfg=config.get("analysis", {}),
                        vector_cache=vector_cache,
                    )
                    append_jsonl(output_path, record)
                    completed.add(key)


def validate_causal_patching_setup(
    run_path: Path,
    *,
    patch_mode: str | None = None,
    require_activations: bool = True,
) -> Path:
    """Validate every non-generative H3 dependency and write a preflight report.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        patch_mode: Full-space or projected-subspace patch mode.
        require_activations: Whether setup validation requires activation files.

    Returns:
        The path of the written or discovered artifact.
    """
    config = load_config(run_path)
    patch_cfg = config["patching"]
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())
    pairs = pairs[: int(patch_cfg.get("max_pairs", len(pairs)))]
    component, layer = resolve_patch_target(patch_cfg)
    modes = resolve_patch_modes(patch_cfg, patch_mode)
    errors: list[str] = []
    warnings: list[str] = []

    # Validate the experimental design before touching model artifacts. These
    # checks prevent a completed run from encoding the wrong control contrast.
    required_conditions = {
        "baseline",
        "equivalent",
        "position_random",
        "mismatched",
    }
    conditions = [str(condition) for condition in patch_cfg["conditions"]]
    if set(conditions) != required_conditions:
        errors.append(
            f"conditions must be exactly {sorted(required_conditions)}, got {conditions}"
        )
    minimum_pairs = int(patch_cfg.get("minimum_pairs", 20))
    maximum_target_remaining = patch_cfg.get("maximum_target_remaining_tokens")
    require_correct_source_targets = bool(
        patch_cfg.get("require_correct_source_targets", False)
    )
    max_new_tokens = int(patch_cfg.get("max_new_tokens", 768))
    source_tail_headroom = int(patch_cfg.get("minimum_source_tail_headroom", 0))
    if len(pairs) < minimum_pairs:
        errors.append(f"pair yield {len(pairs)} is below minimum {minimum_pairs}")
    for pair in pairs:
        evidence = pair.get("path_evidence")
        if not evidence:
            errors.append(f"pair {pair['pair_id']} lacks path_evidence")
            continue
        if evidence["donor_history_hash"] == evidence["target_history_hash"]:
            errors.append(f"pair {pair['pair_id']} has identical history hashes")
        if float(evidence["normalized_edit_distance"]) < float(
            patch_cfg.get("minimum_path_distance", 0.2)
        ):
            errors.append(f"pair {pair['pair_id']} is below the path-distance minimum")
        for side in ("donor", "target"):
            if pair[side].get("graph_signature") != pair["graph_signature"]:
                errors.append(
                    f"pair {pair['pair_id']} {side} graph signature does not match"
                )
            for step in evidence[f"{side}_steps"]:
                transition = json.loads(step)
                if not transition["added"] and not transition["removed"]:
                    errors.append(
                        f"pair {pair['pair_id']} {side} history contains a no-op"
                    )
        continuation_evidence = pair.get("continuation_evidence")
        if maximum_target_remaining is not None:
            if not continuation_evidence:
                errors.append(
                    f"pair {pair['pair_id']} lacks continuation-budget evidence"
                )
            elif int(continuation_evidence["target_remaining_tokens"]) > int(
                maximum_target_remaining
            ):
                errors.append(f"pair {pair['pair_id']} exceeds the target-tail budget")
        if continuation_evidence and (
            int(continuation_evidence["target_remaining_tokens"]) + source_tail_headroom
            > max_new_tokens
        ):
            errors.append(
                f"pair {pair['pair_id']} leaves less than "
                f"{source_tail_headroom} source-tail tokens of headroom"
            )
        if require_correct_source_targets and (
            not continuation_evidence
            or not continuation_evidence.get("target_source_has_answer")
            or not continuation_evidence.get("target_source_correct")
        ):
            errors.append(
                f"pair {pair['pair_id']} lacks a correct source target answer"
            )
    pair_audit = None
    audit_path = patch_cfg.get("pair_audit")
    if audit_path:
        try:
            pair_audit = json.loads(Path(audit_path).read_text())
            if int(pair_audit["yield"]["accepted_pairs"]) != len(pairs):
                errors.append("pair audit yield does not match the active manifest")
            if float(pair_audit["path_distance"]["minimum"]) < float(
                patch_cfg.get("minimum_path_distance", 0.2)
            ):
                errors.append("pair audit contains below-threshold path distance")
            if maximum_target_remaining is not None and int(
                pair_audit["target_source_tail"]["maximum_tokens"]
            ) > int(maximum_target_remaining):
                errors.append("pair audit exceeds the target-tail budget")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            errors.append(f"pair audit validation failed: {error}")

    projection = None
    projection_report = None
    # Subspace patches are only interpretable when the projection provenance
    # matches the exact component/layer used by this run.
    if "subspace" in modes:
        try:
            projection = load_projection_subspace(
                Path(patch_cfg["projection_path"]),
                component=component,
                layer=layer,
            )
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            errors.append(f"projection validation failed: {error}")
        report_path = patch_cfg.get("projection_report")
        if report_path:
            try:
                projection_report = json.loads(Path(report_path).read_text())
                if projection_report.get("component") != component:
                    errors.append("projection report component does not match H3")
                if int(projection_report.get("layer")) != layer:
                    errors.append("projection report layer does not match H3")
                projected_auc = float(
                    projection_report["evaluation"]["projected_cosine_auc"]
                )
                if projected_auc < float(patch_cfg.get("minimum_projection_auc", 0.9)):
                    errors.append(
                        f"projection AUC {projected_auc:.3f} is below threshold"
                    )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                errors.append(f"projection report validation failed: {error}")

    activation_run = Path(patch_cfg["activation_run"])
    generation_index = activation_run / "generation" / "generations.jsonl"
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    activation_complete = generation_index.exists()
    reconstruction_diagnostics: list[dict[str, float]] = []
    if activation_complete:
        rows = {
            (str(row["sample_id"]), int(row["seed"])): row
            for row in read_generation_rows(activation_run)
        }
        try:
            validate_pair_rows(pairs, rows)
            for pair in pairs:
                target = pair["target"]
                target_key = (str(target["sample_id"]), int(target["seed"]))
                for condition in ("equivalent", "position_random", "mismatched"):
                    select_control_donor(
                        condition=condition,
                        pair=pair,
                        pairs=pairs,
                        rows=rows,
                        target_row=rows[target_key],
                    )
        except ValueError as error:
            errors.append(str(error))
        if projection is not None and not errors:
            # Test reconstruction on every actual donor/target vector, not only
            # on the training summary stored with the projection.
            cache: dict[tuple[str, int, str, int, int], torch.Tensor] = {}
            for pair in pairs:
                donor = pair["donor"]
                target = pair["target"]
                donor_key = (str(donor["sample_id"]), int(donor["seed"]))
                target_key = (str(target["sample_id"]), int(target["seed"]))
                donor_index = completion_state_index(donor, rows[donor_key])
                target_index = completion_state_index(target, rows[target_key])
                donor_vector = load_activation_vector(
                    activation_run=activation_run,
                    rows=rows,
                    key=donor_key,
                    state_index=donor_index,
                    component=component,
                    layer=layer,
                    cache=cache,
                )
                target_vector = load_activation_vector(
                    activation_run=activation_run,
                    rows=rows,
                    key=target_key,
                    state_index=target_index,
                    component=component,
                    layer=layer,
                    cache=cache,
                )
                _, diagnostics = projection.swap(
                    target=target_vector,
                    donor=donor_vector,
                )
                reconstruction_diagnostics.append(diagnostics)
    else:
        message = f"activation replay is not present at {activation_run}"
        if require_activations:
            errors.append(message)
        else:
            warnings.append(message)

    expected_cells_per_pair = len(modes) * len(conditions)
    expected_continuations = (
        len(pairs)
        * expected_cells_per_pair
        * int(patch_cfg.get("continuations_per_condition", 5))
    )
    report = {
        "run": run_path.as_posix(),
        "configuration_valid": not errors
        or (
            not require_activations
            and all("activation replay is not present" in error for error in errors)
        ),
        "ready_for_inference": activation_complete and not errors,
        "errors": errors,
        "warnings": warnings,
        "component": component,
        "layer": layer,
        "patch_modes": list(modes),
        "conditions": conditions,
        "pairs": len(pairs),
        "questions": len({str(pair["target"]["sample_id"]) for pair in pairs}),
        "maximum_target_remaining_tokens": max(
            (
                int(pair["continuation_evidence"]["target_remaining_tokens"])
                for pair in pairs
                if pair.get("continuation_evidence")
            ),
            default=None,
        ),
        "pair_audit": pair_audit,
        "activation_rows": len(rows),
        "expected_cells_per_pair": expected_cells_per_pair,
        "expected_continuations": expected_continuations,
        "projection": (
            {
                "path": projection.path.as_posix(),
                "rank": projection.rank,
                "output_dimensions": projection.weight.shape[0],
                "input_dimensions": projection.weight.shape[1],
                "condition_number": projection.condition_number,
                "question_disjoint_auc": (
                    projection_report["evaluation"]["projected_cosine_auc"]
                    if projection_report
                    else None
                ),
            }
            if projection
            else None
        ),
        "reconstruction": summarize_reconstruction(reconstruction_diagnostics),
    }
    report_path = run_path / "preflight" / "report.json"
    write_json(report_path, report)
    if errors:
        raise ValueError(
            f"H3 preflight failed with {len(errors)} error(s); see {report_path}"
        )
    return report_path


def summarize_reconstruction(
    diagnostics: list[dict[str, float]],
) -> dict[str, Any] | None:
    """Summarize subspace reconstruction diagnostics across pairs.

    Args:
        diagnostics: Per-pair reconstruction diagnostics.

    Returns:
        The resulting keyed records or metrics.
    """
    if not diagnostics:
        return None
    fields = diagnostics[0]
    return {
        "pairs": len(diagnostics),
        **{
            field: {
                "mean": float(
                    sum(record[field] for record in diagnostics) / len(diagnostics)
                ),
                "maximum": float(max(record[field] for record in diagnostics)),
            }
            for field in fields
        },
    }


__all__ = [
    "FirstForwardComponentPatch",
    "PATCH_MODES",
    "ProjectionSubspace",
    "completion_state_index",
    "generate_patched_continuation",
    "load_activation_vector",
    "load_completed_patches",
    "load_projection_subspace",
    "output_degeneration_reasons",
    "resolve_patch_modes",
    "resolve_patch_target",
    "run_causal_patching",
    "select_control_donor",
    "summarize_reconstruction",
    "validate_causal_patching_setup",
    "validate_pair_rows",
    "validate_patch_target",
]
