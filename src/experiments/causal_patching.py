"""Run full-vector and H4-subspace causal patches between process isomers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch
from transformers import StoppingCriteriaList

from src.analysis.answers import answers_match, extract_answer
from src.analysis.common import read_generation_rows
from src.experiments.replay_capture import load_source_sample
from src.models.generation_pipeline import GeneratedTextRegexStop, set_seed
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.models.introspection import (
    get_decoder_layers,
    get_input_device,
    resolve_layer_indices,
)
from src.runtime.artifact_store import (
    append_jsonl,
    load_component_states_npz,
    write_json,
)
from src.runtime.config import load_config
from src.runtime.data import load_samples


PATCH_MODES = ("full", "subspace")


@dataclass(slots=True)
class ProjectionSubspace:
    """Hold a validated linear map and its Moore-Penrose inverse."""

    path: Path
    weight: torch.Tensor
    pseudoinverse: torch.Tensor
    rank: int
    condition_number: float

    def swap(
        self,
        *,
        target: torch.Tensor,
        donor: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Replace target row-space coordinates with donor coordinates."""
        target = target.float().cpu()
        donor = donor.float().cpu()
        full_delta = donor - target
        coordinate_delta = self.weight @ full_delta
        subspace_delta = self.pseudoinverse @ coordinate_delta
        reconstructed = target + subspace_delta
        coordinate_residual = self.weight @ reconstructed - self.weight @ donor
        projected_delta = self.pseudoinverse @ (self.weight @ subspace_delta)
        orthogonal_leakage = subspace_delta - projected_delta
        return reconstructed, {
            "full_delta_norm": float(torch.linalg.vector_norm(full_delta)),
            "subspace_delta_norm": float(torch.linalg.vector_norm(subspace_delta)),
            "retained_delta_fraction": float(
                torch.linalg.vector_norm(subspace_delta)
                / torch.linalg.vector_norm(full_delta).clamp_min(1e-8)
            ),
            "coordinate_reconstruction_relative_residual": float(
                torch.linalg.vector_norm(coordinate_residual)
                / torch.linalg.vector_norm(self.weight @ donor).clamp_min(1e-8)
            ),
            "orthogonal_leakage_relative_residual": float(
                torch.linalg.vector_norm(orthogonal_leakage)
                / torch.linalg.vector_norm(subspace_delta).clamp_min(1e-8)
            ),
        }


def run_causal_patching(
    run_path: Path,
    *,
    patch_mode: str | None = None,
    max_pairs: int | None = None,
    continuations_per_condition: int | None = None,
) -> None:
    """Execute configured H3 cells over a prepared process-isomer manifest."""
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
    """Validate every non-generative H3 dependency and write a preflight report."""
    config = load_config(run_path)
    patch_cfg = config["patching"]
    pairs = load_samples(Path(patch_cfg["pairs"]).resolve())
    pairs = pairs[: int(patch_cfg.get("max_pairs", len(pairs)))]
    component, layer = resolve_patch_target(patch_cfg)
    modes = resolve_patch_modes(patch_cfg, patch_mode)
    errors: list[str] = []
    warnings: list[str] = []

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


def generate_patched_continuation(
    *,
    model: Any,
    tokenizer: Any,
    activation_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    pairs: list[dict[str, Any]],
    pair: dict[str, Any],
    patch_mode: str,
    condition: str,
    continuation: int,
    seed: int,
    component: str,
    layer: int,
    projection: ProjectionSubspace | None,
    patch_cfg: dict[str, Any],
    analysis_cfg: dict[str, Any],
    vector_cache: dict[tuple[str, int, str, int, int], torch.Tensor],
) -> dict[str, Any]:
    """Generate one continuation after applying the requested H3 intervention."""
    target = pair["target"]
    target_key = (str(target["sample_id"]), int(target["seed"]))
    target_row = rows[target_key]
    target_sample = load_source_sample(activation_run, target_key[0])
    generated_ids = [int(token) for token in target_row["generated_token_ids"]]
    target_completion_index = completion_state_index(target, target_row)
    token_end = target_completion_index - 1
    prefix_ids = [*target_sample["input_ids"], *generated_ids[: token_end + 1]]
    if prefix_ids[-1] != generated_ids[token_end]:
        raise AssertionError("Target prefix is not aligned to symbolic token_end")

    patch_vector = None
    donor_description = None
    donor_gold_answer = None
    reconstruction = None
    if condition != "baseline":
        donor_point, donor_state_index = select_control_donor(
            condition=condition,
            pair=pair,
            pairs=pairs,
            rows=rows,
            target_row=target_row,
        )
        donor_key = (str(donor_point["sample_id"]), int(donor_point["seed"]))
        donor_vector = load_activation_vector(
            activation_run=activation_run,
            rows=rows,
            key=donor_key,
            state_index=donor_state_index,
            component=component,
            layer=layer,
            cache=vector_cache,
        )
        donor_sample = load_source_sample(activation_run, donor_key[0])
        donor_gold_answer = extract_answer(
            str(donor_sample.get("gold_answer", "")),
            analysis_cfg.get("gold_answer_regex"),
        )
        if patch_mode == "full":
            patch_vector = donor_vector
        elif patch_mode == "subspace":
            if projection is None:
                raise ValueError("Subspace patching requires a projection")
            target_vector = load_activation_vector(
                activation_run=activation_run,
                rows=rows,
                key=target_key,
                state_index=target_completion_index,
                component=component,
                layer=layer,
                cache=vector_cache,
            )
            patch_vector, reconstruction = projection.swap(
                target=target_vector,
                donor=donor_vector,
            )
        else:
            raise ValueError(f"Unknown patch mode: {patch_mode!r}")
        donor_description = {
            "sample_id": donor_key[0],
            "seed": donor_key[1],
            "state_index": donor_state_index,
            "token_end": donor_point.get("token_end"),
            "operator": donor_point.get("operator"),
            "operation_signature": donor_point.get("operation_signature"),
        }

    set_seed(seed)
    input_device = get_input_device(model)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    patch_context = (
        FirstForwardComponentPatch(
            model=model,
            layer=layer,
            component=component,
            vector=patch_vector,
            expected_sequence_length=len(prefix_ids),
        )
        if patch_vector is not None
        else nullcontext()
    )
    max_new_tokens = int(patch_cfg.get("max_new_tokens", 768))
    answer_pattern = analysis_cfg.get("produced_answer_regex")
    stopping_criteria = (
        StoppingCriteriaList(
            [GeneratedTextRegexStop(tokenizer, len(prefix_ids), answer_pattern)]
        )
        if answer_pattern
        else None
    )
    with patch_context:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=float(patch_cfg.get("temperature", 0.0)) > 0,
            temperature=float(patch_cfg.get("temperature", 0.6)),
            top_p=float(patch_cfg.get("top_p", 0.95)),
            top_k=int(patch_cfg.get("top_k", 20)),
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
    continuation_ids = output[0, len(prefix_ids) :].detach().cpu().tolist()
    text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    produced_answer = extract_answer(text, analysis_cfg.get("produced_answer_regex"))
    target_gold_answer = extract_answer(
        str(target_sample.get("gold_answer", "")),
        analysis_cfg.get("gold_answer_regex"),
    )
    degeneration_reasons = output_degeneration_reasons(continuation_ids, text)
    return {
        "pair_id": pair["pair_id"],
        "target_question": target_key[0],
        "patch_mode": patch_mode,
        "condition": condition,
        "continuation": continuation,
        "seed": seed,
        "component": component,
        "layer": layer,
        "projection_path": projection.path.as_posix() if projection else None,
        "target": target,
        "alignment": {
            "mode": "symbolic_step_end",
            "target_token_end": token_end,
            "target_completion_state_index": target_completion_index,
            "target_prefix_tokens": len(prefix_ids),
        },
        "donor": donor_description,
        "reconstruction": reconstruction,
        "generated_token_ids": continuation_ids,
        "produced_text": text,
        "produced_answer": produced_answer,
        "target_gold_answer": target_gold_answer,
        "donor_gold_answer": donor_gold_answer,
        "matches_target_answer": answers_match(produced_answer, target_gold_answer),
        "matches_donor_answer": (
            answers_match(produced_answer, donor_gold_answer)
            if donor_gold_answer is not None
            else None
        ),
        "matches_neither_answer": (
            produced_answer is not None
            and not answers_match(produced_answer, target_gold_answer)
            and (
                donor_gold_answer is None
                or not answers_match(produced_answer, donor_gold_answer)
            )
        ),
        "is_correct": answers_match(produced_answer, target_gold_answer),
        "has_valid_answer": produced_answer is not None,
        "degenerate_output": bool(degeneration_reasons),
        "degeneration_reasons": degeneration_reasons,
        "ended_with_eos": bool(
            continuation_ids
            and tokenizer.eos_token_id is not None
            and continuation_ids[-1] == tokenizer.eos_token_id
        ),
        "hit_token_limit": len(continuation_ids) >= max_new_tokens,
    }


def load_activation_vector(
    *,
    activation_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    key: tuple[str, int],
    state_index: int,
    component: str,
    layer: int,
    cache: dict[tuple[str, int, str, int, int], torch.Tensor],
) -> torch.Tensor:
    cache_key = (key[0], key[1], component, layer, state_index)
    if cache_key not in cache:
        row = rows[key]
        states, layers = load_component_states_npz(
            activation_run / row["hidden_states_file"],
            component,
        )
        if layer not in layers:
            raise ValueError(
                f"Layer {layer} is unavailable in {row['hidden_states_file']}"
            )
        if not 0 <= state_index < len(states):
            raise ValueError(
                f"State index {state_index} is unavailable in "
                f"{row['hidden_states_file']}"
            )
        cache[cache_key] = torch.from_numpy(
            states[state_index, layers.index(layer)].astype("float32")
        )
    return cache[cache_key]


def select_control_donor(
    *,
    condition: str,
    pair: dict[str, Any],
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
    target_row: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    if condition == "equivalent":
        donor = pair["donor"]
        donor_row = rows[(str(donor["sample_id"]), int(donor["seed"]))]
        return donor, completion_state_index(donor, donor_row)

    alternatives = [
        candidate
        for candidate in pairs
        if candidate["pair_id"] != pair["pair_id"]
        and candidate["graph_signature"] != pair["graph_signature"]
    ]
    if not alternatives:
        raise ValueError("H3 controls require at least two distinct graph states")
    if condition == "mismatched":
        candidate = random.Random(int(pair["pair_id"])).choice(alternatives)
        donor = candidate["donor"]
        donor_row = rows[(str(donor["sample_id"]), int(donor["seed"]))]
        return donor, completion_state_index(donor, donor_row)
    if condition == "position_random":
        target_position = completion_state_index(pair["target"], target_row)
        current_keys = {
            (str(pair[side]["sample_id"]), int(pair[side]["seed"]))
            for side in ("donor", "target")
        }
        eligible_points = [
            candidate[side]
            for candidate in alternatives
            for side in ("donor", "target")
            if (
                str(candidate[side]["sample_id"]),
                int(candidate[side]["seed"]),
            )
            not in current_keys
            and len(
                rows[
                    (
                        str(candidate[side]["sample_id"]),
                        int(candidate[side]["seed"]),
                    )
                ]["generated_token_ids"]
            )
            > target_position
        ]
        if not eligible_points:
            raise ValueError(
                f"No unrelated trace reaches position {target_position} for "
                f"pair {pair['pair_id']}"
            )
        donor = random.Random(int(pair["pair_id"])).choice(eligible_points)
        return donor, target_position
    raise ValueError(f"Unknown patching condition: {condition!r}")


def resolve_patch_modes(
    patch_cfg: dict[str, Any],
    override: str | None,
) -> tuple[str, ...]:
    if override and override != "both":
        modes = (override,)
    elif override == "both":
        modes = PATCH_MODES
    else:
        modes = tuple(str(mode) for mode in patch_cfg.get("patch_modes", PATCH_MODES))
    unknown = set(modes) - set(PATCH_MODES)
    if unknown:
        raise ValueError(f"Unsupported patch modes: {sorted(unknown)}")
    return modes


def resolve_patch_target(patch_cfg: dict[str, Any]) -> tuple[str, int]:
    if patch_cfg.get("alignment", "symbolic_step_end") != "symbolic_step_end":
        raise ValueError("H3 patch alignment must be symbolic_step_end")
    component = patch_cfg.get("component")
    layer = patch_cfg.get("layer")
    if component != "auto" and layer != "auto":
        return validate_patch_target(str(component), int(layer))
    report = json.loads(Path(patch_cfg["component_report"]).read_text())
    target = report.get("recommended_patch_target")
    if not target:
        raise ValueError("Component localization has no recommended patch target")
    if target.get("alignment") != "symbolic_step_end":
        raise ValueError("Component report does not authorize completed-step alignment")
    return validate_patch_target(str(target["component"]), int(target["layer"]))


def validate_patch_target(component: str, layer: int) -> tuple[str, int]:
    if component not in {"mlp_output", "attention_output"}:
        raise ValueError(f"Unsupported H3 patch component: {component!r}")
    return component, layer


def load_projection_subspace(
    path: Path,
    *,
    component: str,
    layer: int,
) -> ProjectionSubspace:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("component") != component:
        raise ValueError(
            f"Projection component {checkpoint.get('component')!r} does not match "
            f"patch component {component!r}"
        )
    if int(checkpoint.get("layer")) != layer:
        raise ValueError(
            f"Projection layer {checkpoint.get('layer')} does not match {layer}"
        )
    weight = checkpoint["weight"].float().cpu()
    if weight.ndim != 2 or weight.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(f"Malformed projection weight: {tuple(weight.shape)}")
    rank = int(torch.linalg.matrix_rank(weight).item())
    if rank != weight.shape[0]:
        raise ValueError(
            f"Projection is row-rank deficient: rank {rank}/{weight.shape[0]}"
        )
    singular_values = torch.linalg.svdvals(weight)
    condition_number = float(
        singular_values.max() / singular_values.clamp_min(1e-12).min()
    )
    return ProjectionSubspace(
        path=path,
        weight=weight,
        pseudoinverse=torch.linalg.pinv(weight),
        rank=rank,
        condition_number=condition_number,
    )


def completion_state_index(
    point: dict[str, Any],
    row: dict[str, Any],
) -> int:
    """Derive the fully resolved state after the symbolic interval's final token."""
    token_count = len(row["generated_token_ids"])
    if token_count < 2:
        raise ValueError("Cannot patch a trace with fewer than two generated tokens")
    token_end = int(point["token_end"])
    if not 0 <= token_end < token_count - 1:
        raise ValueError(
            f"token_end {token_end} has no captured completion state "
            f"in a {token_count}-token trace"
        )
    return token_end + 1


def validate_pair_rows(
    pairs: list[dict[str, Any]],
    rows: dict[tuple[str, int], dict[str, Any]],
) -> None:
    for pair in pairs:
        if not pair.get("path_evidence"):
            raise ValueError(f"Pair {pair['pair_id']} lacks path-diversity evidence")
        if (
            pair["path_evidence"]["donor_history_hash"]
            == pair["path_evidence"]["target_history_hash"]
        ):
            raise ValueError(f"Pair {pair['pair_id']} has identical path hashes")
        for side in ("donor", "target"):
            point = pair[side]
            key = (str(point["sample_id"]), int(point["seed"]))
            if key not in rows:
                raise ValueError(f"Pair {pair['pair_id']} is missing {side} row {key}")
            completion_state_index(point, rows[key])


def load_completed_patches(
    path: Path,
) -> set[tuple[int, str, str, int]]:
    if not path.exists():
        return set()
    rows = load_samples(path.resolve())
    if any("patch_mode" not in row for row in rows):
        raise ValueError(
            f"{path} contains legacy rows without patch_mode; move or remove it "
            "before running the two-variant protocol"
        )
    return {
        (
            int(row["pair_id"]),
            str(row["patch_mode"]),
            str(row["condition"]),
            int(row["continuation"]),
        )
        for row in rows
    }


def output_degeneration_reasons(token_ids: list[int], text: str) -> list[str]:
    """Detect conservative, deterministic output-collapse signatures."""
    reasons = []
    if not text.strip():
        reasons.append("empty_output")
    if len(token_ids) >= 32 and longest_identical_run(token_ids) >= 32:
        reasons.append("repeated_token_run")
    if len(token_ids) >= 100:
        unique_ratio = len(set(token_ids)) / len(token_ids)
        if unique_ratio < 0.02:
            reasons.append("very_low_token_diversity")
        ngrams = [
            tuple(token_ids[index : index + 4]) for index in range(len(token_ids) - 3)
        ]
        if ngrams and len(set(ngrams)) / len(ngrams) < 0.05:
            reasons.append("repeated_four_grams")
    return reasons


def longest_identical_run(values: list[int]) -> int:
    longest = 0
    current = 0
    previous = None
    for value in values:
        current = current + 1 if value == previous else 1
        longest = max(longest, current)
        previous = value
    return longest


class FirstForwardComponentPatch:
    """Replace one component's final prefill-token output exactly once."""

    def __init__(
        self,
        *,
        model: Any,
        layer: int,
        component: str,
        vector: torch.Tensor,
        expected_sequence_length: int,
    ) -> None:
        decoder_layers = get_decoder_layers(model)
        resolved = resolve_layer_indices([layer], len(decoder_layers))[0]
        attribute = "mlp" if component == "mlp_output" else "self_attn"
        self.module = getattr(decoder_layers[resolved], attribute)
        self.vector = vector
        self.expected_sequence_length = expected_sequence_length
        self.handle = None
        self.applied = False

    def __enter__(self) -> FirstForwardComponentPatch:
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def _hook(self, _module, _inputs, output):
        if self.applied:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] != self.expected_sequence_length:
            raise RuntimeError(
                "Patch must occur on the full target prefill ending at token_end; "
                f"expected sequence length {self.expected_sequence_length}, "
                f"received {tuple(hidden.shape)}"
            )
        patched = hidden.clone()
        patched[:, -1, :] = self.vector.to(
            device=patched.device,
            dtype=patched.dtype,
        )
        self.applied = True
        if isinstance(output, tuple):
            return (patched, *output[1:])
        return patched

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.remove()
