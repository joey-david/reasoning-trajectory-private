"""Prepare and execute objective-family sentence-boundary interventions."""

from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteriaList

from src.analysis.answers import answers_match, extract_answer
from src.analysis.common import read_generation_rows
from src.analysis.step_classification.segmentation import build_segments
from src.analysis.token_alignment import build_token_spans
from src.experiments.causal_patching import (
    FirstForwardComponentPatch,
    output_degeneration_reasons,
)
from src.experiments.common import balanced_generation_rows
from src.experiments.replay_capture import load_source_sample
from src.experiments.thought_units import load_partitions
from src.models.generation_pipeline import (
    GeneratedTextRegexStop,
    set_seed,
)
from src.models.introspection import get_hidden_size, get_input_device
from src.runtime.artifact_store import append_jsonl
from src.runtime.config import load_config
from src.runtime.data import load_samples, write_jsonl


def prepare_boundary_manifest(run_path: Path) -> Path:
    """Select two position-matched points from each independent boundary family.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The path of the written or discovered artifact.
    """
    config = load_config(run_path)
    intervention_cfg = config["boundary_intervention"]
    source_run = Path(intervention_cfg["source_run"])
    partitions = load_partitions(Path(intervention_cfg["partitions"]))
    methods = {
        str(family): str(method)
        for family, method in intervention_cfg["families"].items()
    }
    target_positions = [
        float(value)
        for value in intervention_cfg.get("target_positions", [1 / 3, 2 / 3])
    ]
    per_sample = int(intervention_cfg.get("source_rollouts_per_question", 1))
    rows = balanced_generation_rows(source_run, per_sample=per_sample)
    token_spans = build_token_spans(source_run, rows)
    manifest: list[dict[str, Any]] = []
    point_id = 0

    for row, spans in zip(rows, token_spans):
        key = (str(row["sample_id"]), int(row["seed"]))
        if key not in partitions:
            raise ValueError(f"Thought-unit partitions do not contain {key}")
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        sentence_count = len(segments)
        for family, method in methods.items():
            if method not in partitions[key]:
                raise ValueError(f"Partition method {method!r} is unavailable")
            boundaries = partitions[key][method]
            selected = position_matched_boundaries(
                boundaries,
                sentence_count=sentence_count,
                target_positions=target_positions,
            )
            for target_position, boundary in zip(target_positions, selected):
                token_end = int(segments[int(boundary)].token_end)
                if token_end >= len(row["generated_token_ids"]) - 1:
                    raise ValueError(
                        f"Boundary {boundary} in {key} has no continuation token"
                    )
                manifest.append(
                    {
                        "point_id": point_id,
                        "sample_id": key[0],
                        "seed": key[1],
                        "family": family,
                        "partition_method": method,
                        "boundary_sentence": int(boundary),
                        "sentence_count": sentence_count,
                        "token_end": token_end,
                        "token_count": len(row["generated_token_ids"]),
                        "normalized_position": float(
                            boundary / max(sentence_count - 2, 1)
                        ),
                        "target_position": target_position,
                    }
                )
                point_id += 1

    path = Path(intervention_cfg["manifest"])
    write_jsonl(path, manifest)
    return path


def position_matched_boundaries(
    boundaries: np.ndarray,
    *,
    sentence_count: int,
    target_positions: list[float],
) -> np.ndarray:
    """Choose distinct available boundaries nearest requested trace positions.

    Args:
        boundaries: Sentence or token boundary indices.
        sentence_count: Number of sentences in the trace.
        target_positions: Normalized target positions to match with controls.

    Returns:
        The resulting numeric array or tensor.
    """
    available = [int(value) for value in np.asarray(boundaries, dtype=int)]
    if len(available) < len(target_positions):
        raise ValueError(
            f"Need {len(target_positions)} boundaries, found {len(available)}"
        )
    selected: list[int] = []
    for target in target_positions:
        boundary = min(
            (value for value in available if value not in selected),
            key=lambda value: abs(
                value / max(sentence_count - 2, 1) - float(target)
            ),
        )
        selected.append(boundary)
    return np.asarray(selected, dtype=np.int32)


def completed_interventions(path: Path) -> set[tuple[int, str, int]]:
    """Load persisted point, condition, and continuation keys.

    Args:
        path: Filesystem path to read from or write to.

    Returns:
        The resulting unique values.
    """
    if not path.exists():
        return set()
    return {
        (
            int(row["point_id"]),
            str(row["condition"]),
            int(row["continuation"]),
        )
        for row in load_samples(path.resolve())
    }


def generate_boundary_continuation(
    *,
    run_path: Path,
    model: Any,
    tokenizer: Any,
    source_run: Path,
    rows: dict[tuple[str, int], dict[str, Any]],
    point: dict[str, Any],
    condition: str,
    continuation: int,
    seed: int,
    component: str,
    layer: int,
    intervention_cfg: dict[str, Any],
    analysis_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Continue one stored prefix with a baseline or zeroed component output.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        model: Loaded model used for inference or transformation.
        tokenizer: Tokenizer aligned with the loaded model.
        source_run: Source run directory containing the original artifacts.
        rows: Generation or analysis records to process.
        point: Boundary intervention specification.
        condition: Intervention or prompt condition name.
        continuation: Continuation replicate index.
        seed: Random seed for reproducible sampling or generation.
        component: Activation component name.
        layer: Model layer index.
        intervention_cfg: Boundary intervention generation configuration.
        analysis_cfg: Answer extraction and scoring configuration.

    Returns:
        The resulting keyed records or metrics.
    """
    key = (str(point["sample_id"]), int(point["seed"]))
    row = rows[key]
    sample = load_source_sample(source_run, key[0])
    generated_ids = [int(token) for token in row["generated_token_ids"]]
    token_end = int(point["token_end"])
    prefix_ids = [
        *[int(token) for token in sample["input_ids"]],
        *generated_ids[: token_end + 1],
    ]

    patch_context = nullcontext()
    if condition == "zero":
        patch_context = FirstForwardComponentPatch(
            model=model,
            layer=layer,
            component=component,
            vector=torch.zeros(get_hidden_size(model), dtype=torch.float32),
            expected_sequence_length=len(prefix_ids),
        )
    elif condition != "baseline":
        raise ValueError(f"Unsupported boundary condition: {condition!r}")

    set_seed(seed)
    input_device = get_input_device(model)
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    max_new_tokens = int(intervention_cfg.get("max_new_tokens", 10000))
    answer_pattern = analysis_cfg.get("produced_answer_regex")
    stopping_criteria = (
        StoppingCriteriaList(
            [GeneratedTextRegexStop(tokenizer, len(prefix_ids), answer_pattern)]
        )
        if answer_pattern
        else None
    )
    temperature = float(intervention_cfg.get("temperature", 0.0))
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": stopping_criteria,
    }
    if temperature > 0:
        generation_kwargs.update(
            {
                "temperature": temperature,
                "top_p": float(intervention_cfg.get("top_p", 0.95)),
                "top_k": int(intervention_cfg.get("top_k", 20)),
            }
        )
    with patch_context:
        output = model.generate(**generation_kwargs)
    continuation_ids = output[0, len(prefix_ids) :].detach().cpu().tolist()
    text = tokenizer.decode(
        continuation_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    produced_answer = extract_answer(text, analysis_cfg.get("produced_answer_regex"))
    gold_answer = extract_answer(
        str(sample.get("gold_answer", "")),
        analysis_cfg.get("gold_answer_regex"),
    )
    degeneration_reasons = output_degeneration_reasons(continuation_ids, text)
    record = {
        **point,
        "condition": condition,
        "continuation": continuation,
        "intervention_seed": seed,
        "component": component,
        "layer": layer,
        "prefix_tokens": len(prefix_ids),
        "generated_token_ids": continuation_ids,
        "produced_text": text,
        "produced_answer": produced_answer,
        "gold_answer": gold_answer,
        "is_correct": answers_match(produced_answer, gold_answer),
        "has_valid_answer": produced_answer is not None,
        "source_is_correct": bool(row.get("is_correct")),
        "source_produced_answer": row.get("produced_answer"),
        "degenerate_output": bool(degeneration_reasons),
        "degeneration_reasons": degeneration_reasons,
        "hit_token_limit": len(continuation_ids) >= max_new_tokens,
    }
    append_jsonl(
        run_path / "interventions" / "continuations.jsonl",
        record,
    )
    return record


def load_intervention_rows(
    source_run: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Index source generations by sample and seed.

    Args:
        source_run: Source run directory containing the original artifacts.

    Returns:
        The resulting keyed records or metrics.
    """
    return {
        (str(row["sample_id"]), int(row["seed"])): row
        for row in read_generation_rows(source_run)
    }


def analyze_boundary_interventions(run_path: Path) -> Path:
    """Summarize complete baseline/zero pairs and make partial coverage explicit.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The path of the written or discovered artifact.
    """
    from src.experiments.objective_causality import objective_specificity

    config = load_config(run_path)
    intervention_cfg = config["boundary_intervention"]
    manifest = load_samples(Path(intervention_cfg["manifest"]).resolve())
    result_path = run_path / "interventions" / "continuations.jsonl"
    rows = load_samples(result_path.resolve()) if result_path.exists() else []
    by_point: defaultdict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_point[int(row["point_id"])][str(row["condition"])] = row
    pairs = [
        (conditions["baseline"], conditions["zero"])
        for conditions in by_point.values()
        if {"baseline", "zero"} <= conditions.keys()
    ]

    families = sorted({str(point["family"]) for point in manifest})
    current_partitions = load_partitions(Path(intervention_cfg["partitions"]))
    partition_match = {}
    # Manifests are immutable run inputs. Compare them with current partitions
    # explicitly because post-hoc rescoring can otherwise silently relabel the
    # intervention being analyzed.
    for family in families:
        method = str(intervention_cfg["families"][family])
        points = [point for point in manifest if str(point["family"]) == family]
        matches = sum(
            int(point["boundary_sentence"])
            in current_partitions[
                (str(point["sample_id"]), int(point["seed"]))
            ][method]
            for point in points
        )
        partition_match[family] = {
            "matched_points": matches,
            "manifest_points": len(points),
            "fraction": matches / max(len(points), 1),
        }
    family_reports = {
        family: paired_intervention_summary(
            [pair for pair in pairs if str(pair[0]["family"]) == family]
        )
        for family in families
    }
    random_pairs = {
        (str(pair[0]["sample_id"]), float(pair[0]["target_position"])): pair
        for pair in pairs
        if str(pair[0]["family"]) == "random"
    }
    answer_pairs = {
        (str(pair[0]["sample_id"]), float(pair[0]["target_position"])): pair
        for pair in pairs
        if str(pair[0]["family"]) == "answer"
    }
    answer_partition_strata = {}
    for label, expected in (
        ("matches_current_gold_partition", True),
        ("does_not_match_current_gold_partition", False),
    ):
        subset = {
            key: pair
            for key, pair in answer_pairs.items()
            if (
                int(pair[0]["boundary_sentence"])
                in current_partitions[
                    (str(pair[0]["sample_id"]), int(pair[0]["seed"]))
                ]["oracle_answer"]
            )
            == expected
        }
        answer_partition_strata[label] = {
            "descriptive_effect": paired_intervention_summary(
                list(subset.values()),
                include_positions=False,
            ),
            "specificity_vs_random": compare_pair_families(
                subset,
                random_pairs,
            ),
        }
    for family in families:
        if family == "random":
            continue
        family_pairs = {
            (str(pair[0]["sample_id"]), float(pair[0]["target_position"])): pair
            for pair in pairs
            if str(pair[0]["family"]) == family
        }
        family_reports[family]["specificity_vs_random"] = compare_pair_families(
            family_pairs,
            random_pairs,
        )
        if family in {"answer", "object", "correctness", "compression"}:
            family_reports[family]["objective_specificity"] = objective_specificity(
                family_pairs,
                random_pairs,
                family,
            )
    completed_questions = {str(pair[0]["sample_id"]) for pair in pairs}
    report = {
        "experiment": "sentence_boundary_attention_output_ablation",
        "status": "complete" if len(rows) == len(manifest) * 2 else "partial",
        "intervention": {
            "component": str(intervention_cfg["component"]),
            "layer": int(intervention_cfg["layer"]),
            "condition": (
                "zero the selected component at the final token of a "
                "sentence-boundary prefix, then generate deterministically"
            ),
        },
        "coverage": {
            "manifest_points": len(manifest),
            "expected_rows": len(manifest)
            * len(intervention_cfg["conditions"])
            * int(intervention_cfg.get("continuations_per_condition", 1)),
            "observed_rows": len(rows),
            "complete_pairs": len(pairs),
            "complete_pair_fraction": len(pairs) / max(len(manifest), 1),
            "manifest_questions": len(
                {str(point["sample_id"]) for point in manifest}
            ),
            "completed_questions": len(completed_questions),
            "unpaired_points": sum(
                not {"baseline", "zero"} <= conditions.keys()
                for conditions in by_point.values()
            ),
        },
        "manifest_provenance": {
            "current_partition_match": partition_match,
            "answer_partition_strata": answer_partition_strata,
            "warning": (
                "The answer manifest predates gold-solution rescoring and tests "
                "the earlier cross-rollout answer proxy wherever the current "
                "oracle no longer matches."
                if partition_match.get("answer", {}).get("fraction") != 1.0
                else None
            ),
        },
        "overall": paired_intervention_summary(pairs),
        "families": family_reports,
        "interpretation": (
            "Descriptive only until all manifest points finish."
            if len(rows) != len(manifest) * 2
            else "Complete matched-boundary intervention result."
        ),
    }
    output_path = run_path / "analysis" / "report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def paired_intervention_summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    include_positions: bool = True,
    include_complete_sensitivity: bool = True,
) -> dict[str, Any]:
    """Compute question-balanced paired effects for one boundary family.

    Args:
        pairs: Matched treatment/control or process-isomer pairs.
        include_positions: Whether to include position-stratified effects.
        include_complete_sensitivity: Whether to report complete-continuation sensitivity.

    Returns:
        The resulting keyed records or metrics.
    """
    metrics: dict[str, list[float]] = {
        "accuracy_delta": [],
        "valid_answer_delta": [],
        "token_limit_delta": [],
        "degeneration_delta": [],
        "answer_changed": [],
        "continuation_unchanged": [],
        "common_prefix_fraction": [],
        "length_ratio": [],
    }
    groups: list[str] = []
    improved = worsened = 0
    for baseline, zero in pairs:
        baseline_correct = bool(baseline["is_correct"])
        zero_correct = bool(zero["is_correct"])
        improved += int(zero_correct and not baseline_correct)
        worsened += int(baseline_correct and not zero_correct)
        baseline_ids = [int(token) for token in baseline["generated_token_ids"]]
        zero_ids = [int(token) for token in zero["generated_token_ids"]]
        common = 0
        for left, right in zip(baseline_ids, zero_ids):
            if left != right:
                break
            common += 1
        metrics["accuracy_delta"].append(float(zero_correct - baseline_correct))
        metrics["valid_answer_delta"].append(
            float(
                bool(zero["has_valid_answer"])
                - bool(baseline["has_valid_answer"])
            )
        )
        metrics["token_limit_delta"].append(
            float(bool(zero["hit_token_limit"]) - bool(baseline["hit_token_limit"]))
        )
        metrics["degeneration_delta"].append(
            float(
                bool(zero["degenerate_output"])
                - bool(baseline["degenerate_output"])
            )
        )
        metrics["answer_changed"].append(
            float(zero.get("produced_answer") != baseline.get("produced_answer"))
        )
        metrics["continuation_unchanged"].append(float(zero_ids == baseline_ids))
        metrics["common_prefix_fraction"].append(
            common / max(min(len(baseline_ids), len(zero_ids)), 1)
        )
        metrics["length_ratio"].append(
            len(zero_ids) / max(len(baseline_ids), 1)
        )
        groups.append(str(baseline["sample_id"]))

    summaries = {
        metric: grouped_mean_interval(values, groups)
        for metric, values in metrics.items()
    }
    report = {
        "pairs": len(pairs),
        "questions": len(set(groups)),
        "source_accuracy": question_balanced_mean(
            [float(bool(baseline["source_is_correct"])) for baseline, _zero in pairs],
            groups,
        ),
        "baseline_accuracy": question_balanced_mean(
            [float(bool(baseline["is_correct"])) for baseline, _zero in pairs],
            groups,
        ),
        "zero_accuracy": question_balanced_mean(
            [float(bool(zero["is_correct"])) for _baseline, zero in pairs],
            groups,
        ),
        "improved_pairs": improved,
        "worsened_pairs": worsened,
        "metrics": summaries,
    }
    if include_positions:
        positions = sorted({float(pair[0]["target_position"]) for pair in pairs})
        report["target_positions"] = {
            str(position): paired_intervention_summary(
                [
                    pair
                    for pair in pairs
                    if float(pair[0]["target_position"]) == position
                ],
                include_positions=False,
                include_complete_sensitivity=False,
            )
            for position in positions
        }
    if include_complete_sensitivity:
        # Long truncated continuations make fallback answer extraction
        # unreliable, so retain the full estimate but report this strict slice.
        completed = [
            pair
            for pair in pairs
            if not pair[0]["hit_token_limit"]
            and not pair[1]["hit_token_limit"]
        ]
        report["complete_continuations_only"] = paired_intervention_summary(
            completed,
            include_positions=False,
            include_complete_sensitivity=False,
        )
    return report


def compare_pair_families(
    target_pairs: dict[
        tuple[str, float], tuple[dict[str, Any], dict[str, Any]]
    ],
    control_pairs: dict[
        tuple[str, float], tuple[dict[str, Any], dict[str, Any]]
    ],
) -> dict[str, Any]:
    """Compare target-boundary effects with position-matched random effects.

    Args:
        target_pairs: Target-family matched pairs.
        control_pairs: Control-family matched pairs.

    Returns:
        The resulting keyed records or metrics.
    """
    keys = sorted(target_pairs.keys() & control_pairs.keys())
    groups = [sample_id for sample_id, _position in keys]

    def effect(
        pair: tuple[dict[str, Any], dict[str, Any]],
        field: str,
    ) -> float:
        """Measure the binary outcome change from baseline to zeroing.

        Args:
            pair: Matched pair to evaluate or intervene on.
            field: Record field to read or summarize.

        Returns:
            The computed scalar metric.
        """
        baseline, zero = pair
        return float(bool(zero[field]) - bool(baseline[field]))

    accuracy = [
        effect(target_pairs[key], "is_correct")
        - effect(control_pairs[key], "is_correct")
        for key in keys
    ]
    token_limit = [
        effect(target_pairs[key], "hit_token_limit")
        - effect(control_pairs[key], "hit_token_limit")
        for key in keys
    ]
    answer_changed = [
        float(
            target_pairs[key][0].get("produced_answer")
            != target_pairs[key][1].get("produced_answer")
        )
        - float(
            control_pairs[key][0].get("produced_answer")
            != control_pairs[key][1].get("produced_answer")
        )
        for key in keys
    ]
    common_prefix = [
        pair_common_prefix_fraction(target_pairs[key])
        - pair_common_prefix_fraction(control_pairs[key])
        for key in keys
    ]
    report = {
        "matched_points": len(keys),
        "questions": len(set(groups)),
        "accuracy_delta_difference": grouped_mean_interval(accuracy, groups),
        "token_limit_delta_difference": grouped_mean_interval(token_limit, groups),
        "answer_changed_difference": grouped_mean_interval(answer_changed, groups),
        "common_prefix_fraction_difference": grouped_mean_interval(
            common_prefix,
            groups,
        ),
    }
    completed_keys = [
        key
        for key in keys
        if not any(
            row["hit_token_limit"]
            for row in (*target_pairs[key], *control_pairs[key])
        )
    ]
    completed_groups = [sample_id for sample_id, _position in completed_keys]
    completed_accuracy = [
        effect(target_pairs[key], "is_correct")
        - effect(control_pairs[key], "is_correct")
        for key in completed_keys
    ]
    report["complete_continuations_only"] = {
        "matched_points": len(completed_keys),
        "questions": len(set(completed_groups)),
        "accuracy_delta_difference": grouped_mean_interval(
            completed_accuracy,
            completed_groups,
        ),
    }
    return report


def pair_common_prefix_fraction(
    pair: tuple[dict[str, Any], dict[str, Any]],
) -> float:
    """Measure the unchanged prefix shared by one baseline/ablation pair.

    Args:
        pair: Matched pair to evaluate or intervene on.

    Returns:
        The computed scalar metric.
    """
    baseline_ids = [int(token) for token in pair[0]["generated_token_ids"]]
    zero_ids = [int(token) for token in pair[1]["generated_token_ids"]]
    common = 0
    for left, right in zip(baseline_ids, zero_ids):
        if left != right:
            break
        common += 1
    return common / max(min(len(baseline_ids), len(zero_ids)), 1)


def question_balanced_mean(values: list[float], groups: list[str]) -> float:
    """Average within questions before averaging across questions.

    Args:
        values: Values to summarize or transform.
        groups: Group labels used to prevent cross-question leakage.

    Returns:
        The computed scalar metric.
    """
    if not values:
        return float("nan")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    return float(np.mean([np.mean(group) for group in grouped.values()]))


def grouped_mean_interval(
    values: list[float],
    groups: list[str],
    *,
    repetitions: int = 2000,
) -> dict[str, Any]:
    """Return a deterministic question-bootstrap interval for a paired metric.

    Args:
        values: Values to summarize or transform.
        groups: Group labels used to prevent cross-question leakage.
        repetitions: Number of grouped resampling repetitions.

    Returns:
        The resulting keyed records or metrics.
    """
    if not values:
        return {"mean": float("nan"), "question_bootstrap_95ci": [float("nan")] * 2}
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        grouped[group].append(float(value))
    questions = sorted(grouped)
    question_means = np.asarray(
        [np.mean(grouped[question]) for question in questions],
        dtype=np.float64,
    )
    rng = np.random.default_rng(42)
    draws = rng.choice(
        question_means,
        size=(repetitions, len(question_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(question_means.mean()),
        "question_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
    }
