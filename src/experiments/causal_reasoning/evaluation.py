"""Residual-state capture and intervention scoring for causal questions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.depth_relief.benchmark import (
    PromptSpec,
    candidate_token_ids,
    format_prompt_spec,
)
from src.experiments.depth_relief.hf import (
    PromptEvaluation,
    checkpoint_token_indices,
    evaluate_prompt,
    patched_logits,
)
from src.experiments.depth_relief.metrics import softmax
from src.models.introspection import get_decoder_layers
from src.runtime.artifact_store import append_jsonl


def _formatted_prompts(
    tokenizer: Any,
    case: dict[str, Any],
    prompt_config: dict[str, Any],
) -> dict[str, PromptSpec]:
    prompts = {}
    for name, raw in case["prompts"].items():
        prompts[str(name)] = format_prompt_spec(
            tokenizer,
            PromptSpec(
                text=str(raw["text"]),
                checkpoint_start=int(raw["checkpoint_start"]),
                checkpoint_end=int(raw["checkpoint_end"]),
            ),
            {"prompt": prompt_config},
        )
    return prompts


def _score_distribution(
    probabilities: np.ndarray,
    *,
    expected: int,
    unconstrained_prediction: int | None,
) -> dict[str, Any]:
    prediction = int(np.argmax(probabilities))
    alternatives = np.delete(probabilities, expected)
    return {
        "prediction": prediction,
        "unconstrained_prediction": unconstrained_prediction,
        "expected": expected,
        "is_expected": prediction == expected,
        "is_expected_unconstrained": unconstrained_prediction == expected,
        "expected_probability": float(probabilities[expected]),
        "strongest_alternative_probability": float(alternatives.max())
        if len(alternatives)
        else 0.0,
        "candidate_probabilities": probabilities.tolist(),
    }


def _baseline_record(
    evaluation: PromptEvaluation,
    *,
    expected: int,
) -> dict[str, Any]:
    return {
        **_score_distribution(
            np.asarray(evaluation.record["final_candidate_probabilities"]),
            expected=expected,
            unconstrained_prediction=evaluation.record[
                "unconstrained_prediction"
            ],
        ),
        "candidate_probability_mass": float(
            evaluation.record["candidate_probability_mass"]
        ),
        "prompt_token_count": int(evaluation.token_count),
    }


def _selected_positions(
    evaluation: PromptEvaluation,
    width: int | str,
) -> tuple[int, ...]:
    positions = evaluation.checkpoint_token_indices
    if width == "all":
        return positions
    width = int(width)
    if not 1 <= width <= len(positions):
        raise ValueError(
            f"Requested {width} checkpoint tokens from {len(positions)}"
        )
    return positions[-width:]


def _patch_layers(
    *,
    mode: str,
    center: int,
    probe_layers: list[int],
    layer_count: int,
) -> list[int]:
    if mode == "single":
        return [center]
    if mode == "window3":
        return [
            layer
            for layer in range(max(0, center - 1), min(layer_count, center + 2))
        ]
    if mode == "all":
        return list(range(layer_count))
    raise ValueError(f"Unknown causal reasoning layer mode: {mode!r}")


def _feature_path(run_path: Path, case_id: str) -> Path:
    return run_path / "evaluation" / "features" / f"{case_id}.npz"


def _save_features(
    *,
    run_path: Path,
    case: dict[str, Any],
    evaluation: PromptEvaluation,
    probe_layers: list[int],
) -> str:
    path = _feature_path(run_path, str(case["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.stack(
        [
            evaluation.checkpoint_states[layer][-1].numpy()
            for layer in probe_layers
        ]
    ).astype(np.float16)
    np.savez_compressed(
        path,
        states=values,
        layers=np.asarray(probe_layers, dtype=np.int16),
    )
    return path.relative_to(run_path).as_posix()


def _representation_rows(
    case: dict[str, Any],
    evaluations: dict[str, PromptEvaluation],
    probe_layers: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for pair in case.get("representation_pairs", []):
        left = evaluations[str(pair["left"])]
        right = evaluations[str(pair["right"])]
        for layer in probe_layers:
            left_state = left.checkpoint_states[layer][-1].numpy()
            right_state = right.checkpoint_states[layer][-1].numpy()
            denominator = float(
                np.linalg.norm(left_state) * np.linalg.norm(right_state)
            )
            rows.append(
                {
                    "pair": str(pair["name"]),
                    "left": str(pair["left"]),
                    "right": str(pair["right"]),
                    "layer": layer,
                    "cosine_similarity": (
                        float(np.dot(left_state, right_state) / denominator)
                        if denominator
                        else 0.0
                    ),
                    "normalized_distance": float(
                        np.linalg.norm(left_state - right_state)
                        / max(np.linalg.norm(left_state), 1e-12)
                    ),
                }
            )
    return rows


def evaluate_case(
    *,
    model: Any,
    tokenizer: Any,
    run_path: Path,
    case: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every baseline and residual swap for one paired case."""
    layer_count = len(get_decoder_layers(model))
    probe_layers = sorted(
        {
            int(layer)
            for layer in config["probe_layers"]
            if 0 <= int(layer) < layer_count
        }
    )
    if not probe_layers:
        raise ValueError("No configured probe layer exists in the loaded model")
    prompts = _formatted_prompts(tokenizer, case, config["prompt"])
    evaluations: dict[str, PromptEvaluation] = {}
    candidate_ids: dict[str, list[int]] = {}
    for name, prompt in prompts.items():
        ids = candidate_token_ids(
            tokenizer, prompt.text, case["candidate_symbols"]
        )
        candidate_ids[name] = ids
        evaluations[name] = evaluate_prompt(
            model=model,
            tokenizer=tokenizer,
            text=prompt.text,
            candidate_ids=ids,
            threshold=float(config.get("settling_jsd_threshold", 0.5)),
            checkpoint_indices=checkpoint_token_indices(tokenizer, prompt),
        )

    result_rows = []
    for specification in case["evaluations"]:
        name = str(specification["name"])
        recipient_name = str(specification["recipient"])
        recipient = evaluations[recipient_name]
        expected = int(specification["expected"])
        source_name = specification.get("source")
        if source_name is None:
            result_rows.append(
                {
                    "condition": name,
                    "source": None,
                    "recipient": recipient_name,
                    "layer_mode": "baseline",
                    "layer": None,
                    "token_width": 0,
                    **_baseline_record(recipient, expected=expected),
                }
            )
            continue
        source = evaluations[str(source_name)]
        width_spec = specification.get("token_width", 1)
        source_positions = _selected_positions(source, width_spec)
        recipient_positions = _selected_positions(recipient, width_spec)
        if len(source_positions) != len(recipient_positions):
            raise ValueError(
                f"{case['id']} patch spans differ: "
                f"{len(source_positions)} source versus "
                f"{len(recipient_positions)} recipient tokens"
            )
        width = len(source_positions)
        modes = [
            str(value)
            for value in specification.get("layer_modes", ["single"])
        ]
        centers: list[int | None] = [
            None
        ] if modes == ["all"] else probe_layers
        for mode in modes:
            mode_centers = [None] if mode == "all" else centers
            for center in mode_centers:
                layers = _patch_layers(
                    mode=mode,
                    center=probe_layers[-1] if center is None else center,
                    probe_layers=probe_layers,
                    layer_count=layer_count,
                )
                patches = {
                    layer: (
                        recipient_positions,
                        source.checkpoint_states[layer][-width:],
                    )
                    for layer in layers
                }
                logits = patched_logits(
                    model=model,
                    tokenizer=tokenizer,
                    text=prompts[recipient_name].text,
                    patches=patches,
                )
                ids = candidate_ids[recipient_name]
                probabilities = softmax(logits[ids])
                full_probabilities = softmax(logits)
                top_id = int(np.argmax(logits))
                unconstrained = ids.index(top_id) if top_id in ids else None
                baseline = np.asarray(
                    recipient.record["final_candidate_probabilities"]
                )
                scored = _score_distribution(
                    probabilities,
                    expected=expected,
                    unconstrained_prediction=unconstrained,
                )
                scored["expected_probability_change"] = float(
                    probabilities[expected] - baseline[expected]
                )
                baseline_prediction = int(np.argmax(baseline))
                scored["recipient_baseline_is_expected"] = (
                    baseline_prediction == expected
                )
                scored["accuracy_change"] = int(scored["is_expected"]) - int(
                    baseline_prediction == expected
                )
                scored["candidate_probability_mass"] = float(
                    full_probabilities[ids].sum()
                )
                result_rows.append(
                    {
                        "condition": name,
                        "source": str(source_name),
                        "recipient": recipient_name,
                        "layer_mode": mode,
                        "layer": center,
                        "patched_layers": layers,
                        "token_width": width,
                        "representation_pair": specification.get(
                            "representation_pair"
                        ),
                        **scored,
                    }
                )

    feature_file = None
    if feature_prompt := case.get("feature_prompt"):
        feature_file = _save_features(
            run_path=run_path,
            case=case,
            evaluation=evaluations[str(feature_prompt)],
            probe_layers=probe_layers,
        )
    return {
        "schema_version": 1,
        "id": case["id"],
        "experiment": case["experiment"],
        "group": case["group"],
        "split": case["split"],
        "labels": case["labels"],
        "probe_layers": probe_layers,
        "feature_file": feature_file,
        "representation": _representation_rows(
            case, evaluations, probe_layers
        ),
        "results": result_rows,
    }


def persist_case(run_path: Path, row: dict[str, Any]) -> None:
    """Append one complete case while keeping the case ID as its resume key."""
    append_jsonl(run_path / "evaluation" / "cases.jsonl", row)
