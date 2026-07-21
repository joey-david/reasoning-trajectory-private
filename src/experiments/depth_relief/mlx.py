"""Compute-light MLX screen using the same benchmark and depth metric as HF runs."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.llama import create_attention_mask

from .benchmark import (
    candidate_token_ids,
    condition_specs,
    decimal_state_symbols,
    format_model_prompt,
    format_prompt_spec,
    render_prompt,
    render_write_prompt,
)
from .metrics import jensen_shannon, settling_depth, softmax


def _batch_layer_distributions(
    *, model: Any, tokenizer: Any, prompts: list[str], candidate_ids: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return candidate distributions and full-vocabulary JSD by prompt/layer."""
    token_rows = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    lengths = {len(row) for row in token_rows}
    if len(lengths) != 1:
        raise ValueError(f"MLX prompt batch is not token-aligned: {sorted(lengths)}")
    h = model.model.embed_tokens(mx.array(token_rows))
    full_mask = create_attention_mask(h, None)
    sliding_mask = (
        create_attention_mask(h, None, window_size=model.model.sliding_window)
        if model.model.swa_idx is not None
        else None
    )
    states = []
    for layer in model.model.layers:
        mask = sliding_mask if layer.use_sliding else full_mask
        h = layer(h, mask, cache=None)
        state = h[:, -1, :]
        mx.eval(h, state)
        states.append(state)
    final_normalized = model.model.norm(h[:, -1, :])
    final_logits = (
        model.model.embed_tokens.as_linear(final_normalized)
        if model.args.tie_word_embeddings
        else model.lm_head(final_normalized)
    ).astype(mx.float32)
    mx.eval(final_logits)
    final_full = np.stack([softmax(row) for row in np.asarray(final_logits)])
    full_top_ids = np.asarray(final_logits).argmax(axis=1)
    candidate_mass = final_full[:, candidate_ids].sum(axis=1)
    candidate_rows = []
    divergence_rows = []
    for state in states:
        normalized = model.model.norm(state)
        logits = (
            model.model.embed_tokens.as_linear(normalized)
            if model.args.tie_word_embeddings
            else model.lm_head(normalized)
        )
        logits = logits.astype(mx.float32)
        mx.eval(logits)
        values = np.asarray(logits, dtype=np.float32)
        candidate_rows.append(np.stack([softmax(row[candidate_ids]) for row in values]))
        divergence_rows.append(
            np.asarray(
                [jensen_shannon(softmax(row), final) for row, final in zip(values, final_full)]
            )
        )
    return (
        np.stack(candidate_rows, axis=1),
        np.stack(divergence_rows, axis=1),
        full_top_ids,
        candidate_mass,
    )


def _records(
    distributions: np.ndarray,
    divergences: np.ndarray,
    full_top_ids: np.ndarray,
    candidate_mass: np.ndarray,
    candidate_ids: list[int],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    result = []
    for rows, jsd, top_id, mass in zip(
        distributions, divergences, full_top_ids, candidate_mass
    ):
        prediction = int(np.argmax(rows[-1]))
        unconstrained = candidate_ids.index(int(top_id)) if int(top_id) in candidate_ids else None
        threshold_grid = sorted({threshold, 0.25, 0.5, 0.75})
        result.append(
            {
                "prediction": prediction,
                "unconstrained_prediction": unconstrained,
                "unconstrained_token_id": int(top_id),
                "candidate_probability_mass": float(mass),
                "settling_depth": settling_depth(jsd, threshold=threshold),
                "settling_depth_by_threshold": {
                    str(value): settling_depth(jsd, threshold=value)
                    for value in threshold_grid
                },
                "dtr_jsd": jsd.tolist(),
                "dtr_jsd_auc": float(np.mean(jsd)),
                "candidate_probabilities": rows.tolist(),
                "final_candidate_probabilities": rows[-1].tolist(),
            }
        )
    return result


def evaluate_case_mlx(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate self-writing and every matched checkpoint dose locally."""
    candidate_count = 2 ** int(case["bits"])
    candidates = decimal_state_symbols(candidate_count)
    threshold = float(config.get("settling_jsd_threshold", 0.5))
    write_prompt = format_model_prompt(tokenizer, render_write_prompt(case), config)
    write_ids = candidate_token_ids(tokenizer, write_prompt, candidates)
    writer_distribution, writer_jsd, writer_top_ids, writer_mass = _batch_layer_distributions(
        model=model,
        tokenizer=tokenizer,
        prompts=[write_prompt],
        candidate_ids=write_ids,
    )
    writer = _records(
        writer_distribution,
        writer_jsd,
        writer_top_ids,
        writer_mass,
        write_ids,
        threshold=threshold,
    )[0]
    self_state = int(writer["prediction"])

    specs = condition_specs(case, self_state=self_state)
    prompts = [
        format_prompt_spec(tokenizer, render_prompt(case, spec), config).text
        for spec in specs
    ]
    candidate_ids = candidate_token_ids(tokenizer, prompts[0], candidates)
    distributions, divergences, full_top_ids, candidate_mass = _batch_layer_distributions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        candidate_ids=candidate_ids,
    )
    records = _records(
        distributions,
        divergences,
        full_top_ids,
        candidate_mass,
        candidate_ids,
        threshold=threshold,
    )
    conditions = {}
    token_counts = {len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts}
    if len(token_counts) != 1:
        raise ValueError("MLX conditions violate exact token alignment")
    for spec, record in zip(specs, records):
        record.update(
            {
                "token_count": next(iter(token_counts)),
                "revealed_bits": spec["revealed_bits"],
                "register_state": spec["state"],
                "expected_next_state": int(spec["expected_next_state"]),
                "is_expected": record["prediction"] == int(spec["expected_next_state"]),
                "is_expected_unconstrained": record["unconstrained_prediction"]
                == int(spec["expected_next_state"]),
                "is_correct": record["prediction"] == int(case["next_state"]),
                "is_correct_unconstrained": record["unconstrained_prediction"]
                == int(case["next_state"]),
            }
        )
        conditions[str(spec["name"])] = record
    writer_top = [int(np.argmax(row)) for row in writer["candidate_probabilities"]]
    return {
        "schema_version": 1,
        "backend": "mlx",
        "id": case["id"],
        "family": case["family"],
        "bits": case["bits"],
        "next_state": case["next_state"],
        "counterfactual_next_state": case["counterfactual_next_state"],
        "writer": {
            **writer,
            "true_state": case["current_state"],
            "self_state": self_state,
            "is_correct": self_state == int(case["current_state"]),
            "is_correct_unconstrained": writer["unconstrained_prediction"]
            == int(case["current_state"]),
            "correct_top1_at_any_layer": int(case["current_state"]) in writer_top,
        },
        "conditions": conditions,
        "causal": None,
    }
