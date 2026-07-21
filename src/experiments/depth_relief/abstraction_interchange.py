"""Implicit-to-implicit causal interchange across matched histories."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch

from .benchmark import apply_rule, candidate_token_ids, state_symbols
from .factorization import render_factorization_prompts
from .handoff import trace_position
from .hf import patched_logits
from .metrics import cluster_bootstrap_mean_ci, jensen_shannon
from .qualification import score_logits


PATCH_MODES = (
    "state_different",
    "full_different",
    "random_different",
    "state_same",
    "full_same",
)


def _correct(row: dict[str, Any], condition: str) -> bool:
    return bool(row["conditions"][condition]["is_expected_unconstrained"])


def behavior_qualified(row: dict[str, Any]) -> bool:
    """Require every local transition and synthesized state to be correct."""
    return (
        _correct(row, "read")
        and _correct(row, "update")
        and _correct(row, "synthesize")
        and all(
            _correct(row, f"history_step_{index}")
            for index in range(1, int(row["history_steps"]) + 1)
        )
    )


def build_interchange_pairs(
    cases: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Pair failed recipients with same-state and minimal different-state donors."""
    indexed_cases = {str(case["id"]): case for case in cases}
    indexed_rows = {str(row["id"]): row for row in rows}
    if set(indexed_cases) != set(indexed_rows):
        raise ValueError("Behavior rows do not exactly cover abstraction cases")
    qualified = {
        case_id for case_id, row in indexed_rows.items() if behavior_qualified(row)
    }
    by_cell = {
        (
            str(case["abstraction_group"]),
            str(case["format"]),
            int(case["current_state"]),
            int(case["path_code"]),
        ): str(case["id"])
        for case in cases
    }
    state_count = 2 ** int(cases[0]["bits"])
    pairs = []
    for recipient_id in sorted(qualified):
        recipient = indexed_cases[recipient_id]
        if _correct(indexed_rows[recipient_id], "compose"):
            continue
        group = str(recipient["abstraction_group"])
        representation = str(recipient["format"])
        state = int(recipient["current_state"])
        path_code = int(recipient["path_code"])
        same_candidates = [
            by_cell[(group, representation, state, candidate_path)]
            for offset in range(1, state_count)
            for candidate_path in [(path_code + offset) % state_count]
            if by_cell[(group, representation, state, candidate_path)] in qualified
        ]
        different_candidates = [
            by_cell[(group, representation, candidate_state, path_code)]
            for offset in range(1, state_count)
            for candidate_state in [(state + offset) % state_count]
            if by_cell[(group, representation, candidate_state, path_code)]
            in qualified
        ]
        if not same_candidates or not different_candidates:
            continue
        same_id = same_candidates[0]
        different_id = different_candidates[0]
        same = indexed_cases[same_id]
        different = indexed_cases[different_id]
        if recipient["history"][:-1] != different["history"][:-1]:
            raise ValueError("Different-state donor is not a minimal history contrast")
        if recipient["final_rule"] != different["final_rule"]:
            raise ValueError("Different-state donor changed the final operation")
        if int(same["current_state"]) != state:
            raise ValueError("Same-state donor changed the current state")
        pairs.append(
            {
                "schema_version": 1,
                "id": recipient_id,
                "split": str(recipient["abstraction_split"]),
                "group": group,
                "recipient_id": recipient_id,
                "same_state_source_id": same_id,
                "different_state_source_id": different_id,
                "recipient_state": state,
                "different_state": int(different["current_state"]),
                "recipient_target": int(recipient["next_state"]),
                "different_target": int(different["next_state"]),
            }
        )
    return pairs


def select_interchange_pairs(
    pairs: list[dict[str, Any]], *, max_per_group: int, seed: int
) -> list[dict[str, Any]]:
    """Choose a deterministic, state-balanced causal subset within each group."""
    if max_per_group < 1:
        raise ValueError("max_per_group must be positive")
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair["group"]), {}).setdefault(
            int(pair["recipient_state"]), []
        ).append(pair)
    selected = []
    for group, by_state in sorted(grouped.items()):
        for candidates in by_state.values():
            candidates.sort(
                key=lambda pair: hashlib.sha256(
                    f"{seed}:{pair['id']}".encode()
                ).hexdigest()
            )
        group_selected: list[dict[str, Any]] = []
        depth = 0
        while len(group_selected) < max_per_group:
            added = False
            for state in sorted(by_state):
                if depth >= len(by_state[state]):
                    continue
                group_selected.append(by_state[state][depth])
                added = True
                if len(group_selected) >= max_per_group:
                    break
            if not added:
                break
            depth += 1
        selected.extend(group_selected)
    return selected


def fit_implicit_state_subspaces(
    *,
    cases: dict[str, dict[str, Any]],
    captures: dict[str, dict[str, Any]],
    activations: dict[str, dict[str, np.ndarray]],
    rank: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Fit state directions on the complete balanced training grid."""
    train_ids = [
        case_id
        for case_id, case in cases.items()
        if case["abstraction_split"] == "train"
    ]
    state_count = 2 ** int(next(iter(cases.values()))["bits"])
    grouped: dict[int, list[np.ndarray]] = {state: [] for state in range(state_count)}
    for case_id in train_ids:
        case = cases[case_id]
        grouped[int(case["current_state"])].append(
            trace_position(
                activations[case_id],
                captures[case_id],
                f"history_step_{int(case['history_steps'])}",
            )
        )
    missing = [state for state, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"Qualified training cases miss states: {missing}")
    centroids = np.stack(
        [np.mean(grouped[state], axis=0) for state in range(state_count)]
    ).astype(np.float32)
    centered = centroids - centroids.mean(axis=0, keepdims=True)
    layer_count, hidden_size = centered.shape[1:]
    fitted_rank = min(rank, state_count - 1, hidden_size)
    rng = np.random.default_rng(seed)
    state_bases, random_bases = [], []
    for layer in range(layer_count):
        _, _, vt = np.linalg.svd(centered[:, layer], full_matrices=False)
        state_bases.append(vt[:fitted_rank].T.astype(np.float32))
        noise = rng.normal(size=(hidden_size, fitted_rank))
        noise -= state_bases[-1] @ (state_bases[-1].T @ noise)
        random_basis, _ = np.linalg.qr(noise)
        random_bases.append(random_basis.astype(np.float32))
    return {
        "state_basis": np.stack(state_bases),
        "random_basis": np.stack(random_bases),
        "rank": np.asarray(fitted_rank),
        "training_count": np.asarray(len(train_ids)),
    }


def score_interchange_patches_hf(
    *,
    model: Any,
    tokenizer: Any,
    recipient: dict[str, Any],
    different_source: dict[str, Any],
    config: dict[str, Any],
    layer: int,
    token_index: int,
    recipient_state: np.ndarray,
    different_state: np.ndarray,
    same_state: np.ndarray,
    state_basis: np.ndarray,
    random_basis: np.ndarray,
) -> dict[str, Any]:
    """Patch one history endpoint with matched implicit source states."""
    prompt = next(
        prompt
        for prompt in render_factorization_prompts(
            tokenizer=tokenizer, case=recipient, config=config
        )
        if prompt["name"] == "compose"
    )
    candidate_ids = candidate_token_ids(
        tokenizer, prompt["text"], state_symbols(recipient)
    )
    different_delta = different_state - recipient_state
    same_delta = same_state - recipient_state
    state_different = state_basis @ (state_basis.T @ different_delta)
    random_different = random_basis @ (random_basis.T @ different_delta)
    state_norm = float(np.linalg.norm(state_different))
    random_norm = float(np.linalg.norm(random_different))
    if random_norm > 1e-8:
        random_different *= state_norm / random_norm
    vectors = {
        "state_different": state_different,
        "full_different": different_delta,
        "random_different": random_different,
        "state_same": state_basis @ (state_basis.T @ same_delta),
        "full_same": same_delta,
    }
    modulus = 2 ** int(recipient["bits"])
    different_target = apply_rule(
        recipient["final_rule"], int(different_source["current_state"]), modulus
    )
    targets = {
        mode: (
            different_target if "different" in mode else int(recipient["next_state"])
        )
        for mode in PATCH_MODES
    }
    conditions = {}
    for mode in PATCH_MODES:
        replacement = torch.from_numpy(
            (recipient_state + vectors[mode]).astype(np.float32)
        )[None, :]
        record = score_logits(
            patched_logits(
                model=model,
                tokenizer=tokenizer,
                text=prompt["text"],
                patches={layer: ((token_index,), replacement)},
            ),
            candidate_ids,
        )
        target = targets[mode]
        record.update(
            expected_next_state=target,
            is_expected_unconstrained=record["unconstrained_prediction"] == target,
        )
        conditions[mode] = record
    return {
        "schema_version": 1,
        "id": str(recipient["id"]),
        "layer": layer,
        "recipient_state": int(recipient["current_state"]),
        "different_state": int(different_source["current_state"]),
        "conditions": conditions,
    }


def summarize_interchange(
    *,
    captures: dict[str, dict[str, Any]],
    pairs: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Select a layer on validation and report causal abstraction on test."""
    indexed = {(str(row["id"]), int(row["layer"])): row for row in patches}
    layers = sorted({int(row["layer"]) for row in patches})
    pair_by_id = {str(pair["id"]): pair for pair in pairs}

    def clusters(ids: list[str]) -> list[str]:
        return [str(pair_by_id[pair_id]["group"]) for pair_id in ids]

    def cluster_mean(values: list[float], ids: list[str]) -> float:
        labels = clusters(ids)
        return float(
            np.mean(
                [
                    np.mean(
                        [value for value, label in zip(values, labels) if label == group]
                    )
                    for group in sorted(set(labels))
                ]
            )
        )

    def probability(record: dict[str, Any], target: int) -> float:
        return float(np.exp(record["final_candidate_logprobabilities"][target]))

    def values(
        pair_id: str, layer: int, mode: str
    ) -> tuple[float, float, float, bool]:
        pair = pair_by_id[pair_id]
        target = int(
            pair["different_target"] if "different" in mode else pair["recipient_target"]
        )
        baseline = captures[pair_id]["conditions"]["compose"]
        patched = indexed[(pair_id, layer)]["conditions"][mode]
        shift = probability(patched, target) - probability(baseline, target)
        divergence = jensen_shannon(
            np.asarray(baseline["final_candidate_probabilities"]),
            np.asarray(patched["final_candidate_probabilities"]),
        )
        mass_shift = abs(
            float(patched["candidate_probability_mass"])
            - float(baseline["candidate_probability_mass"])
        )
        return (
            shift,
            divergence,
            mass_shift,
            bool(patched["is_expected_unconstrained"]),
        )

    validation_ids = [str(pair["id"]) for pair in pairs if pair["split"] == "validation"]
    validation_scores = {
        layer: cluster_mean(
            [
                values(pair_id, layer, "state_different")[0]
                - values(pair_id, layer, "random_different")[0]
                - values(pair_id, layer, "state_same")[1]
                - values(pair_id, layer, "state_same")[2]
                for pair_id in validation_ids
            ],
            validation_ids,
        )
        for layer in layers
    }
    selected = max(layers, key=lambda layer: (validation_scores[layer], -layer))
    test_ids = [str(pair["id"]) for pair in pairs if pair["split"] == "test"]
    metrics = {}
    for mode_index, mode in enumerate(PATCH_MODES):
        rows = [values(pair_id, selected, mode) for pair_id in test_ids]
        metrics[mode] = {
            "target_probability_shift": cluster_bootstrap_mean_ci(
                [row[0] for row in rows],
                clusters(test_ids),
                seed=2100 + mode_index,
            ),
            "candidate_jsd": cluster_bootstrap_mean_ci(
                [row[1] for row in rows],
                clusters(test_ids),
                seed=2110 + mode_index,
            ),
            "candidate_mass_absolute_shift": cluster_bootstrap_mean_ci(
                [row[2] for row in rows],
                clusters(test_ids),
                seed=2115 + mode_index,
            ),
            "accuracy": cluster_bootstrap_mean_ci(
                [row[3] for row in rows],
                clusters(test_ids),
                seed=2120 + mode_index,
            ),
        }
    state_over_random = cluster_bootstrap_mean_ci(
        [
            values(pair_id, selected, "state_different")[0]
            - values(pair_id, selected, "random_different")[0]
            for pair_id in test_ids
        ],
        clusters(test_ids),
        seed=2130,
    )
    thresholds = {
        "min_state_shift_lower": 0.05,
        "min_state_over_random_lower": 0.05,
        "min_full_shift_lower": 0.10,
        "max_state_same_jsd_upper": 0.05,
        "max_state_same_mass_shift_upper": 0.05,
        "max_full_same_jsd_upper": 0.05,
        "max_full_same_mass_shift_upper": 0.05,
        **gate,
    }

    def lower(stat: dict[str, Any]) -> float:
        return float(stat["ci95"][0])

    def upper(stat: dict[str, Any]) -> float:
        return float(stat["ci95"][1])

    compact = (
        lower(metrics["state_different"]["target_probability_shift"])
        >= float(thresholds["min_state_shift_lower"])
        and lower(state_over_random)
        >= float(thresholds["min_state_over_random_lower"])
    )
    full = (
        lower(metrics["full_different"]["target_probability_shift"])
        >= float(thresholds["min_full_shift_lower"])
    )
    state_invariant = (
        upper(metrics["state_same"]["candidate_jsd"])
        <= float(thresholds["max_state_same_jsd_upper"])
        and upper(metrics["state_same"]["candidate_mass_absolute_shift"])
        <= float(thresholds["max_state_same_mass_shift_upper"])
    )
    endpoint_invariant = (
        upper(metrics["full_same"]["candidate_jsd"])
        <= float(thresholds["max_full_same_jsd_upper"])
        and upper(metrics["full_same"]["candidate_mass_absolute_shift"])
        <= float(thresholds["max_full_same_mass_shift_upper"])
    )
    if compact and state_invariant and endpoint_invariant:
        interpretation = "causal_history_quotient_at_the_endpoint"
    elif compact and state_invariant:
        interpretation = "compact_causal_state_with_path_residue"
    elif full and endpoint_invariant:
        interpretation = "distributed_causal_state_at_history_endpoint"
    elif full:
        interpretation = "path_bound_causal_history_representation"
    else:
        interpretation = "no_causal_state_at_the_tested_history_endpoint"
    return {
        "schema_version": 1,
        "split_counts": {
            split: sum(pair["split"] == split for pair in pairs)
            for split in ("train", "validation", "test")
        },
        "program_context_counts": {
            split: len(
                {
                    str(pair["group"])
                    for pair in pairs
                    if pair["split"] == split
                }
            )
            for split in ("train", "validation", "test")
        },
        "inference_unit": "program context",
        "layer_selection": {
            "selected": selected,
            "validation_scores": validation_scores,
        },
        "heldout": {"case_count": len(test_ids), "metrics": metrics},
        "state_over_random": state_over_random,
        "gate": {
            "thresholds": thresholds,
            "checks": {
                "compact_state": compact,
                "full_state": full,
                "state_subspace_invariance": state_invariant,
                "full_endpoint_invariance": endpoint_invariant,
            },
            "passed": compact and state_invariant,
        },
        "interpretation": interpretation,
    }
