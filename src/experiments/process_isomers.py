"""Mine graph-equivalent traces with demonstrably different symbolic histories."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.common import read_generation_rows
from src.runtime.artifact_store import write_json
from src.runtime.data import load_samples, write_jsonl


def write_process_isomer_pairs(
    h2_dir: Path,
    output_path: Path,
    *,
    activation_run: Path | None = None,
    generation_run: Path | None = None,
    audit_path: Path | None = None,
    per_sample: int = 3,
    max_pairs: int = 30,
    min_pairs: int = 20,
    min_path_edits: int = 2,
    min_normalized_path_distance: float = 0.2,
    max_pairs_per_question: int = 2,
    max_trajectory_reuse: int = 2,
    max_target_remaining_tokens: int | None = None,
    require_target_correct: bool = False,
) -> Path:
    """Write exact-state pairs whose ordered symbolic derivations differ."""
    updates = load_unique_updates(h2_dir)
    available = available_trace_lengths(activation_run)
    generation_rows = generation_trace_metadata(generation_run)
    if max_target_remaining_tokens is not None and generation_rows is None:
        raise ValueError(
            "generation_run is required when max_target_remaining_tokens is set"
        )
    rejection_counts: Counter[str] = Counter()

    by_trajectory: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for update in updates:
        by_trajectory[(str(update["sample_id"]), int(update["seed"]))].append(update)

    allowed_seeds: defaultdict[str, list[int]] = defaultdict(list)
    for sample_id, seed in sorted(by_trajectory):
        allowed_seeds[sample_id].append(seed)
    allowed = {
        sample_id: set(sorted(seeds)[:per_sample])
        for sample_id, seeds in allowed_seeds.items()
    }

    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_updates = 0
    for trace_key, trace_updates in by_trajectory.items():
        trace_updates.sort(key=lambda update: int(update["update_index"]))
        history: list[str] = []
        previous_graph = ""
        for update in trace_updates:
            state = str(update.get("graph_signature", ""))
            if state != previous_graph:
                history.append(structural_step_signature(update, previous_graph))
            previous_graph = state
            update["_structural_history"] = tuple(history)
            if not state:
                rejection_counts["empty_graph_state"] += 1
                continue
            if trace_key[1] not in allowed[trace_key[0]]:
                rejection_counts["seed_outside_replay_subset"] += 1
                continue
            if update["operator"] in {"BIND", "EXTRACT"}:
                rejection_counts["unsupported_endpoint_operator"] += 1
                continue
            if available is not None:
                token_count = available.get(trace_key)
                if token_count is None:
                    rejection_counts["missing_activation_trace"] += 1
                    continue
                if int(update["token_end"]) + 1 >= token_count:
                    rejection_counts["missing_completed_state"] += 1
                    continue
            by_state[state].append(update)
            eligible_updates += 1

    candidates = []
    for state_updates in by_state.values():
        for left, right in combinations(state_updates, 2):
            left_key = (str(left["sample_id"]), int(left["seed"]))
            right_key = (str(right["sample_id"]), int(right["seed"]))
            if left_key == right_key:
                rejection_counts["same_trajectory"] += 1
                continue
            if left["sample_id"] != right["sample_id"]:
                rejection_counts["different_question"] += 1
                continue
            left_history = left["_structural_history"]
            right_history = right["_structural_history"]
            edits = sequence_edit_distance(left_history, right_history)
            normalized = edits / max(len(left_history), len(right_history), 1)
            if edits == 0:
                rejection_counts["identical_symbolic_history"] += 1
                continue
            if edits < min_path_edits or normalized < min_normalized_path_distance:
                rejection_counts["insufficient_path_diversity"] += 1
                continue
            lexical_overlap = len(
                set(left.get("lexical_items", [])) & set(right.get("lexical_items", []))
            )
            donor = left
            target = right
            donor_remaining = None
            target_remaining = None
            if generation_rows is not None:
                left_generation = generation_rows.get(left_key)
                right_generation = generation_rows.get(right_key)
                if left_generation is None or right_generation is None:
                    rejection_counts["missing_generation_trace"] += 1
                    continue
                left_remaining = (
                    int(left_generation["token_count"]) - int(left["token_end"]) - 1
                )
                right_remaining = (
                    int(right_generation["token_count"]) - int(right["token_end"]) - 1
                )
                if left_remaining < 0 or right_remaining < 0:
                    rejection_counts["endpoint_after_generation"] += 1
                    continue
                endpoint_options = [
                    (left, left_remaining, left_generation, right, right_remaining),
                    (right, right_remaining, right_generation, left, left_remaining),
                ]
                within_budget = [
                    option
                    for option in endpoint_options
                    if (
                        max_target_remaining_tokens is None
                        or option[1] <= max_target_remaining_tokens
                    )
                ]
                if not within_budget:
                    rejection_counts["target_tail_exceeds_budget"] += 1
                    continue
                eligible_targets = [
                    option
                    for option in within_budget
                    if not require_target_correct
                    or (option[2]["has_answer"] and option[2]["is_correct"])
                ]
                if not eligible_targets:
                    rejection_counts["no_correct_target"] += 1
                    continue
                (
                    target,
                    target_remaining,
                    _target_generation,
                    donor,
                    donor_remaining,
                ) = min(eligible_targets, key=lambda option: option[1])
            candidates.append(
                {
                    "graph_signature": target["graph_signature"],
                    "donor": donor,
                    "target": target,
                    "path_edit_distance": edits,
                    "normalized_path_distance": normalized,
                    "lexical_overlap": lexical_overlap,
                    "token_distance": abs(
                        int(left["token_end"]) - int(right["token_end"])
                    ),
                    "donor_remaining_tokens": donor_remaining,
                    "target_remaining_tokens": target_remaining,
                }
            )

    candidates = deduplicate_pair_candidates(candidates, rejection_counts)
    candidates.sort(
        key=lambda candidate: (
            -candidate["normalized_path_distance"],
            -candidate["path_edit_distance"],
            candidate["lexical_overlap"],
            candidate["target_remaining_tokens"]
            if candidate["target_remaining_tokens"] is not None
            else 0,
            -candidate["token_distance"],
        )
    )
    trajectory_reuse: Counter[tuple[str, int]] = Counter()
    question_pairs: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    for candidate in candidates:
        donor = candidate["donor"]
        target = candidate["target"]
        donor_key = (str(donor["sample_id"]), int(donor["seed"]))
        target_key = (str(target["sample_id"]), int(target["seed"]))
        question = str(target["sample_id"])
        if question_pairs[question] >= max_pairs_per_question:
            rejection_counts["question_pair_cap"] += 1
            continue
        if (
            trajectory_reuse[donor_key] >= max_trajectory_reuse
            or trajectory_reuse[target_key] >= max_trajectory_reuse
        ):
            rejection_counts["trajectory_reuse_cap"] += 1
            continue
        donor_history = list(donor["_structural_history"])
        target_history = list(target["_structural_history"])
        pair = {
            "pair_id": len(pairs),
            "graph_signature": target["graph_signature"],
            "donor": patch_point(donor),
            "target": patch_point(target),
            "same_question": True,
            "lexical_overlap": candidate["lexical_overlap"],
            "path_evidence": {
                "edit_distance": candidate["path_edit_distance"],
                "normalized_edit_distance": candidate["normalized_path_distance"],
                "donor_history_length": len(donor_history),
                "target_history_length": len(target_history),
                "donor_history_hash": history_hash(donor_history),
                "target_history_hash": history_hash(target_history),
                "donor_steps": donor_history,
                "target_steps": target_history,
            },
        }
        if candidate["target_remaining_tokens"] is not None:
            target_generation = generation_rows[target_key]
            pair["continuation_evidence"] = {
                "target_remaining_tokens": candidate["target_remaining_tokens"],
                "donor_remaining_tokens": candidate["donor_remaining_tokens"],
                "target_source_has_answer": target_generation["has_answer"],
                "target_source_correct": target_generation["is_correct"],
            }
        pairs.append(pair)
        trajectory_reuse.update((donor_key, target_key))
        question_pairs[question] += 1
        if len(pairs) >= max_pairs:
            break

    if generation_rows is not None:
        pairs.sort(
            key=lambda pair: pair["continuation_evidence"]["target_remaining_tokens"]
        )
        for pair_id, pair in enumerate(pairs):
            pair["pair_id"] = pair_id
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, pairs)
    audit_path = audit_path or output_path.with_name(f"{output_path.stem}_audit.json")
    distances = np.asarray(
        [pair["path_evidence"]["normalized_edit_distance"] for pair in pairs],
        dtype=np.float64,
    )
    remaining_tokens = np.asarray(
        [
            pair["continuation_evidence"]["target_remaining_tokens"]
            for pair in pairs
            if "continuation_evidence" in pair
        ],
        dtype=np.int64,
    )
    audit = {
        "source": h2_dir.as_posix(),
        "activation_run": activation_run.as_posix() if activation_run else None,
        "generation_run": generation_run.as_posix() if generation_run else None,
        "criteria": {
            "exact_graph_signature": True,
            "same_question": True,
            "graph_changing_history_only": True,
            "per_sample": per_sample,
            "min_path_edits": min_path_edits,
            "min_normalized_path_distance": min_normalized_path_distance,
            "max_pairs_per_question": max_pairs_per_question,
            "max_trajectory_reuse": max_trajectory_reuse,
            "max_target_remaining_tokens": max_target_remaining_tokens,
            "require_target_correct": require_target_correct,
        },
        "yield": {
            "source_updates": len(updates),
            "eligible_updates": eligible_updates,
            "graph_states": len(by_state),
            "path_distinct_candidates": len(candidates),
            "accepted_pairs": len(pairs),
            "accepted_questions": len(question_pairs),
            "accepted_trajectories": len(trajectory_reuse),
        },
        "path_distance": {
            "minimum": float(distances.min()) if len(distances) else None,
            "mean": float(distances.mean()) if len(distances) else None,
            "maximum": float(distances.max()) if len(distances) else None,
        },
        "target_source_tail": {
            "minimum_tokens": int(remaining_tokens.min())
            if len(remaining_tokens)
            else None,
            "median_tokens": float(np.median(remaining_tokens))
            if len(remaining_tokens)
            else None,
            "maximum_tokens": int(remaining_tokens.max())
            if len(remaining_tokens)
            else None,
            "source_answer_present": sum(
                bool(pair["continuation_evidence"]["target_source_has_answer"])
                for pair in pairs
                if "continuation_evidence" in pair
            ),
            "source_answer_correct": sum(
                bool(pair["continuation_evidence"]["target_source_correct"])
                for pair in pairs
                if "continuation_evidence" in pair
            ),
        },
        "rejections": dict(sorted(rejection_counts.items())),
    }
    write_json(audit_path, audit)
    if len(pairs) < min_pairs:
        raise ValueError(
            f"Only {len(pairs)} path-distinct process-isomer pairs survived; "
            f"see {audit_path}"
        )
    return output_path


def deduplicate_pair_candidates(
    candidates: list[dict[str, Any]],
    rejection_counts: Counter[str],
) -> list[dict[str, Any]]:
    """Collapse repeated visits with the same trace histories and graph state."""
    unique_candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in candidates:
        donor = candidate["donor"]
        target = candidate["target"]
        key = (
            candidate["graph_signature"],
            str(donor["sample_id"]),
            int(donor["seed"]),
            history_hash(list(donor["_structural_history"])),
            str(target["sample_id"]),
            int(target["seed"]),
            history_hash(list(target["_structural_history"])),
        )
        current = unique_candidates.get(key)
        preference = (
            int(target["update_index"]),
            int(donor["update_index"]),
            int(target["token_end"]),
            int(donor["token_end"]),
        )
        if current is None:
            unique_candidates[key] = candidate
            continue
        current_preference = (
            int(current["target"]["update_index"]),
            int(current["donor"]["update_index"]),
            int(current["target"]["token_end"]),
            int(current["donor"]["token_end"]),
        )
        if preference < current_preference:
            unique_candidates[key] = candidate
        rejection_counts["duplicate_pair_history"] += 1
    return list(unique_candidates.values())


def load_unique_updates(h2_dir: Path) -> list[dict[str, Any]]:
    """Deduplicate update rows if an H2 artifact contains multiple layers."""
    unique = {}
    for update in load_samples((h2_dir / "updates.jsonl").resolve()):
        key = (
            str(update["sample_id"]),
            int(update["seed"]),
            int(update["update_index"]),
        )
        unique.setdefault(key, update)
    return list(unique.values())


def available_trace_lengths(
    activation_run: Path | None,
) -> dict[tuple[str, int], int] | None:
    if activation_run is None:
        return None
    return {
        (str(row["sample_id"]), int(row["seed"])): len(row["generated_token_ids"])
        for row in read_generation_rows(activation_run)
        if row.get("hidden_states_file")
    }


def generation_trace_metadata(
    generation_run: Path | None,
) -> dict[tuple[str, int], dict[str, Any]] | None:
    if generation_run is None:
        return None
    return {
        (str(row["sample_id"]), int(row["seed"])): {
            "token_count": len(row["generated_token_ids"]),
            "has_answer": row.get("produced_answer") is not None,
            "is_correct": bool(row.get("is_correct")),
        }
        for row in read_generation_rows(generation_run)
    }


def structural_step_signature(update: dict[str, Any], previous_graph: str) -> str:
    """Describe one graph transition without using surface phrasing."""
    previous = set(filter(None, previous_graph.split("|")))
    current = set(filter(None, str(update["graph_signature"]).split("|")))
    added = sorted(current - previous)
    removed = sorted(previous - current)
    return json.dumps(
        {
            "operator": update["operator"],
            "operation": update["operation_signature"],
            "added": added,
            "removed": removed,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def sequence_edit_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Return Levenshtein distance between two ordered symbolic histories."""
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def history_hash(history: list[str]) -> str:
    payload = "\n".join(history).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def patch_point(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": update["sample_id"],
        "seed": update["seed"],
        "temperature": update.get("temperature", 0.6),
        "token_end": update["token_end"],
        "operator": update["operator"],
        "operation_signature": update["operation_signature"],
        "graph_signature": update["graph_signature"],
        "value": update["value"],
        "update_index": update["update_index"],
    }
