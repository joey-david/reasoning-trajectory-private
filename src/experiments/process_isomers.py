"""Mine strict symbolic-state-equivalent trace pairs for causal patching."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from src.runtime.data import load_samples, write_jsonl


def write_process_isomer_pairs(
    h2_dir: Path,
    output_path: Path,
    *,
    per_sample: int = 2,
    max_pairs: int = 30,
) -> Path:
    """Write cross-question pairs with identical cumulative symbolic graphs."""
    updates = load_samples((h2_dir / "updates.jsonl").resolve())
    allowed_seeds: defaultdict[str, list[int]] = defaultdict(list)
    for update in updates:
        sample_id = str(update["sample_id"])
        seed = int(update["seed"])
        if seed not in allowed_seeds[sample_id]:
            allowed_seeds[sample_id].append(seed)
    allowed = {
        sample_id: set(sorted(seeds)[:per_sample])
        for sample_id, seeds in allowed_seeds.items()
    }

    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        state = str(update.get("graph_signature", ""))
        if (
            state
            and int(update["seed"]) in allowed[str(update["sample_id"])]
            and update["operator"] not in {"BIND", "EXTRACT"}
        ):
            by_state[state].append(update)

    candidates: list[
        tuple[tuple[int, int, int, int], dict[str, Any], dict[str, Any]]
    ] = []
    for state, state_updates in by_state.items():
        for left, right in combinations(state_updates, 2):
            same_trajectory = (
                left["sample_id"] == right["sample_id"]
                and left["seed"] == right["seed"]
            )
            history_distance = abs(int(left["token_end"]) - int(right["token_end"]))
            if same_trajectory or history_distance < 10:
                continue
            lexical_overlap = len(
                set(left.get("lexical_items", [])) & set(right.get("lexical_items", []))
            )
            score = (
                0 if left["sample_id"] == right["sample_id"] else 1,
                lexical_overlap,
                -history_distance,
                len(state),
            )
            candidates.append((score, left, right))

    pairs: list[dict[str, Any]] = []
    used_trajectories: set[tuple[str, int]] = set()
    for _, left, right in sorted(candidates, key=lambda item: item[0]):
        left_key = (str(left["sample_id"]), int(left["seed"]))
        right_key = (str(right["sample_id"]), int(right["seed"]))
        if left_key in used_trajectories and right_key in used_trajectories:
            continue
        pairs.append(
            {
                "pair_id": len(pairs),
                "graph_signature": left["graph_signature"],
                "donor": patch_point(left),
                "target": patch_point(right),
                "lexical_overlap": len(
                    set(left.get("lexical_items", []))
                    & set(right.get("lexical_items", []))
                ),
            }
        )
        used_trajectories.update((left_key, right_key))
        if len(pairs) >= max_pairs:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, pairs)
    return output_path


def patch_point(update: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": update["sample_id"],
        "seed": update["seed"],
        "temperature": update.get("temperature", 0.6),
        "token_end": update["token_end"],
        "state_index": update["state_index"],
        "operator": update["operator"],
        "operation_signature": update["operation_signature"],
        "value": update["value"],
        "update_index": update["update_index"],
    }
