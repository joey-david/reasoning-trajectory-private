"""Behavioral equivalence and predicted-code interchange from saved calls."""

from __future__ import annotations

from collections import defaultdict
import random
from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci
from .state_handoff_data import TEST_PATH, read_programs
from .state_interface_data import (
    interface_code_symbols,
    semantic_states_for_code,
)
from .state_interface_evaluation import (
    interface_evaluation_dir,
    read_interface_evaluation_cases,
)


def _prediction(result: dict[str, Any] | None) -> int | None:
    if not result:
        return None
    value = result.get("unconstrained_prediction")
    return int(value) if value is not None else None


def _consumer_table(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], int | None]:
    """Recover code-to-answer behavior for every held-out FINAL context."""
    table: dict[tuple[str, int], int | None] = {}
    for row in rows:
        key = (str(row["program_context"]), int(row["true_code"]))
        observed = _prediction(row["gold_final"])
        if key in table and table[key] != observed:
            raise ValueError(f"Consumer output is not deterministic for {key}")
        table[key] = observed
    return table


def _same_state_pairs(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    groups: defaultdict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                int(row["history_steps"]),
                str(row["program_context"]),
                int(row["current_state"]),
            )
        ].append(row)
    pairs = []
    for group in groups.values():
        ordered = sorted(group, key=lambda row: int(row["path_code"]))
        for index, left in enumerate(ordered):
            pairs.extend((left, right) for right in ordered[index + 1 :])
    return pairs


def analyze_predicted_code_equivalence(
    run_path: Path, condition: str
) -> dict[str, Any]:
    """Test whether independently emitted codes have the same causal meaning."""
    rows = read_interface_evaluation_cases(run_path, condition)
    if not rows:
        raise RuntimeError(f"No interface evaluation rows for {condition}")
    programs = {
        str(row["id"]): row for row in read_programs(run_path / TEST_PATH)
    }
    experiment = load_config(run_path).get("state_handoff_training", {})
    interface_config = experiment.get("interfaces", {})
    symbols = interface_code_symbols(condition, interface_config)
    consumer = _consumer_table(rows)
    contexts = sorted({str(row["program_context"]) for row in rows})
    missing = [
        (context, code)
        for context in contexts
        for code in range(len(symbols))
        if (context, code) not in consumer
    ]
    if missing:
        raise ValueError(f"Gold calls do not cover the consumer table: {missing[:5]}")

    predicted_sufficiency = []
    predicted_semantic_accuracy = []
    for row in rows:
        code = row["predicted_code"]
        if code is None:
            predicted_sufficiency.append(False)
            predicted_semantic_accuracy.append(False)
            continue
        context = str(row["program_context"])
        predicted_sufficiency.append(
            consumer[(context, int(code))]
            == int(programs[str(row["id"])]["next_state"])
        )
        compatible = semantic_states_for_code(
            condition=condition,
            case=programs[str(row["id"])],
            code_index=int(code),
            interface_config=interface_config,
        )
        predicted_semantic_accuracy.append(int(row["current_state"]) in compatible)

    exact_agreement = []
    quotient_agreement = []
    predicted_donor_preserves = []
    for recipient, donor in _same_state_pairs(rows):
        left = recipient["predicted_code"]
        right = donor["predicted_code"]
        exact_agreement.append(left is not None and left == right)
        if left is None or right is None:
            quotient_agreement.append(False)
            predicted_donor_preserves.append(False)
            continue
        context = str(recipient["program_context"])
        quotient_agreement.append(
            consumer[(context, int(left))] == consumer[(context, int(right))]
        )
        predicted_donor_preserves.append(
            consumer[(context, int(right))]
            == int(programs[str(recipient["id"])]["next_state"])
        )

    by_group: defaultdict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[
            (
                int(row["history_steps"]),
                str(row["program_context"]),
                int(row["path_code"]),
            )
        ].append(row)
    donor_follows_intended = []
    predicted_call_matches_intervention = []
    for recipient in rows:
        group = by_group[
            (
                int(recipient["history_steps"]),
                str(recipient["program_context"]),
                int(recipient["path_code"]),
            )
        ]
        for donor in group:
            if int(donor["current_state"]) == int(recipient["current_state"]):
                continue
            code = donor["predicted_code"]
            if code is None:
                donor_follows_intended.append(False)
                predicted_call_matches_intervention.append(False)
                continue
            context = str(recipient["program_context"])
            observed = consumer[(context, int(code))]
            donor_follows_intended.append(
                observed == int(programs[str(donor["id"])]["next_state"])
            )
            predicted_call_matches_intervention.append(
                _prediction(donor["predicted_final"]) == observed
            )

    rng = random.Random(72_230)
    frequencies = [
        int(row["predicted_code"])
        for row in rows
        if row["predicted_code"] is not None
    ]
    random_code_preserves = []
    frequency_code_preserves = []
    for row in rows:
        context = str(row["program_context"])
        expected = int(programs[str(row["id"])]["next_state"])
        random_code_preserves.append(
            consumer[(context, rng.randrange(len(symbols)))] == expected
        )
        frequency_code_preserves.append(
            consumer[(context, frequencies[rng.randrange(len(frequencies))])]
            == expected
        )

    metrics = {
        "predicted_code_sufficiency": predicted_sufficiency,
        "predicted_code_semantic_accuracy": predicted_semantic_accuracy,
        "same_state_exact_agreement": exact_agreement,
        "same_state_quotient_agreement": quotient_agreement,
        "same_state_predicted_donor_preservation": predicted_donor_preserves,
        "different_state_predicted_donor_follows_intended_state": (
            donor_follows_intended
        ),
        "saved_predicted_call_matches_code_intervention": (
            predicted_call_matches_intervention
        ),
        "random_code_preservation": random_code_preserves,
        "frequency_matched_code_preservation": frequency_code_preserves,
    }
    summary = {
        "schema_version": 1,
        "condition": condition,
        "case_count": len(rows),
        "program_context_count": len(contexts),
        "codebook_size": len(symbols),
        "metrics": {
            name: bootstrap_mean_ci(values, seed=72_300 + index)
            for index, (name, values) in enumerate(metrics.items())
        },
        "contract": (
            "Consumer outputs come from saved gold-code calls under the same held-out "
            "FINAL context. Donor codes come from independent model predictions."
        ),
    }
    output = interface_evaluation_dir(run_path, condition)
    write_json(output / "predicted_equivalence_summary.json", summary)
    return summary
