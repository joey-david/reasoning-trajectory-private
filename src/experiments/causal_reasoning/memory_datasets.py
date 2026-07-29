"""Second-wave dataset for unresolved dependency state."""

from __future__ import annotations

import random
from typing import Any

from .dataset_utils import experiment_row, marker_prompt


def _debt_expression(
    family: str, rng: random.Random
) -> tuple[str, int]:
    if family == "arithmetic":
        base = rng.randrange(10)
        add = rng.randrange(1, 10)
        return f"x = ({base} + {add}) modulo 10", (base + add) % 10
    if family == "logical":
        base = rng.randrange(8)
        mask = rng.randrange(1, 8)
        return f"x = {base} XOR {mask}", base ^ mask
    table = rng.sample(range(8), 8)
    pointer = rng.randrange(8)
    rendered = ", ".join(str(value) for value in table)
    return f"x = entry {pointer} of [{rendered}]", table[pointer]


def _debt_prompt(
    *,
    expression: str,
    value: int,
    shift: int,
    distance: int,
    mode: str,
    debt_count: int,
) -> dict[str, Any]:
    lines = [
        f"Final rule: add {shift} to x modulo 10.",
    ]
    if mode == "known":
        lines.append(f"x is already known: x = {value}.")
    elif mode == "recoverable":
        lines.extend(
            [
                f"The defining expression is available: {expression}.",
                "Leave x symbolic until it is needed.",
            ]
        )
    elif mode in {"deferred", "unpaid"}:
        lines.append("x is unavailable now and is promised later.")
    elif mode == "irrelevant":
        lines.extend(
            [
                f"x is already known: x = {value}.",
                "y is unavailable, but the final rule does not depend on y.",
            ]
        )
    else:
        raise ValueError(f"Unknown debt mode: {mode}")
    if debt_count == 2 and mode != "irrelevant":
        lines.append("y is also unavailable and irrelevant to the final rule.")
    lines.append("<<DEBT>>")
    lines.extend(
        f"Delay {step}: preserve every pending dependency."
        for step in range(distance)
    )
    if mode == "deferred":
        lines.append(f"Promised value arrives: x = {value}.")
    elif mode == "recoverable":
        lines.append("Resolve x from its earlier defining expression now.")
    elif mode == "unpaid":
        lines.append("The promised value never arrives.")
    lines.extend(
        [
            "Apply the final rule if its dependency is available.",
            "Return one digit.",
            "Answer=",
        ]
    )
    return marker_prompt(lines, "<<DEBT>>")


def build_information_debt(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Measure whether unresolved dependencies survive type, count, and delay."""
    rng = random.Random(seed)
    families = ("arithmetic", "logical", "retrieval")
    distances = (2, 8, 24)
    modes = ("known", "recoverable", "deferred", "irrelevant")
    rows = []
    for index in range(count):
        family = families[index % len(families)]
        distance = distances[(index // len(families)) % len(distances)]
        debt_count = 1 + (index % 2)
        expression, value = _debt_expression(family, rng)
        shift = rng.randrange(1, 10)
        answer = (value + shift) % 10
        prompts = {
            mode: _debt_prompt(
                expression=expression,
                value=value,
                shift=shift,
                distance=distance,
                mode=mode,
                debt_count=debt_count,
            )
            for mode in (*modes, "unpaid")
        }
        feature_mode = modes[index % len(modes)]
        rows.append(
            experiment_row(
                experiment="information_debt",
                index=index,
                count=count,
                prompts=prompts,
                evaluations=[
                    {
                        "name": f"{mode}_dependency",
                        "recipient": mode,
                        "expected": answer,
                    }
                    for mode in (*modes, "unpaid")
                ],
                representation_pairs=[
                    {
                        "name": "deferred_vs_known",
                        "left": "deferred",
                        "right": "known",
                    },
                    {
                        "name": "recoverable_vs_deferred",
                        "left": "recoverable",
                        "right": "deferred",
                    },
                    {
                        "name": "irrelevant_vs_deferred",
                        "left": "irrelevant",
                        "right": "deferred",
                    },
                ],
                labels={
                    "debt_kind": modes.index(feature_mode),
                    "debt_count": debt_count,
                    "reuse_distance": distance,
                    "operation_family": families.index(family),
                },
                candidates=[str(value) for value in range(10)],
                feature_prompt=feature_mode,
            )
        )
    return rows
