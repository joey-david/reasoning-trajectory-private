"""Second-wave datasets for alignment, future utility, and correction."""

from __future__ import annotations

import random
from typing import Any

from .dataset_utils import experiment_row, value_prompt


def _aligned_value_prompt(
    *, value: int, shift: int | None, run_label: str
) -> dict[str, Any]:
    continuation = (
        "Return that active value unchanged."
        if shift is None
        else f"Continue from that value: add {shift} modulo 10."
    )
    return value_prompt(
        (
            f"Run {run_label}. Maintain one active decimal value modulo 10.\n"
            "The latest assignment is authoritative.\n"
            "Active value = "
        ),
        value,
        (
            f"\n{continuation}\n"
            "Return one digit.\nAnswer="
        ),
    )


def _misaligned_value_prompt(
    *, value: int, shift: int | None, run_label: str
) -> dict[str, Any]:
    continuation = (
        "Report the carried amount unchanged."
        if shift is None
        else f"Increase the carried amount by {shift}, wrapping after nine."
    )
    return value_prompt(
        (
            f"Record {run_label}. Arithmetic is modulo ten.\n"
            "Discard every earlier quantity. The amount to carry forward is "
        ),
        value,
        (
            f".\n{continuation}\n"
            "Give only the resulting numeral.\nAnswer="
        ),
    )


def build_trace_alignment(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Cross formal value equality with trace-coordinate alignment."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        target = rng.randrange(10)
        donor = rng.randrange(10)
        while donor == target:
            donor = rng.randrange(10)
        operation = "read" if index % 2 == 0 else "add"
        shift = None if operation == "read" else rng.randrange(1, 10)
        target_answer = (
            target if shift is None else (target + shift) % 10
        )
        donor_answer = donor if shift is None else (donor + shift) % 10
        prompts = {
            "target": _aligned_value_prompt(
                value=target, shift=shift, run_label="aligned"
            ),
            "aligned_different": _aligned_value_prompt(
                value=donor, shift=shift, run_label="aligned"
            ),
            "aligned_same": _aligned_value_prompt(
                value=target, shift=shift, run_label="aligned"
            ),
            "misaligned_different": _misaligned_value_prompt(
                value=donor, shift=shift, run_label="alternate"
            ),
            "misaligned_same": _misaligned_value_prompt(
                value=target, shift=shift, run_label="alternate"
            ),
        }
        evaluations = [
            {
                "name": "target_baseline",
                "recipient": "target",
                "expected": target_answer,
            },
            {
                "name": "aligned_different_swap",
                "recipient": "target",
                "source": "aligned_different",
                "expected": donor_answer,
                "representation_pair": "aligned_different",
            },
            {
                "name": "misaligned_different_swap",
                "recipient": "target",
                "source": "misaligned_different",
                "expected": donor_answer,
                "representation_pair": "misaligned_different",
            },
            {
                "name": "aligned_same_swap",
                "recipient": "target",
                "source": "aligned_same",
                "expected": target_answer,
                "representation_pair": "aligned_same",
            },
            {
                "name": "misaligned_same_swap",
                "recipient": "target",
                "source": "misaligned_same",
                "expected": target_answer,
                "representation_pair": "misaligned_same",
            },
        ]
        pairs = [
            {
                "name": name,
                "left": "target",
                "right": name,
            }
            for name in (
                "aligned_different",
                "misaligned_different",
                "aligned_same",
                "misaligned_same",
            )
        ]
        rows.append(
            experiment_row(
                experiment="trace_alignment",
                index=index,
                count=count,
                prompts=prompts,
                evaluations=evaluations,
                representation_pairs=pairs,
                labels={
                    "target_value": target,
                    "donor_value": donor,
                    "operation": operation,
                    "shift": shift,
                    "target_answer": target_answer,
                    "donor_answer": donor_answer,
                },
            )
        )
    return rows


def _utility_prompt(
    *,
    a: int,
    b: int,
    goal: str,
    checkpoint: str,
    uses: int,
    distance: int,
) -> dict[str, Any]:
    a_prefix = (
        f"Goal: later use register {goal} exactly {uses} time(s).\n"
        f"Register A: {a - 1} + 1 = "
    )
    middle = f"\nRegister B: {b + 1} - 1 = "
    tail = "\n" + "\n".join(
        f"Delay {step}: registers stay unchanged." for step in range(distance)
    )
    tail += (
        f"\nReturn register {goal} multiplied by {uses}, modulo 10."
        "\nReturn one digit.\nAnswer="
    )
    if checkpoint == "A":
        return value_prompt(a_prefix, a, middle + str(b) + tail)
    return value_prompt(a_prefix + str(a) + middle, b, tail)


def build_prospective_utility(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Mark the same result as useful or irrelevant before its later use."""
    rng = random.Random(seed)
    distances = (2, 8, 24)
    rows = []
    for index in range(count):
        a, b = rng.sample(range(1, 10), 2)
        uses = 1 + index % 3
        distance = distances[(index // 3) % len(distances)]
        a_answer = (a * uses) % 10
        b_answer = (b * uses) % 10
        prompts = {
            "a_relevant": _utility_prompt(
                a=a,
                b=b,
                goal="A",
                checkpoint="A",
                uses=uses,
                distance=distance,
            ),
            "a_irrelevant": _utility_prompt(
                a=a,
                b=b,
                goal="B",
                checkpoint="A",
                uses=uses,
                distance=distance,
            ),
            "b_relevant": _utility_prompt(
                a=a,
                b=b,
                goal="B",
                checkpoint="B",
                uses=uses,
                distance=distance,
            ),
            "b_irrelevant": _utility_prompt(
                a=a,
                b=b,
                goal="A",
                checkpoint="B",
                uses=uses,
                distance=distance,
            ),
        }
        feature_prompt = "a_relevant" if index % 2 == 0 else "a_irrelevant"
        rows.append(
            experiment_row(
                experiment="prospective_utility",
                index=index,
                count=count,
                prompts=prompts,
                evaluations=[
                    {
                        "name": "goal_a_baseline",
                        "recipient": "a_relevant",
                        "expected": a_answer,
                    },
                    {
                        "name": "goal_b_baseline",
                        "recipient": "b_relevant",
                        "expected": b_answer,
                    },
                    {
                        "name": "a_relevance_transfer",
                        "recipient": "a_irrelevant",
                        "source": "a_relevant",
                        "expected": a_answer,
                        "representation_pair": "a_utility_flip",
                    },
                    {
                        "name": "b_relevance_transfer",
                        "recipient": "b_irrelevant",
                        "source": "b_relevant",
                        "expected": b_answer,
                        "representation_pair": "b_utility_flip",
                    },
                ],
                representation_pairs=[
                    {
                        "name": "a_utility_flip",
                        "left": "a_relevant",
                        "right": "a_irrelevant",
                    },
                    {
                        "name": "b_utility_flip",
                        "left": "b_relevant",
                        "right": "b_irrelevant",
                    },
                ],
                labels={
                    "will_reuse": int(index % 2 == 0),
                    "dependency_count": uses,
                    "reuse_distance": distance,
                },
                feature_prompt=feature_prompt,
            )
        )
    return rows


def _correction_prompt(
    *,
    correct: int,
    wrong: int,
    shift: int,
    mode: str,
) -> dict[str, Any]:
    task = (
        "Maintain the verified decimal state modulo 10.\n"
        f"After the state is fixed, add {shift} modulo 10.\n"
    )
    suffix = "\nContinue from that state now.\nReturn one digit.\nAnswer="
    if mode == "corrected":
        prefix = (
            task
            + f"An earlier note incorrectly said the state was {wrong}.\n"
            + "Correction: verified current state = "
        )
        return value_prompt(prefix, correct, suffix)
    if mode == "uncorrected":
        return value_prompt(task + "Current state = ", wrong, suffix)
    if mode == "restart":
        return value_prompt(task + "Verified current state = ", correct, suffix)
    raise ValueError(f"Unknown correction mode: {mode}")


def build_correction_hysteresis(
    *, count: int, seed: int
) -> list[dict[str, Any]]:
    """Separate text correction from the state carried by the corrected token."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        correct = rng.randrange(10)
        wrong = rng.randrange(10)
        while wrong == correct:
            wrong = rng.randrange(10)
        shift = rng.randrange(1, 10)
        correct_answer = (correct + shift) % 10
        wrong_answer = (wrong + shift) % 10
        prompts = {
            mode: _correction_prompt(
                correct=correct,
                wrong=wrong,
                shift=shift,
                mode=mode,
            )
            for mode in ("corrected", "uncorrected", "restart")
        }
        rows.append(
            experiment_row(
                experiment="correction_hysteresis",
                index=index,
                count=count,
                prompts=prompts,
                evaluations=[
                    {
                        "name": "text_correction",
                        "recipient": "corrected",
                        "expected": correct_answer,
                    },
                    {
                        "name": "hidden_correction",
                        "recipient": "uncorrected",
                        "source": "restart",
                        "expected": correct_answer,
                        "representation_pair": "correct_vs_wrong",
                    },
                    {
                        "name": "text_and_hidden_correction",
                        "recipient": "corrected",
                        "source": "restart",
                        "expected": correct_answer,
                        "representation_pair": "restart_vs_corrected",
                    },
                    {
                        "name": "old_state_into_correction",
                        "recipient": "corrected",
                        "source": "uncorrected",
                        "expected": wrong_answer,
                        "representation_pair": "correct_vs_wrong",
                    },
                    {
                        "name": "corrected_restart",
                        "recipient": "restart",
                        "expected": correct_answer,
                    },
                ],
                representation_pairs=[
                    {
                        "name": "correct_vs_wrong",
                        "left": "restart",
                        "right": "uncorrected",
                    },
                    {
                        "name": "restart_vs_corrected",
                        "left": "restart",
                        "right": "corrected",
                    },
                ],
                labels={
                    "correct_state": correct,
                    "wrong_state": wrong,
                    "shift": shift,
                },
            )
        )
    return rows
