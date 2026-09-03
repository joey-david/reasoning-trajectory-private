"""Premise test for the causal source of a later arithmetic read."""

from __future__ import annotations

from collections.abc import Iterable
import re
import random
from typing import Any

import numpy as np
import torch
from transformers import StoppingCriteriaList

from src.experiments.depth_relief.benchmark import candidate_token_ids
from src.experiments.depth_relief.hf import patched_logits
from src.experiments.depth_relief.metrics import bootstrap_mean_ci
from src.models.introspection import get_decoder_layers, get_input_device
from src.models.generation_utils import GeneratedTextRegexStop


NAMES = ("Ada", "Bela", "Cora", "Dani")
ANSWER_RE = re.compile(
    r"Answer\s*[:=]\s*(?P<answer>-?\d+)"
    r"|final score is\s*(?:\\\(\s*)?(?:\\boxed\{)?(?P<final>-?\d+)"
    r"|\\boxed\{(?P<boxed>-?\d+)\}",
    re.IGNORECASE,
)


class _Text:
    """Build text while retaining exact spans without visible markers."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: dict[str, list[int]] = {}
        self.length = 0

    def add(self, value: str) -> None:
        self.parts.append(value)
        self.length += len(value)

    def mark(self, key: str, value: int) -> None:
        rendered = str(value)
        start = self.length
        self.add(rendered)
        self.spans[key] = [start, self.length]

    def finish(self) -> dict[str, Any]:
        return {"text": "".join(self.parts), "spans": self.spans}


def _apply(value: int, operation: str, amount: int) -> int:
    direction = 1 if operation == "add" else -1
    return (value + direction * amount) % 10


def _render(
    *,
    initial: dict[str, int],
    events: list[dict[str, Any]],
    target: str,
) -> tuple[dict[str, Any], int]:
    text = _Text()
    text.add("Four players keep scores modulo 10; after 9 comes 0.\n")
    text.add("Starting scores: ")
    text.add("; ".join(f"{name} has {initial[name]}" for name in NAMES))
    text.add(".\nEvents, in order:\n")
    states = dict(initial)
    target_event = max(i for i, event in enumerate(events) if event["name"] == target)
    trace: list[tuple[str, int, str, int, int]] = []
    for index, event in enumerate(events):
        name = str(event["name"])
        operation = str(event["operation"])
        amount = int(event["amount"])
        text.add(f"{index + 1}. {name} must {operation} ")
        if index == target_event:
            text.mark("problem_source", amount)
        else:
            text.add(str(amount))
        text.add(".\n")
        before = states[name]
        states[name] = _apply(before, operation, amount)
        trace.append((name, before, operation, amount, states[name]))

    text.add(
        f"Work through the events in order. What is {target}'s final score? "
        "Show the arithmetic, then end with Answer=<one digit>.\nSolution:\n"
    )
    for index, (name, before, operation, amount, after) in enumerate(trace):
        symbol = "+" if operation == "add" else "-"
        text.add(f"{name}: {before} {symbol} {amount} mod 10 = ")
        if index == target_event:
            text.mark("cot_write", after)
        else:
            text.add(str(after))
        text.add(".\n")
    text.add(f"Therefore, {target}'s final score is\nAnswer=")
    return text.finish(), states[target]


def _question(teacher: dict[str, Any]) -> str:
    return str(teacher["text"]).split("Solution:\n", 1)[0] + "Solution:\n"


def build_cases(*, count: int, seed: int) -> list[dict[str, Any]]:
    """Build clean/corrupt matched problems with short and long delays."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        difficulty = "easy" if index % 2 == 0 else "hard"
        target = NAMES[index % len(NAMES)]
        initial = {name: rng.randrange(10) for name in NAMES}
        events: list[dict[str, Any]] = []
        target_writes = 1 if difficulty == "easy" else 2
        for _ in range(target_writes):
            events.append(
                {
                    "name": target,
                    "operation": rng.choice(("add", "subtract")),
                    "amount": rng.randrange(1, 9),
                }
            )
            for _ in range(1 if difficulty == "easy" else 4):
                events.append(
                    {
                        "name": rng.choice([name for name in NAMES if name != target]),
                        "operation": rng.choice(("add", "subtract")),
                        "amount": rng.randrange(1, 9),
                    }
                )
        clean, clean_answer = _render(initial=initial, events=events, target=target)
        corrupt_events = [dict(event) for event in events]
        target_event = max(
            i for i, event in enumerate(events) if event["name"] == target
        )
        original = int(corrupt_events[target_event]["amount"])
        replacement = original % 8 + 1
        corrupt_events[target_event]["amount"] = replacement
        corrupt, corrupt_answer = _render(
            initial=initial, events=corrupt_events, target=target
        )
        if clean_answer == corrupt_answer:
            raise AssertionError("counterfactual must change the target answer")
        rows.append(
            {
                "schema_version": 1,
                "id": f"routing_premise_{index:04d}",
                "difficulty": difficulty,
                "target": target,
                "event_count": len(events),
                "clean": clean,
                "corrupt": corrupt,
                "question": _question(clean),
                "clean_answer": clean_answer,
                "corrupt_answer": corrupt_answer,
            }
        )
    return rows


def validate_cases(rows: list[dict[str, Any]], *, expected: int) -> dict[str, Any]:
    """Check deterministic pair and span contracts before model work."""
    ids = [str(row["id"]) for row in rows]
    checks = {
        "count": len(rows) == expected,
        "unique_ids": len(ids) == len(set(ids)),
        "balanced_difficulty": abs(
            sum(row["difficulty"] == "easy" for row in rows)
            - sum(row["difficulty"] == "hard" for row in rows)
        )
        <= 1,
        "single_digit_answers": all(
            0 <= int(row[key]) < 10
            for row in rows
            for key in ("clean_answer", "corrupt_answer")
        ),
        "aligned_characters": all(
            len(row["clean"]["text"]) == len(row["corrupt"]["text"])
            and row["clean"]["spans"] == row["corrupt"]["spans"]
            for row in rows
        ),
        "plain_questions": all(
            row["question"].endswith("Solution:\n")
            and "Therefore," not in row["question"]
            for row in rows
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid routing premise data: {checks}")
    return {"schema_version": 1, "case_count": len(rows), "checks": checks}


def _token_positions(tokenizer: Any, text: str, span: list[int]) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = map(int, span)
    positions = tuple(
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    )
    if len(positions) != 1:
        raise ValueError(f"expected one token for {text[start:end]!r}, got {positions}")
    return positions


def validate_token_contract(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Require aligned pairs, one-token spans, and one-token answer digits."""
    clean = str(row["clean"]["text"])
    corrupt = str(row["corrupt"]["text"])
    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt, add_special_tokens=False)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("clean and corrupt prompts have different token counts")
    positions = {
        key: _token_positions(tokenizer, clean, span)
        for key, span in row["clean"]["spans"].items()
    }
    candidates = candidate_token_ids(tokenizer, clean, tuple(map(str, range(10))))
    return {
        "token_count": len(clean_ids),
        "changed_token_count": sum(a != b for a, b in zip(clean_ids, corrupt_ids)),
        "positions": {key: list(value) for key, value in positions.items()},
        "candidate_ids": candidates,
    }


def _margin(logits: np.ndarray, correct: int, contrast: int) -> float:
    return float(logits[correct] - logits[contrast])


def _model_outputs(
    model: Any, tokenizer: Any, text: str
) -> tuple[np.ndarray, list[torch.Tensor]]:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    device = get_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    states = [value[0].detach().cpu() for value in output.hidden_states[1:]]
    return output.logits[0, -1].float().cpu().numpy(), states


def _free_generation(
    model: Any, tokenizer: Any, question: str, max_new_tokens: int
) -> dict[str, Any]:
    encoded = tokenizer(question, add_special_tokens=False, return_tensors="pt")
    device = get_input_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [
                    GeneratedTextRegexStop(
                        tokenizer,
                        int(encoded["input_ids"].shape[1]),
                        ANSWER_RE.pattern,
                    )
                ]
            ),
        )
    suffix = generated[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(suffix, skip_special_tokens=True)
    answer = parse_answer(text)
    return {"text": text, "answer": answer, "parsed": answer is not None}


def parse_answer(text: str) -> int | None:
    """Read the first explicit answer form and ignore text generated after it."""
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    value = next(value for value in match.groupdict().values() if value is not None)
    return int(value) % 10


def evaluate_case(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    layer_stride: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure causal effects of problem, prior-result, and read sites."""
    contract = validate_token_contract(tokenizer, row)
    clean_text = str(row["clean"]["text"])
    corrupt_text = str(row["corrupt"]["text"])
    clean_logits, clean_states = _model_outputs(model, tokenizer, clean_text)
    corrupt_logits, _ = _model_outputs(model, tokenizer, corrupt_text)
    clean_id = int(contract["candidate_ids"][int(row["clean_answer"])])
    corrupt_id = int(contract["candidate_ids"][int(row["corrupt_answer"])])
    clean_margin = _margin(clean_logits, clean_id, corrupt_id)
    corrupt_margin = _margin(corrupt_logits, clean_id, corrupt_id)
    denominator = clean_margin - corrupt_margin
    positions = {
        **{key: tuple(value) for key, value in contract["positions"].items()},
        "read_site": (contract["token_count"] - 1,),
    }
    effects: dict[str, list[dict[str, float | int]]] = {}
    layers = range(0, len(get_decoder_layers(model)), layer_stride)
    for site, site_positions in positions.items():
        site_rows = []
        for layer in layers:
            values = clean_states[layer][list(site_positions)]
            logits = patched_logits(
                model=model,
                tokenizer=tokenizer,
                text=corrupt_text,
                patches={layer: (site_positions, values)},
            )
            patched_margin = _margin(logits, clean_id, corrupt_id)
            site_rows.append(
                {
                    "layer": layer,
                    "patched_margin": patched_margin,
                    "normalized_indirect_effect": (
                        (patched_margin - corrupt_margin) / denominator
                        if abs(denominator) > 1e-6
                        else 0.0
                    ),
                }
            )
        effects[site] = site_rows
    generation = _free_generation(
        model, tokenizer, str(row["question"]), max_new_tokens
    )
    return {
        "schema_version": 1,
        "id": row["id"],
        "difficulty": row["difficulty"],
        "event_count": row["event_count"],
        "clean_answer": row["clean_answer"],
        "corrupt_answer": row["corrupt_answer"],
        "token_contract": contract,
        "clean_margin": clean_margin,
        "corrupt_margin": corrupt_margin,
        "clean_prefers_clean": clean_margin > 0,
        "corrupt_prefers_corrupt": corrupt_margin < 0,
        "effects": effects,
        "generation": generation,
        "generation_correct": generation["answer"] == row["clean_answer"],
    }


def _peak(row: dict[str, Any], site: str) -> float:
    return max(
        float(cell["normalized_indirect_effect"])
        for cell in row["effects"][site]
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce the premise pilot and make its next-step decision explicit."""
    if not rows:
        raise ValueError("cannot summarize an empty premise pilot")
    eligible = [
        row
        for row in rows
        if row["clean_prefers_clean"] and row["corrupt_prefers_corrupt"]
    ]
    difference = [
        _peak(row, "cot_write") - _peak(row, "problem_source")
        for row in eligible
    ]
    source_gap = bootstrap_mean_ci(difference, seed=260530233) if difference else {
        "mean": None,
        "ci95": [None, None],
        "n": 0,
    }
    parsed_answers = {str(row["id"]): parse_answer(row["generation"]["text"]) for row in rows}
    accuracy = {
        difficulty: bootstrap_mean_ci(
            [
                parsed_answers[str(row["id"])] == int(row["clean_answer"])
                for row in rows
                if row["difficulty"] == difficulty
            ],
            seed=913 + index,
        )
        for index, difficulty in enumerate(("easy", "hard"))
    }
    parsed = sum(value is not None for value in parsed_answers.values()) / len(rows)
    low, high = source_gap["ci95"]
    identified = low is not None and (float(low) > 0.0 or float(high) < 0.0)
    source = (
        "cot_write"
        if identified and float(source_gap["mean"]) > 0
        else "problem_source"
        if identified
        else "ambiguous"
    )
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "eligible_counterfactuals": len(eligible),
        "parsed_generation_rate": parsed,
        "generation_accuracy": accuracy,
        "peak_effect": {
            site: bootstrap_mean_ci([_peak(row, site) for row in eligible], seed=920 + i)
            for i, site in enumerate(("problem_source", "cot_write", "read_site"))
        },
        "cot_minus_problem_effect": source_gap,
        "decision": {
            "causal_source": source,
            "identified": identified,
            "continue_large_run": identified and len(eligible) >= 150,
        },
    }


def select_shard(
    rows: Iterable[dict[str, Any]], *, shard: int, shards: int, max_cases: int | None
) -> list[dict[str, Any]]:
    """Select a deterministic resume-safe array shard."""
    selected = [row for index, row in enumerate(rows) if index % shards == shard]
    return selected if max_cases is None else selected[:max_cases]
