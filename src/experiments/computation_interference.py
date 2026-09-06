"""Test whether prior arithmetic results enter a later, different computation."""

from __future__ import annotations

from contextlib import contextmanager
import random
import re

import numpy as np
import torch

from reasoning_trajectory.token_alignment import token_range_for_chars
from src.experiments.state_routing_heads import select_heads
from src.experiments.state_routing_repair import forward_logits_ids
from src.experiments.symbolic import safe_arithmetic_eval
from src.models.introspection import get_decoder_layers, get_input_device


def build_cases(*, screen_count: int, test_count: int, seed: int) -> list[dict]:
    """Pair true histories; match the misleading result across three controls."""
    rng = random.Random(seed)
    rows, used = [], set()
    for split, count in (("screen", screen_count), ("test", test_count)):
        for index in range(count):
            op = "+" if index % 2 == 0 else "-"
            while True:
                a, b = rng.randint(600, 650), rng.randint(100, 149)
                if (a, b) not in used:
                    used.add((a, b))
                    break
            other = "-" if op == "+" else "+"
            answer = int(safe_arithmetic_eval(f"{a}{op}{b}"))
            lure = int(safe_arithmetic_eval(f"{a}{other}{b}"))
            delta = rng.randint(50, 90)
            u, v = (a + delta, b + delta) if other == "-" else (a - delta, b + delta)
            v2 = rng.randint(160, 190)
            u2 = lure - v2 if op == "+" else lure + v2
            sources = {
                "repeat": f"{a} {op} {b}",
                "operator_lure": f"{a} {other} {b}",
                "value_control": f"{u} {other} {v}",
                "operator_control": f"{u2} {op} {v2}",
            }
            # All operands and results have three digits, including control spans.
            assert all(100 <= n <= 999 for n in (u, v, u2, v2, answer, lure))
            fillers = []
            while len(fillers) < (8 if split == "screen" else 24):
                x, y = rng.randint(300, 450), rng.randint(100, 199)
                sign = rng.choice(("+", "-"))
                value = int(safe_arithmetic_eval(f"{x}{sign}{y}"))
                if not {x, y, value} & {a, b, answer, lure}:
                    fillers.append((f"{x} {sign} {y}", value))
            source_index = len(fillers) - (2 if (index // 2) % 2 == 0 else 6)
            arms = {}
            for arm, expression in sources.items():
                value = int(safe_arithmetic_eval(expression))
                assert value == (answer if arm == "repeat" else lure)
                lines = fillers.copy()
                lines.insert(source_index, (expression, value))
                prefix, spans = "Working calculations:\n", []
                for lhs, rhs in lines:
                    prefix += f"{lhs} = "
                    spans.append([len(prefix), len(prefix) + len(str(rhs))])
                    prefix += f"{rhs}\n"
                prefix += f"Now compute the requested subtotal:\n{a} {op} {b} = "
                arms[arm] = {
                    "prefix": prefix,
                    "source_span": spans[source_index],
                    "control_span": spans[source_index + (1 if index % 2 == 0 else -1)],
                }
            rows.append(
                {
                    "id": f"interference-{seed}-{split}-{index:04d}",
                    "split": split,
                    "operator": op,
                    "answer": answer,
                    "lure": lure,
                    "final_answer": 2 * answer + 7,
                    "question": f"Compute 2 * ({a} {op} {b}) + 7 using ordinary arithmetic. "
                    "Continue the working calculations. Complete the subtotal line with "
                    "its integer value, then compute the final answer. Finish with a separate line "
                    "Answer: <the final integer>.",
                    "arms": arms,
                }
            )
    return rows


def token_contract(tokenizer, row: dict, arm: str) -> dict:
    """Use the model's chat template once, then continue an assistant prefix."""
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["question"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    context = chat + row["arms"][arm]["prefix"]
    encoded = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    result = {"input_ids": ids, "context": context}
    for name in ("source", "control"):
        start, end = row["arms"][arm][f"{name}_span"]
        span = token_range_for_chars(offsets, len(chat) + start, len(chat) + end)
        if span is None:
            raise ValueError(f"unmapped {name} span")
        positions = list(range(span[0], span[1] + 1))
        if (
            tokenizer.decode([ids[p] for p in positions]).strip()
            != context[len(chat) + start : len(chat) + end]
        ):
            raise ValueError(f"{name} tokens include text outside the number")
        result[f"{name}_positions"] = positions
    for name in ("answer", "lure"):
        continued = tokenizer(context + str(row[name]), add_special_tokens=False)[
            "input_ids"
        ]
        if continued[: len(ids)] != ids or len(continued) <= len(ids):
            raise ValueError("numeric continuation changes prefix tokenization")
        result[f"{name}_ids"] = continued[len(ids) :]
    if result["answer_ids"][0] == result["lure_ids"][0]:
        raise ValueError("head screen requires distinct first answer tokens")
    if len(result["source_positions"]) != len(result["control_positions"]):
        raise ValueError("source and control token counts differ")
    return result


@contextmanager
def block_edges(model, specs: list[dict], positions: list[int], query_start: int):
    """Block selected head-to-source edges; keep HF's causal mask and softmax."""
    if positions and (min(positions) < 0 or max(positions) >= query_start):
        raise ValueError("source positions must precede the intervention query")
    by_layer = {}
    for spec in specs:
        by_layer.setdefault(int(spec["layer"]), []).append(spec)
    handles = []
    try:
        for index, selected in by_layer.items():
            attention = get_decoder_layers(model)[index].self_attn

            def patch(_module, args, kwargs, selected=selected):
                hidden = (
                    kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                )
                batch, length = hidden.shape[:2]
                mask = kwargs.get("attention_mask")
                if (
                    mask is None
                    or mask.ndim != 4
                    or mask.shape[-2:] != (length, length)
                ):
                    raise ValueError(
                        "edge blocking requires an eager, uncached causal mask"
                    )
                if mask.dtype == torch.bool:
                    raise ValueError("expected an additive attention mask")
                mask = mask.expand(
                    batch, model.config.num_attention_heads, length, length
                ).clone()
                for spec in selected:
                    batch_index = spec.get("batch_index", slice(None))
                    mask[
                        batch_index, int(spec["head"]), query_start:, positions
                    ] = -torch.inf
                return args, {**kwargs, "attention_mask": mask}

            handles.append(attention.register_forward_pre_hook(patch, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def screen_case(model, tokenizer, row: dict, *, head_batch_size: int) -> dict:
    """Measure every head's causal source-edge effect, without failure filtering."""
    contract = token_contract(tokenizer, row, "operator_lure")
    ids = torch.tensor([contract["input_ids"]], device=get_input_device(model))
    good, bad = contract["answer_ids"][0], contract["lure_ids"][0]
    base = forward_logits_ids(model, contract["input_ids"])
    margin = float((base[good] - base[bad]).item())
    head_count = model.config.num_attention_heads
    scores = np.empty((len(get_decoder_layers(model)), head_count))
    for layer in range(scores.shape[0]):
        for start in range(0, head_count, head_batch_size):
            heads = list(range(start, min(start + head_batch_size, head_count)))
            specs = [
                {"layer": layer, "head": head, "batch_index": i}
                for i, head in enumerate(heads)
            ]
            with (
                block_edges(
                    model, specs, contract["source_positions"], ids.shape[1] - 1
                ),
                torch.inference_mode(),
            ):
                logits = (
                    model(input_ids=ids.repeat(len(heads), 1), use_cache=False)
                    .logits[:, -1]
                    .float()
                )
            scores[layer, heads] = (
                logits[:, good] - logits[:, bad]
            ).cpu().numpy() - margin
    return {
        "id": row["id"],
        "original_logit_difference": margin,
        "scores": scores.tolist(),
    }


def freeze_heads(rows: list[dict], *, count: int, seed: int) -> dict:
    # The shared selector's unit denominator gives an unnormalized mean effect.
    return select_heads(
        [
            {**row, "clean_logit_difference": row["original_logit_difference"] + 1}
            for row in rows
        ],
        count=count,
        seed=seed,
    )


def score_continuation(text: str, row: dict) -> dict:
    """Score the full generated subtotal and final answer, not forced digits."""
    first = re.match(r"\s*(-?\d+)[ \t]*\.?[ \t]*(?:\n|$)", text)
    final = (
        re.fullmatch(
            r"\s*Answer:\s*(-?\d+)\s*\.?\s*", text.strip().splitlines()[-1], re.I
        )
        if text.strip()
        else None
    )
    subtotal = int(first[1]) if first else None
    answer = int(final[1]) if final else None
    return {
        "subtotal": subtotal,
        "answer": answer,
        "subtotal_parsed": subtotal is not None,
        "final_parsed": answer is not None,
        "subtotal_correct": subtotal == row["answer"],
        "lure_followed": subtotal == row["lure"],
        "final_correct": answer == row["final_answer"],
    }


def measure_case(
    model,
    tokenizer,
    row: dict,
    arm: str,
    condition: str,
    heads: dict,
    *,
    max_new_tokens: int,
    eos_token_id,
) -> dict:
    contract = token_contract(tokenizer, row, arm)
    ids = torch.tensor([contract["input_ids"]], device=get_input_device(model))
    selected = (
        heads["layer_matched_controls"]
        if condition == "control_heads"
        else heads["selected"]
    )
    if condition == "baseline":
        selected = []
    positions = (
        contract["control_positions"]
        if condition == "control_source"
        else contract["source_positions"]
    )
    with (
        block_edges(model, selected, positions, ids.shape[1] - 1),
        torch.inference_mode(),
    ):
        first = model(input_ids=ids, use_cache=False, output_attentions=True)
        probabilities = first.logits[0, -1].float().softmax(-1)
        source_mass = [
            float(
                first.attentions[h["layer"]][
                    0, h["head"], -1, contract["source_positions"]
                ]
                .sum()
                .item()
            )
            for h in heads["selected"]
        ]
        evidence = {
            "first_token_answer_probability": float(
                probabilities[contract["answer_ids"][0]].item()
            ),
            "first_token_lure_probability": float(
                probabilities[contract["lure_ids"][0]].item()
            ),
            "selected_head_source_attention": source_mass,
        }
        del first
        output = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            do_sample=False,
            use_cache=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    tokens = output[0, ids.shape[1] :].tolist()
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return {
        "id": row["id"],
        "arm": arm,
        "condition": condition,
        "operator": row["operator"],
        "input_ids": contract["input_ids"],
        "generated_token_ids": tokens,
        "produced_text": text,
        "capped": len(tokens) >= max_new_tokens,
        **evidence,
        **score_continuation(text, row),
    }


def summarize(rows: list[dict]) -> dict:
    from scipy.stats import binomtest

    keys = [(r["id"], r["arm"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate measurements")
    cells = {}
    for row in rows:
        cells.setdefault(f"{row['arm']}:{row['condition']}", []).append(row)
    counts = {
        key: {
            "n": len(cell),
            **{
                metric: sum(r[metric] for r in cell)
                for metric in (
                    "subtotal_correct",
                    "lure_followed",
                    "final_correct",
                    "subtotal_parsed",
                    "final_parsed",
                    "capped",
                )
            },
        }
        for key, cell in cells.items()
    }
    paired = {}
    for arm in sorted({r["arm"] for r in rows}):
        base = {r["id"]: r for r in cells.get(f"{arm}:baseline", [])}
        for condition in ("source", "control_heads", "control_source"):
            changed = cells.get(f"{arm}:{condition}", [])
            if {r["id"] for r in changed} != set(base):
                continue
            diffs = [
                int(r["final_correct"]) - int(base[r["id"]]["final_correct"])
                for r in changed
            ]
            wins, losses = diffs.count(1), diffs.count(-1)
            paired[f"{arm}:{condition}-baseline"] = {
                "n": len(diffs),
                "wins": wins,
                "losses": losses,
                "accuracy_difference": float(np.mean(diffs)),
                "mcnemar_p_unadjusted": binomtest(wins, wins + losses).pvalue
                if wins + losses
                else 1.0,
            }
    return {"status": "mechanism_discovery_pilot", "counts": counts, "paired": paired}
