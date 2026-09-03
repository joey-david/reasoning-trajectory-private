"""Causal head screen and held-out routing-horizon test."""

from __future__ import annotations

from collections.abc import Iterable
import math
import random
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.depth_relief.benchmark import candidate_token_ids
from src.experiments.layer_replications.symbolic import causal_mediation
from src.experiments.state_routing_horizon import (
    NAMES,
    _Text,
    _apply,
    _free_generation,
    _token_positions,
)
from src.models.introspection import get_decoder_layers, get_input_device


LENGTHS = {"screen": (8,), "development": (4, 6, 8), "test": (12, 16, 20)}


def _render_trace(
    *, initial: dict[str, int], events: list[dict[str, Any]], target: str
) -> tuple[dict[str, Any], int]:
    text = _Text()
    text.add("Four players keep scores modulo 10; after 9 comes 0.\n")
    text.add("Starting scores: ")
    text.add("; ".join(f"{name} has {initial[name]}" for name in NAMES))
    text.add(".\nEvents, in order:\n")
    for index, event in enumerate(events):
        text.add(f"{index + 1}. {event['name']} must {event['operation']} ")
        text.mark(f"amount_{index}", int(event["amount"]))
        text.add(".\n")
    text.add(
        f"Work through all events in order. What is {target}'s final score? "
        "Use one short line per event, then end with Answer=<one digit>.\nSolution:\n"
    )
    states = dict(initial)
    writes = []
    for index, event in enumerate(events):
        name = str(event["name"])
        before = states[name]
        states[name] = _apply(before, str(event["operation"]), int(event["amount"]))
        symbol = "+" if event["operation"] == "add" else "-"
        text.add(f"{index + 1}. {name}: {before} {symbol} {event['amount']} mod 10 = ")
        key = f"write_{index}"
        text.mark(key, states[name])
        writes.append({"event": index, "name": name, "span_key": key})
        text.add(".\n")
    text.add(f"Therefore, {target}'s final score is\nAnswer=")
    rendered = text.finish()
    rendered["writes"] = writes
    return rendered, states[target]


def _question(teacher: dict[str, Any]) -> str:
    return str(teacher["text"]).split("Solution:\n", 1)[0] + "Solution:\n"


def build_head_cases(
    *, screen_count: int, development_count: int, test_count: int, seed: int
) -> list[dict[str, Any]]:
    """Build a locked short-fit/long-test bank with two early target writes."""
    rng = random.Random(seed)
    counts = {
        "screen": screen_count,
        "development": development_count,
        "test": test_count,
    }
    rows = []
    global_index = 0
    for split, count in counts.items():
        for split_index in range(count):
            event_count = LENGTHS[split][split_index % len(LENGTHS[split])]
            target = NAMES[global_index % len(NAMES)]
            initial = {name: rng.randrange(10) for name in NAMES}
            distractors = [name for name in NAMES if name != target]
            events = [
                {
                    "name": target,
                    "operation": rng.choice(("add", "subtract")),
                    "amount": rng.randrange(1, 9),
                },
                {
                    "name": rng.choice(distractors),
                    "operation": rng.choice(("add", "subtract")),
                    "amount": rng.randrange(1, 9),
                },
                {
                    "name": target,
                    "operation": rng.choice(("add", "subtract")),
                    "amount": rng.randrange(1, 9),
                },
            ]
            while len(events) < event_count:
                events.append(
                    {
                        "name": rng.choice(distractors),
                        "operation": rng.choice(("add", "subtract")),
                        "amount": rng.randrange(1, 9),
                    }
                )
            clean, clean_answer = _render_trace(
                initial=initial, events=events, target=target
            )
            corrupt_events = [dict(event) for event in events]
            old_amount = int(corrupt_events[2]["amount"])
            corrupt_events[2]["amount"] = old_amount % 8 + 1
            corrupt, corrupt_answer = _render_trace(
                initial=initial, events=corrupt_events, target=target
            )
            if clean_answer == corrupt_answer:
                raise AssertionError("counterfactual must change the answer")
            rows.append(
                {
                    "schema_version": 1,
                    "id": f"routing_head_{global_index:05d}",
                    "split": split,
                    "event_count": event_count,
                    "target": target,
                    "clean": clean,
                    "corrupt": corrupt,
                    "question": _question(clean),
                    "clean_answer": clean_answer,
                    "corrupt_answer": corrupt_answer,
                }
            )
            global_index += 1
    return rows


def validate_head_cases(
    rows: list[dict[str, Any]], expected: dict[str, int]
) -> dict[str, Any]:
    ids = [str(row["id"]) for row in rows]
    counts = {split: sum(row["split"] == split for row in rows) for split in expected}
    checks = {
        "unique_ids": len(ids) == len(set(ids)),
        "split_counts": counts == expected,
        "aligned_pairs": all(
            len(row["clean"]["text"]) == len(row["corrupt"]["text"])
            and row["clean"]["spans"] == row["corrupt"]["spans"]
            for row in rows
        ),
        "two_target_writes": all(
            sum(write["name"] == row["target"] for write in row["clean"]["writes"]) == 2
            for row in rows
        ),
        "long_test": min(row["event_count"] for row in rows if row["split"] == "test")
        > max(row["event_count"] for row in rows if row["split"] == "development"),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid head-routing data: {checks}")
    return {"schema_version": 1, "case_count": len(rows), "checks": checks}


def token_contract(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    clean = str(row["clean"]["text"])
    corrupt = str(row["corrupt"]["text"])
    clean_ids = tokenizer.encode(clean, add_special_tokens=False)
    corrupt_ids = tokenizer.encode(corrupt, add_special_tokens=False)
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("clean and corrupt token lengths differ")
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


def screen_case(
    *, model: Any, tokenizer: Any, row: dict[str, Any], head_batch_size: int
) -> dict[str, Any]:
    """Reuse the complete batched CMA screen at the final read token."""
    contract = token_contract(tokenizer, row)
    clean_id = int(contract["candidate_ids"][int(row["clean_answer"])])
    corrupt_id = int(contract["candidate_ids"][int(row["corrupt_answer"])])
    adapter = {
        "id": row["id"],
        "context_type": row["split"],
        "base_rule": "state_routing",
        "patch_positions": [],
        "token_count": contract["token_count"],
        "donor_prompt": row["clean"]["text"],
        "target_prompt": row["corrupt"]["text"],
        "target_answer_id": corrupt_id,
        "causal_answer_id": clean_id,
    }
    result = causal_mediation(
        model,
        tokenizer,
        adapter,
        mechanism="retrieval",
        head_batch_size=head_batch_size,
    )
    clean_ids = tokenizer(
        row["clean"]["text"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(get_input_device(model))
    with torch.inference_mode():
        clean_logits = model(input_ids=clean_ids, use_cache=False).logits[0, -1]
    clean_difference = float((clean_logits[clean_id] - clean_logits[corrupt_id]).item())
    result.update(
        {
            "schema_version": 1,
            "event_count": row["event_count"],
            "clean_logit_difference": clean_difference,
            "eligible": clean_difference > 0
            and float(result["original_logit_difference"]) < 0,
        }
    )
    return result


def select_heads(
    rows: list[dict[str, Any]], *, count: int, seed: int
) -> dict[str, Any]:
    """Freeze the heads with the largest mean clean-to-corrupt causal effect."""
    if not rows:
        raise ValueError("cannot select heads without screen rows")
    eligible = [row for row in rows if row.get("eligible", True)]
    if not eligible:
        raise ValueError("no eligible clean/corrupt screen rows")
    scores = np.asarray(
        [
            np.asarray(row["scores"], dtype=np.float64)
            / max(
                float(row.get("clean_logit_difference", 1.0))
                - float(row.get("original_logit_difference", 0.0)),
                1e-6,
            )
            for row in eligible
        ]
    )
    means = scores.mean(axis=0)
    ranked = sorted(
        (
            (float(means[layer, head]), layer, head)
            for layer, head in np.ndindex(means.shape)
        ),
        reverse=True,
    )
    selected = []
    per_layer: dict[int, int] = {}
    for effect, layer, head in ranked:
        if per_layer.get(layer, 0) >= means.shape[1] // 2:
            continue
        selected.append({"layer": layer, "head": head, "mean_effect": effect})
        per_layer[layer] = per_layer.get(layer, 0) + 1
        if len(selected) == count:
            break
    rng = random.Random(seed)
    selected_keys = {(row["layer"], row["head"]) for row in selected}
    controls = []
    for row in selected:
        candidates = [
            head
            for head in range(means.shape[1])
            if (row["layer"], head) not in selected_keys
            and all(
                control["layer"] != row["layer"] or control["head"] != head
                for control in controls
            )
        ]
        if not candidates:
            raise ValueError("not enough layer-matched control heads")
        controls.append({"layer": row["layer"], "head": rng.choice(candidates)})
    return {
        "schema_version": 1,
        "screen_cases": len(rows),
        "eligible_screen_cases": len(eligible),
        "matrix_shape": list(means.shape),
        "selected": selected,
        "layer_matched_controls": controls,
    }


def _attention_value_metrics(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    heads: list[dict[str, int]],
) -> tuple[dict[str, float], np.ndarray]:
    contract = token_contract(tokenizer, row)
    text = str(row["clean"]["text"])
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    encoded = {key: value.to(get_input_device(model)) for key, value in encoded.items()}
    layers = get_decoder_layers(model)
    values: list[torch.Tensor | None] = [None] * len(layers)
    handles = []
    for index, layer in enumerate(layers):
        projection = layer.self_attn.v_proj

        def capture(
            _module: Any,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
            layer_index: int = index,
        ) -> None:
            values[layer_index] = output.detach()

        handles.append(projection.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            output = model(
                **encoded, use_cache=False, output_attentions=True, return_dict=True
            )
    finally:
        for handle in handles:
            handle.remove()
    if output.attentions is None or any(value is None for value in values):
        raise RuntimeError("model did not expose eager attention and value outputs")
    write_positions = [
        int(contract["positions"][write["span_key"]][0])
        for write in row["clean"]["writes"]
    ]
    valid_index = max(
        i
        for i, write in enumerate(row["clean"]["writes"])
        if write["name"] == row["target"]
    )
    valid_position = write_positions[valid_index]
    wrong_positions = [
        position for i, position in enumerate(write_positions) if i != valid_index
    ]
    token_count = int(contract["token_count"])
    valid = 0.0
    wrong = np.zeros(len(wrong_positions), dtype=np.float64)
    support = 0.0
    raw_valid = 0.0
    entropies = []
    num_heads = int(model.config.num_attention_heads)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", num_heads))
    groups = num_heads // num_kv_heads
    for spec in heads:
        layer_index, head = int(spec["layer"]), int(spec["head"])
        attention = output.attentions[layer_index][0, head, -1].float().cpu().numpy()
        value = values[layer_index]
        assert value is not None
        head_width = value.shape[-1] // num_kv_heads
        kv_head = head // groups
        value_head = value[0, :, kv_head * head_width : (kv_head + 1) * head_width]
        norms = value_head.float().norm(dim=-1).cpu().numpy()
        weighted = attention * norms
        valid += float(weighted[valid_position])
        wrong += weighted[wrong_positions]
        support += float(weighted[:token_count].sum())
        raw_valid += float(attention[valid_position])
        positive = attention[attention > 0]
        entropies.append(float(-(positive * np.log(positive)).sum()))
    epsilon = 1e-12
    metrics = {
        "routing_margin": math.log(valid + epsilon)
        - math.log(float(wrong.max()) + epsilon),
        "valid_support_fraction": valid / max(support, epsilon),
        "raw_valid_attention": raw_valid / len(heads),
        "attention_entropy": float(np.mean(entropies)),
        "valid_contribution": valid,
        "strongest_wrong_contribution": float(wrong.max()),
        "source_distance": float(token_count - 1 - valid_position),
    }
    logits = output.logits[0, -1].float().cpu().numpy()
    return metrics, logits


def measure_case(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    head_spec: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    selected, controls = head_spec["selected"], head_spec["layer_matched_controls"]
    metrics, logits = _attention_value_metrics(model, tokenizer, row, selected)
    control_metrics, _ = _attention_value_metrics(model, tokenizer, row, controls)
    contract = token_contract(tokenizer, row)
    candidate_logits = np.asarray(
        [logits[int(contract["candidate_ids"][digit])] for digit in range(10)]
    )
    answer = int(row["clean_answer"])
    answer_margin = float(
        candidate_logits[answer] - np.max(np.delete(candidate_logits, answer))
    )
    generation = _free_generation(
        model, tokenizer, str(row["question"]), max_new_tokens
    )
    return {
        "schema_version": 1,
        "id": row["id"],
        "split": row["split"],
        "event_count": row["event_count"],
        "clean_answer": answer,
        "generation": generation,
        "generation_correct": generation["answer"] == answer,
        "answer_logit_margin": answer_margin,
        "routing": metrics,
        "control_routing": control_metrics,
    }


def _fit_score(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    source: str = "routing",
) -> dict[str, Any]:
    def features(rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                [
                    float(row[source][field])
                    if field in row[source]
                    else float(row[field])
                    for field in fields
                ]
                for row in rows
            ]
        )

    y_train = np.asarray([bool(row["generation_correct"]) for row in train], dtype=int)
    y_test = np.asarray([bool(row["generation_correct"]) for row in test], dtype=int)
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {
            "fields": list(fields),
            "source": source,
            "roc_auc": None,
            "brier": None,
        }
    model = make_pipeline(
        StandardScaler(), LogisticRegression(random_state=260530233, max_iter=1000)
    )
    model.fit(features(train), y_train)
    probability = model.predict_proba(features(test))[:, 1]
    return {
        "fields": list(fields),
        "source": source,
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "brier": float(brier_score_loss(y_test, probability)),
    }


def summarize_measurements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    development = [row for row in rows if row["split"] == "development"]
    test = [row for row in rows if row["split"] == "test"]
    if not development or not test:
        raise ValueError("both development and test measurements are required")
    routing = _fit_score(
        development, test, ("routing_margin", "valid_support_fraction")
    )
    baselines = {
        "length": _fit_score(development, test, ("event_count",)),
        "confidence": _fit_score(development, test, ("answer_logit_margin",)),
        "entropy": _fit_score(development, test, ("attention_entropy",)),
        "layer_matched_heads": _fit_score(
            development,
            test,
            ("routing_margin", "valid_support_fraction"),
            source="control_routing",
        ),
    }
    baseline_aucs = [
        float(row["roc_auc"])
        for row in baselines.values()
        if row["roc_auc"] is not None
    ]
    auc_advantage = (
        float(routing["roc_auc"]) - max(baseline_aucs)
        if routing["roc_auc"] is not None and baseline_aucs
        else None
    )
    return {
        "schema_version": 1,
        "case_count": len(rows),
        "parsed_rate": sum(row["generation"]["parsed"] for row in rows) / len(rows),
        "accuracy_by_split": {
            split: sum(row["generation_correct"] for row in subset) / len(subset)
            for split, subset in (("development", development), ("test", test))
        },
        "routing_predictor": routing,
        "baselines": baselines,
        "auc_advantage_over_best_baseline": auc_advantage,
        "gate": {
            "auc_at_least_075": routing["roc_auc"] is not None
            and routing["roc_auc"] >= 0.75,
            "brier_at_most_018": routing["brier"] is not None
            and routing["brier"] <= 0.18,
            "auc_advantage_at_least_007": auc_advantage is not None
            and auc_advantage >= 0.07,
        },
    }


def select_shard(
    rows: Iterable[dict[str, Any]], *, split: str, shard: int, shards: int
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["split"] == split]
    return [row for index, row in enumerate(eligible) if index % shards == shard]
