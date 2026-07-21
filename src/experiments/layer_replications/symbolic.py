"""Yang et al. identity-rule causal-mediation replication."""

from __future__ import annotations

import json
import re
from pathlib import Path
import random
from typing import Any

import torch
import torch.nn.functional as F

from src.experiments.layer_replications.common import (
    read_jsonl,
    replication_dir,
    write_jsonl,
)
from src.models.hf_loader import load_hf_tokenizer
from src.models.introspection import get_decoder_layers, get_input_device
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


MECHANISMS = ("symbol_abstraction", "symbolic_induction", "retrieval")
CONTEXTS = ("abstract", "token")
RULES = ("ABA", "ABB")


def screen_path(run_path: Path) -> Path:
    """Return baseline task-performance rows used to admit CMA pairs."""
    return replication_dir(run_path) / "yang_symbolic/screen.jsonl"


def selection_path(run_path: Path) -> Path:
    """Return the fixed correct-pair selection used for causal patching."""
    return replication_dir(run_path) / "yang_symbolic/selection.jsonl"


def cma_path(run_path: Path) -> Path:
    """Return per-pair causal-mediation matrices."""
    return replication_dir(run_path) / "yang_symbolic/cma.jsonl"


def _decoded_word_tokens(tokenizer: Any) -> list[str]:
    """Reproduce the upstream letter-only vocabulary construction."""
    added = set(tokenizer.get_added_vocab())
    words = sorted(
        token
        for token in tokenizer.get_vocab()
        if token not in added and re.fullmatch(r"[a-zA-Z]+", token)
    )
    if len(words) < 64:
        raise RuntimeError(f"only {len(words)} stable letter-only tokens found")
    return words


def _answer_token_id(tokenizer: Any, prompt: str, answer: str) -> int:
    """Resolve the single next-token answer under the exact prompt prefix."""
    prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    filled = tokenizer(prompt + answer, add_special_tokens=False)["input_ids"]
    suffix = filled[len(prefix) :]
    if len(suffix) != 1:
        raise ValueError(f"answer {answer!r} is not one token after the prompt")
    return int(suffix[0])


def _render_rule(rule: str, first: str, second: str) -> tuple[str, str, str]:
    """Render one complete ABA/ABB identity-rule example."""
    if rule == "ABA":
        return first, second, first
    if rule == "ABB":
        return first, second, second
    raise ValueError(f"unknown identity rule: {rule}")


def _build_pair(
    tokenizer: Any,
    words: list[str],
    *,
    shots: int,
    context_type: str,
    base_rule: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Independently implement the paper's abstract/token context construction."""
    chosen = rng.sample(words, 2 * (shots + 1))
    donor_examples: list[str] = []
    target_examples: list[str] = []
    for index in range(shots):
        first, second = chosen[2 * index : 2 * index + 2]
        donor = _render_rule(base_rule, first, second)
        if context_type == "abstract":
            target = (donor[1], donor[0], donor[2])
        elif context_type == "token":
            target = donor
        else:
            raise ValueError(f"unknown context type: {context_type}")
        donor_examples.append("^".join(donor))
        target_examples.append("^".join(target))

    query_first, query_second = chosen[-2:]
    donor_query = f"{query_first}^{query_second}^"
    target_query = f"{query_second}^{query_first}^"
    donor_prompt = "\n".join([*donor_examples, donor_query])
    target_prompt = "\n".join([*target_examples, target_query])
    donor_index = 0 if base_rule == "ABA" else 1
    target_index = (1 - donor_index) if context_type == "abstract" else donor_index
    donor_answer = (query_first, query_second)[donor_index]
    target_answer = (query_second, query_first)[target_index]
    causal_answer = (
        (query_second, query_first)[donor_index]
        if context_type == "abstract"
        else donor_answer
    )
    expected_length = 6 * shots + 4
    donor_ids = tokenizer(donor_prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_prompt, add_special_tokens=False)["input_ids"]
    if len(donor_ids) != expected_length or len(target_ids) != expected_length:
        return None
    try:
        donor_answer_id = _answer_token_id(tokenizer, donor_prompt, donor_answer)
        target_answer_id = _answer_token_id(tokenizer, target_prompt, target_answer)
        causal_answer_id = _answer_token_id(tokenizer, target_prompt, causal_answer)
    except ValueError:
        return None
    patch_positions = (
        [6 * index + 4 for index in range(shots)]
        if context_type == "abstract"
        else [expected_length - 1]
    )
    return {
        "context_type": context_type,
        "base_rule": base_rule,
        "donor_prompt": donor_prompt,
        "target_prompt": target_prompt,
        "donor_answer": donor_answer,
        "target_answer": target_answer,
        "causal_answer": causal_answer,
        "donor_answer_id": donor_answer_id,
        "target_answer_id": target_answer_id,
        "causal_answer_id": causal_answer_id,
        "patch_positions": patch_positions,
        "token_count": expected_length,
    }


def prepare_dataset(
    run_path: Path, *, candidates_per_cell: int | None = None
) -> dict[str, Any]:
    """Build deterministic single-token identity-rule context pairs."""
    config = load_config(run_path)
    experiment = config["symbolic_mechanisms"]
    benchmark = experiment["benchmark"]
    tokenizer = load_hf_tokenizer(config["model"])
    words = _decoded_word_tokens(tokenizer)
    shots = int(benchmark["in_context_examples"])
    target = int(candidates_per_cell or benchmark["candidate_pairs_per_direction"])
    rng = random.Random(int(benchmark["seed"]))
    rows: list[dict[str, Any]] = []
    for context_type in CONTEXTS:
        for base_rule in RULES:
            accepted = 0
            attempts = 0
            while accepted < target:
                attempts += 1
                if attempts > target * 100:
                    raise RuntimeError(
                        f"could not construct {target} {context_type}/{base_rule} pairs"
                    )
                row = _build_pair(
                    tokenizer,
                    words,
                    shots=shots,
                    context_type=context_type,
                    base_rule=base_rule,
                    rng=rng,
                )
                if row is None:
                    continue
                row["id"] = f"{context_type}-{base_rule.lower()}-{accepted:04d}"
                rows.append(row)
                accepted += 1
    write_jsonl(run_path / "dataset.jsonl", rows)
    manifest = {
        "model": config["model"]["name"],
        "model_revision": config["model"].get("revision"),
        "in_context_examples": shots,
        "pairs": len(rows),
        "pairs_per_context_and_direction": target,
        "eligible_word_tokens": len(words),
        "vocabulary_filter": "upstream_letter_only",
    }
    write_json(
        replication_dir(run_path) / "yang_symbolic/dataset_manifest.json", manifest
    )
    return manifest


def load_pairs(run_path: Path) -> list[dict[str, Any]]:
    """Load prepared context pairs."""
    rows = read_jsonl(run_path / "dataset.jsonl")
    if not rows:
        raise FileNotFoundError(
            f"{run_path / 'dataset.jsonl'} is absent or empty; run prepare-symbolic"
        )
    return rows


def screen_pair(
    model: torch.nn.Module, tokenizer: Any, row: dict[str, Any]
) -> dict[str, Any]:
    """Check that both clean contexts predict their rule-consistent answer."""
    device = get_input_device(model)  # type: ignore[arg-type]
    prompts = [row["donor_prompt"], row["target_prompt"]]
    encoded = tokenizer(
        prompts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits
    last = encoded["attention_mask"].sum(-1) - 1
    final = logits[torch.arange(2, device=device), last].float()
    expected = torch.tensor(
        [row["donor_answer_id"], row["target_answer_id"]], device=device
    )
    probabilities = F.softmax(final, dim=-1)[torch.arange(2, device=device), expected]
    correct = final.argmax(-1) == expected
    return {
        "id": row["id"],
        "context_type": row["context_type"],
        "base_rule": row["base_rule"],
        "donor_correct": bool(correct[0].item()),
        "target_correct": bool(correct[1].item()),
        "both_correct": bool(correct.all().item()),
        "donor_answer_probability": float(probabilities[0].item()),
        "target_answer_probability": float(probabilities[1].item()),
    }


def select_valid_pairs(run_path: Path) -> dict[str, Any]:
    """Freeze the first paper-sized set of clean-correct pairs per cell."""
    config = load_config(run_path)
    needed = int(config["symbolic_mechanisms"]["valid_pairs_per_direction"])
    pairs = {str(row["id"]): row for row in load_pairs(run_path)}
    screened = read_jsonl(screen_path(run_path))
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for context_type in CONTEXTS:
        for base_rule in RULES:
            cell = [
                row
                for row in screened
                if row["context_type"] == context_type
                and row["base_rule"] == base_rule
                and row["both_correct"]
            ]
            cell.sort(key=lambda row: str(row["id"]))
            key = f"{context_type}/{base_rule}"
            counts[key] = len(cell)
            if len(cell) < needed:
                raise RuntimeError(
                    f"symbolic screen found {len(cell)}/{needed} clean-correct {key} pairs"
                )
            selected.extend(pairs[str(row["id"])] for row in cell[:needed])
    write_jsonl(selection_path(run_path), selected)
    report = {
        "selected_per_cell": needed,
        "available": counts,
        "selected": len(selected),
    }
    write_json(
        replication_dir(run_path) / "yang_symbolic/selection_report.json", report
    )
    return report


def cma_task_key(pair_id: str, mechanism: str) -> str:
    """Build a stable pair/mechanism key."""
    return f"{pair_id}:{mechanism}"


def cma_tasks(run_path: Path) -> list[dict[str, Any]]:
    """Expand the fixed selection into the three paper mechanisms."""
    selected = read_jsonl(selection_path(run_path))
    if not selected:
        raise FileNotFoundError("symbolic selection is absent; run select-symbolic")
    tasks = []
    for index, row in enumerate(selected):
        mechanisms = (
            ("symbol_abstraction", "symbolic_induction")
            if row["context_type"] == "abstract"
            else ("retrieval",)
        )
        tasks.extend(
            {"pair_index": index, "mechanism": mechanism} for mechanism in mechanisms
        )
    return tasks


def _attention_output_projection(layer: torch.nn.Module) -> torch.nn.Module:
    """Resolve the pre-projection concatenated head output boundary."""
    attention = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
    projection = getattr(attention, "o_proj", None) or getattr(
        attention, "out_proj", None
    )
    if projection is None:
        raise TypeError(
            f"could not find attention output projection on {type(layer).__name__}"
        )
    return projection


def _capture_head_outputs(
    model: torch.nn.Module, input_ids: torch.Tensor, positions: list[int]
) -> list[torch.Tensor]:
    """Capture every layer's concatenated per-head values before output projection."""
    captures: list[torch.Tensor | None] = [None] * len(get_decoder_layers(model))  # type: ignore[arg-type]
    handles = []
    for index, layer in enumerate(get_decoder_layers(model)):  # type: ignore[arg-type]
        projection = _attention_output_projection(layer)

        def capture(
            _module: Any, inputs: tuple[torch.Tensor, ...], layer_index: int = index
        ) -> None:
            captures[layer_index] = inputs[0][:, positions].detach().clone()

        handles.append(projection.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in captures):
        raise RuntimeError("not every decoder layer exposed attention head outputs")
    return [value for value in captures if value is not None]


def causal_mediation(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    mechanism: str,
    head_batch_size: int,
) -> dict[str, Any]:
    """Compute one pair's complete layer-by-head CMA matrix."""
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism: {mechanism}")
    positions = (
        [int(value) for value in row["patch_positions"]]
        if mechanism == "symbol_abstraction"
        else [int(row["token_count"]) - 1]
    )
    device = get_input_device(model)  # type: ignore[arg-type]
    donor_ids = tokenizer(
        row["donor_prompt"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    target_ids = tokenizer(
        row["target_prompt"], return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    if donor_ids.shape != target_ids.shape:
        raise ValueError("donor and target contexts must have identical token shapes")

    with torch.inference_mode():
        target_logits = (
            model(input_ids=target_ids, use_cache=False).logits[0, -1].float()
        )
    original_id = int(row["target_answer_id"])
    causal_id = int(row["causal_answer_id"])
    original_difference = target_logits[causal_id] - target_logits[original_id]
    donor = _capture_head_outputs(model, donor_ids, positions)
    layers = get_decoder_layers(model)  # type: ignore[arg-type]
    head_count = int(getattr(model.config, "num_attention_heads"))
    hidden = donor[0].shape[-1]
    if hidden % head_count:
        raise ValueError(
            f"attention width {hidden} is not divisible by {head_count} heads"
        )
    head_width = hidden // head_count
    scores = torch.empty((len(layers), head_count), dtype=torch.float32)

    for layer_index, layer in enumerate(layers):
        projection = _attention_output_projection(layer)
        for start in range(0, head_count, head_batch_size):
            heads = list(range(start, min(start + head_batch_size, head_count)))
            batch_ids = target_ids.repeat(len(heads), 1)

            def patch(
                _module: Any, inputs: tuple[torch.Tensor, ...]
            ) -> tuple[Any, ...]:
                values = inputs[0].clone()
                for batch_index, head in enumerate(heads):
                    head_slice = slice(head * head_width, (head + 1) * head_width)
                    values[batch_index, positions, head_slice] = donor[layer_index][
                        0, :, head_slice
                    ]
                return (values, *inputs[1:])

            handle = projection.register_forward_pre_hook(patch)
            try:
                with torch.inference_mode():
                    logits = (
                        model(input_ids=batch_ids, use_cache=False)
                        .logits[:, -1]
                        .float()
                    )
            finally:
                handle.remove()
            differences = logits[:, causal_id] - logits[:, original_id]
            scores[layer_index, heads] = (differences - original_difference).cpu()

    return {
        "key": cma_task_key(str(row["id"]), mechanism),
        "id": row["id"],
        "context_type": row["context_type"],
        "base_rule": row["base_rule"],
        "mechanism": mechanism,
        "patch_positions": positions,
        "original_logit_difference": float(original_difference.item()),
        "scores": scores.tolist(),
    }


def validate(run_path: Path, *, require_dataset: bool = True) -> dict[str, Any]:
    """Validate the model and exact identity-rule protocol contract."""
    config = load_config(run_path)
    experiment = config["symbolic_mechanisms"]
    model = config["model"]
    checks = {
        "paper_shot_count": int(experiment["benchmark"]["in_context_examples"]) == 10,
        "paper_valid_pairs": int(experiment["valid_pairs_per_direction"]) == 100,
        "paper_permutations": int(experiment["permutation_trials"]) == 5000,
        "qwen_cross_model_replication": model["name"] == "Qwen/Qwen2.5-7B",
        "pinned_model_revision": bool(model.get("revision")),
        "head_count_declared": int(model["attention_heads"]) > 0,
    }
    if require_dataset:
        rows = load_pairs(run_path)
        manifest_path = (
            replication_dir(run_path) / "yang_symbolic/dataset_manifest.json"
        )
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        expected = 4 * int(experiment["benchmark"]["candidate_pairs_per_direction"])
        checks["dataset_cell_count"] = len(rows) == expected
        checks["matched_token_lengths"] = all(
            len(row["patch_positions"]) in {1, 10} and int(row["token_count"]) == 64
            for row in rows
        )
        checks["upstream_vocabulary_filter"] = (
            manifest.get("vocabulary_filter") == "upstream_letter_only"
            and int(manifest.get("eligible_word_tokens", 0)) == 27_376
        )
    if not all(checks.values()):
        raise ValueError(f"invalid symbolic replication: {checks}")
    return {"checks": checks, "valid": True}
