"""Tensor, checkpoint, and adapter helpers for state-handoff training."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable

from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _collate(
    *, sequences: list[dict[str, Any]], tokenizer: Any, max_length: int, device: Any
) -> dict[str, Any]:
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Training tokenizer needs a pad token")
    input_ids = torch.full(
        (len(sequences), max_length), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for index, sequence in enumerate(sequences):
        length = len(sequence["input_ids"])
        if length > max_length:
            raise ValueError(f"Sequence exceeds configured max length {max_length}")
        input_ids[index, :length] = torch.tensor(
            sequence["input_ids"], dtype=torch.long, device=device
        )
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(
            sequence["labels"], dtype=torch.long, device=device
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mappings": [str(sequence["mapping"]) for sequence in sequences],
        "horizons": [int(sequence["history_steps"]) for sequence in sequences],
    }


def _per_sequence_loss(logits: Any, labels: Any) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    positions = labels.ne(-100).nonzero(as_tuple=False)
    if len(positions) != labels.shape[0]:
        raise ValueError("Every training sequence must expose exactly one target token")
    if not torch.equal(
        positions[:, 0], torch.arange(labels.shape[0], device=labels.device)
    ):
        raise ValueError("Training targets do not align one-per-sequence")
    prediction_positions = positions[:, 1] - 1
    if bool((prediction_positions < 0).any()):
        raise ValueError("A target token has no causal prefix")
    selected = logits[
        torch.arange(logits.shape[0], device=logits.device), prediction_positions
    ].float()
    targets = labels[positions[:, 0], positions[:, 1]]
    losses = functional.cross_entropy(selected, targets, reduction="none")
    return losses, selected.argmax(-1).eq(targets)


def _condition_loss(
    losses: Any, mappings: list[str], condition: str
) -> tuple[Any, Any | None, Any]:
    import torch

    state_mask = torch.tensor(
        [mapping == "state" for mapping in mappings],
        dtype=torch.bool,
        device=losses.device,
    )
    answer_mask = ~state_mask
    state_loss = losses[state_mask].mean() if bool(state_mask.any()) else None
    answer_loss = losses[answer_mask].mean()
    total = answer_loss if condition == "outcome_only" else state_loss + answer_loss
    return total, state_loss, answer_loss


def _pair_batches(
    pairs: list[list[dict[str, Any]]],
    *,
    pair_batch_size: int,
    epochs: int,
    seed: int,
) -> Iterable[list[list[dict[str, Any]]]]:
    for epoch in range(epochs):
        order = list(range(len(pairs)))
        random.Random(seed + epoch).shuffle(order)
        for start in range(0, len(order), pair_batch_size):
            yield [pairs[index] for index in order[start : start + pair_batch_size]]


def _evaluate_training_pairs(
    *,
    model: Any,
    tokenizer: Any,
    pairs: list[list[dict[str, Any]]],
    max_length: int,
    microbatch: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    device = model.get_input_embeddings().weight.device
    counts: defaultdict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    flat = [sequence for pair in pairs for sequence in pair]
    with torch.inference_mode():
        for start in range(0, len(flat), microbatch):
            sequences = flat[start : start + microbatch]
            batch = _collate(
                sequences=sequences,
                tokenizer=tokenizer,
                max_length=max_length,
                device=device,
            )
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            _, correct = _per_sequence_loss(output.logits, batch["labels"])
            for horizon, mapping, value in zip(
                batch["horizons"], batch["mappings"], correct.tolist()
            ):
                cell = counts[(horizon, mapping)]
                cell[0] += int(value)
                cell[1] += 1
    model.train()
    return {
        str(horizon): {
            mapping: values[0] / values[1]
            for (cell_horizon, mapping), values in sorted(counts.items())
            if cell_horizon == horizon
        }
        for horizon in sorted({key[0] for key in counts})
    }


def _save_checkpoint(
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    directory: Path,
    state: dict[str, Any],
) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter", safe_serialization=True)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        directory / "training_state.pt",
    )
    write_json(directory / "trainer_state.json", state)


def _load_resume_state(
    *, model: Any, optimizer: Any, scheduler: Any, checkpoint: Path
) -> dict[str, Any]:
    import torch

    payload = torch.load(
        checkpoint / "training_state.pt",
        map_location=model.get_input_embeddings().weight.device,
        weights_only=False,
    )
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        cuda_rng = []
        for state in payload["cuda_rng"]:
            state = state.detach().to(device="cpu")
            if state.dtype != torch.uint8:
                raise TypeError("Saved CUDA RNG state is not a byte tensor")
            cuda_rng.append(state)
        torch.cuda.set_rng_state_all(cuda_rng)
    return json.loads((checkpoint / "trainer_state.json").read_text())


def _base_and_adapter(
    *,
    run_path: Path,
    condition: str,
    checkpoint: Path | None,
    model: Any | None,
    tokenizer: Any | None,
) -> tuple[Any, Any]:
    config = load_config(run_path)
    if model is None or tokenizer is None:
        model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    initial_adapter = (
        config.get("state_handoff_training", {})
        .get("interfaces", {})
        .get("initial_adapters", {})
        .get(condition)
    )
    if checkpoint is not None:
        model = PeftModel.from_pretrained(
            model, checkpoint / "adapter", is_trainable=True
        )
    elif initial_adapter:
        model = PeftModel.from_pretrained(
            model, Path(str(initial_adapter)), is_trainable=True
        )
    else:
        lora = config.get("state_handoff_training", {}).get("lora", {})
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora.get("rank", 16)),
                lora_alpha=int(lora.get("alpha", 32)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=list(lora.get("target_modules", DEFAULT_TARGET_MODULES)),
                bias="none",
            ),
        )
    model.train()
    return model, tokenizer
