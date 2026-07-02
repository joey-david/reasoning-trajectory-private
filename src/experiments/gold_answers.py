"""Capture teacher-forced gold-answer representations for MI estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.generation_pipeline import capture_selected_activations
from src.runtime.artifact_store import (
    append_jsonl,
    sanitize_filename,
    save_hidden_states_npz,
    write_json,
)
from src.runtime.data import load_samples


def completed_gold_answers(path: Path) -> set[str]:
    """Return sample IDs already present in a gold-answer manifest."""
    if not path.exists():
        return set()
    return {
        str(row["sample_id"])
        for row in load_samples(path.resolve())
        if row.get("hidden_states_file")
    }


def capture_gold_answer(
    *,
    run_path: Path,
    model: Any,
    tokenizer: Any,
    sample: dict[str, Any],
    layers: list[int],
    storage_dtype: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Teacher-force one complete gold solution and persist selected states."""
    sample_id = str(sample.get("id") or sample.get("problem_id"))
    answer = sample.get("gold_answer")
    if answer is None or not str(answer).strip():
        raise ValueError(f"Sample {sample_id} has no gold answer")
    answer_text = str(answer)
    encoded = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    answer_ids = [int(token_id) for token_id in encoded["input_ids"]]
    if not answer_ids:
        raise ValueError(f"Gold answer for {sample_id} tokenized to zero tokens")
    if len(answer_ids) > max_tokens:
        raise ValueError(
            f"Gold answer for {sample_id} has {len(answer_ids)} tokens, "
            f"above configured maximum {max_tokens}"
        )
    prefix_id = tokenizer.bos_token_id
    if prefix_id is None:
        prefix_id = tokenizer.eos_token_id
    if prefix_id is None:
        raise ValueError("Tokenizer needs a BOS or EOS token for causal alignment")

    hidden_states, _components = capture_selected_activations(
        model=model,
        full_seq_ids=[int(prefix_id), *answer_ids],
        prompt_len=1,
        num_generated=len(answer_ids),
        layer_indices=layers,
        components=[],
    )
    hidden_path = (
        run_path
        / "gold_answers"
        / "hidden_states"
        / f"{sanitize_filename(sample_id)}.npz"
    )
    save_hidden_states_npz(
        path=hidden_path,
        hidden_states=hidden_states,
        layer_indices=layers,
        storage_dtype=storage_dtype,
    )
    record = {
        "sample_id": sample_id,
        "gold_answer": answer_text,
        "token_ids": answer_ids,
        "token_count": len(answer_ids),
        "layers": layers,
        "hidden_states_file": hidden_path.relative_to(run_path).as_posix(),
        "representation": (
            "causal states aligned to every teacher-forced gold-answer token"
        ),
    }
    append_jsonl(run_path / "gold_answers" / "manifest.jsonl", record)
    return record


def write_gold_answer_metadata(
    run_path: Path,
    *,
    model_name: str,
    layers: list[int],
    storage_dtype: str,
) -> None:
    """Write the stable gold-answer artifact contract."""
    path = run_path / "gold_answers" / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_name": model_name,
        "layers": layers,
        "activation_storage_dtype": storage_dtype,
        "alignment": (
            "state zero predicts gold token zero from a single BOS/EOS prefix"
        ),
        "recommended_summary": "mean over gold-answer token states",
    }
    write_json(path, payload)
