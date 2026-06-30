"""Teacher-force completed generations to capture additional activations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from src.experiments.common import balanced_generation_rows
from src.models.generation_pipeline import (
    capture_selected_activations,
    compute_timestep_artifacts,
    single_token_id,
)
from src.models.hf_loader import load_hf_model_and_tokenizer
from src.runtime.artifact_store import sanitize_filename, save_generation_output
from src.runtime.config import load_config
from src.runtime.generation_output import (
    HIDDEN_STATE_CONVENTION,
    CompleteGenerationOutput,
)
from src.runtime.paths import resolve_repo_path
from src.runtime.run_io import load_generation_index


def replay_capture_run(run_path: Path) -> None:
    """Capture configured layers/components without regenerating source text."""
    config = load_config(run_path)
    replay_cfg = config["replay"]
    capture_cfg = config["capture"]
    source_run = Path(replay_cfg["source_run"])
    rows = balanced_generation_rows(
        source_run,
        per_sample=int(replay_cfg.get("per_sample", 10)),
        require_hidden_states=False,
    )
    sample_ids_path = replay_cfg.get("sample_ids_path")
    if sample_ids_path:
        wanted = {
            str(row["id"]) for row in load_jsonl(resolve_repo_path(sample_ids_path))
        }
        rows = [row for row in rows if str(row["sample_id"]) in wanted]
    max_trajectories = int(replay_cfg.get("max_trajectories", 0))
    if max_trajectories > 0:
        rows = rows[:max_trajectories]

    model, tokenizer = load_hf_model_and_tokenizer(config["model"])
    layers = [int(layer) for layer in capture_cfg.get("layers", [-1])]
    components = [str(name) for name in capture_cfg.get("components", [])]
    storage_dtype = str(capture_cfg.get("activation_storage_dtype", "int8_scaled"))
    completed = load_generation_index(run_path)

    for row in tqdm(rows, desc="teacher-forced capture", unit="trace"):
        key = (
            str(row["sample_id"]),
            int(row["seed"]),
            float(row["temperature"]),
        )
        if key in completed:
            continue
        sample = load_source_sample(source_run, key[0])
        input_ids = [int(token) for token in sample["input_ids"]]
        generated_ids = [int(token) for token in row["generated_token_ids"]]
        hidden_states, component_states = capture_selected_activations(
            model=model,
            full_seq_ids=input_ids + generated_ids,
            prompt_len=len(input_ids),
            num_generated=len(generated_ids),
            layer_indices=layers,
            components=components,
        )
        timestep_artifacts = (
            compute_timestep_artifacts(
                model=model,
                tokenizer=tokenizer,
                hidden_states=hidden_states,
                generated_token_ids=generated_ids,
                prompt_len=len(input_ids),
                gold_token_id=single_token_id(tokenizer, sample.get("gold_answer")),
            )
            if capture_cfg.get("diagnostics", False)
            else []
        )
        output = CompleteGenerationOutput(
            sample_id=key[0],
            seed=key[1],
            temperature=key[2],
            model_name=str(config["model"]["name"]),
            layer_indices=layers,
            hidden_state_convention=HIDDEN_STATE_CONVENTION,
            prompt=str(sample.get("prompt", "")),
            input_ids=input_ids,
            generated_token_ids=generated_ids,
            dp1_idx=int(sample.get("dp1_idx", len(input_ids))),
            dp2_idx=row.get("dp2_idx"),
            reasoning_length=row.get("reasoning_length"),
            produced_text=str(row.get("produced_text", "")),
            produced_answer=row.get("produced_answer"),
            gold_answer=sample.get("gold_answer"),
            is_correct=row.get("is_correct"),
            timestep_artifacts=timestep_artifacts,
            hidden_states_file=None,
        )
        save_generation_output(
            run_path=run_path,
            output=output,
            hidden_states=hidden_states,
            component_states=component_states,
            storage_dtype=storage_dtype,
        )
        completed.add(key)


def load_source_sample(source_run: Path, sample_id: str) -> dict[str, Any]:
    path = (
        source_run / "generation" / "samples" / f"{sanitize_filename(sample_id)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing source sample record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
