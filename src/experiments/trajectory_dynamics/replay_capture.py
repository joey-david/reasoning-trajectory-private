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
from src.runtime.artifact_store import (
    artifact_stem,
    write_json,
)
from src.runtime.config import load_config
from src.runtime.generation_output import (
    HIDDEN_STATE_CONVENTION,
    CompleteGenerationOutput,
)
from src.runtime.paths import resolve_repo_path
from src.runtime.run_io import load_generation_index


def replay_capture_run(run_path: Path) -> None:
    """Capture configured layers/components without regenerating source text.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        None.
    """
    config = load_config(run_path)
    replay_cfg = config["replay"]
    capture_cfg = config["capture"]
    source_run = Path(replay_cfg["source_run"])
    rows = selected_replay_rows(replay_cfg)

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
    """Load one per-sample generation record from a source run.

    Args:
        source_run: Source run directory containing the original artifacts.
        sample_id: Stable sample identifier.

    Returns:
        The resulting keyed records or metrics.
    """
    path = (
        source_run / "generation" / "samples" / f"{sanitize_filename(sample_id)}.json"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing source sample record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL records from disk.

    Args:
        path: Filesystem path to read from or write to.

    Returns:
        The resulting ordered records or values.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def selected_replay_rows(replay_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve the exact source rows used by generation and index repair.

    Args:
        replay_cfg: Replay selection and source-run configuration.

    Returns:
        The resulting ordered records or values.
    """
    rows = balanced_generation_rows(
        Path(replay_cfg["source_run"]),
        per_sample=int(replay_cfg.get("per_sample", 10)),
        require_hidden_states=False,
    )
    sample_ids_path = replay_cfg.get("sample_ids_path")
    if sample_ids_path:
        wanted = {
            str(row["id"]) for row in load_jsonl(resolve_repo_path(sample_ids_path))
        }
        rows = [row for row in rows if str(row["sample_id"]) in wanted]
    pair_manifest_path = replay_cfg.get("pair_manifest_path")
    if pair_manifest_path:
        pairs = load_jsonl(resolve_repo_path(pair_manifest_path))
        wanted_trajectories = {
            (str(pair[side]["sample_id"]), int(pair[side]["seed"]))
            for pair in pairs
            for side in ("donor", "target")
        }
        rows = [
            row
            for row in rows
            if (str(row["sample_id"]), int(row["seed"])) in wanted_trajectories
        ]
        found = {(str(row["sample_id"]), int(row["seed"])) for row in rows}
        missing = sorted(wanted_trajectories - found)
        if missing:
            raise ValueError(
                f"Pair manifest references {len(missing)} unavailable source traces: "
                f"{missing[:5]}"
            )
    max_trajectories = int(replay_cfg.get("max_trajectories", 0))
    return rows[:max_trajectories] if max_trajectories > 0 else rows


def rebuild_replay_index(run_path: Path) -> int:
    """Rebuild replay JSON metadata from source rows and completed NPZ files.

    Args:
        run_path: Run directory containing the configuration and artifacts.

    Returns:
        The computed index, count, or status code.
    """
    config = load_config(run_path)
    replay_cfg = config["replay"]
    capture_cfg = config["capture"]
    source_run = Path(replay_cfg["source_run"])
    layers = [int(layer) for layer in capture_cfg.get("layers", [-1])]
    components = [str(name) for name in capture_cfg.get("components", [])]
    generation_dir = run_path / "generation"
    hidden_dir = generation_dir / "hidden_states"
    samples_dir = generation_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    rebuilt: list[dict[str, Any]] = []
    for row in selected_replay_rows(replay_cfg):
        stem = artifact_stem(
            str(row["sample_id"]),
            int(row["seed"]),
            float(row["temperature"]),
        )
        hidden_path = hidden_dir / f"{stem}.npz"
        if not hidden_path.exists():
            continue
        sample = load_source_sample(source_run, str(row["sample_id"]))
        write_json(
            samples_dir / f"{sanitize_filename(str(row['sample_id']))}.json",
            sample,
        )
        rebuilt.append(
            {
                "sample_id": row["sample_id"],
                "seed": row["seed"],
                "temperature": row["temperature"],
                "generated_token_ids": row["generated_token_ids"],
                "produced_text": row["produced_text"],
                "produced_answer": row.get("produced_answer"),
                "is_correct": row.get("is_correct"),
                "dp2_idx": row.get("dp2_idx"),
                "reasoning_length": row.get("reasoning_length"),
                "hidden_states_file": hidden_path.relative_to(run_path).as_posix(),
            }
        )
    generation_dir.mkdir(parents=True, exist_ok=True)
    generation_path = generation_dir / "generations.jsonl"
    with generation_path.open("w", encoding="utf-8") as handle:
        for row in rebuilt:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        generation_dir / "metadata.json",
        {
            "schema_version": 2,
            "model_name": config["model"]["name"],
            "layer_indices": layers,
            "hidden_state_convention": HIDDEN_STATE_CONVENTION,
            "activation_storage_dtype": capture_cfg.get(
                "activation_storage_dtype", "int8_scaled"
            ),
            "components": components,
        },
    )
    return len(rebuilt)
