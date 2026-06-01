from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import Trajectory
from reasoning_trajectory.extract.token_steps import steps_from_text


def from_legacy_record(record: dict, token_hidden_states: np.ndarray | None = None) -> Trajectory:
    text = record.get("final_text") or record.get("generated_text") or record.get("response") or ""
    prompt = record.get("prompt") or record.get("question") or ""
    steps = steps_from_text(text, token_hidden_states)
    return Trajectory(
        trajectory_id=str(record.get("trajectory_id") or record.get("id") or record.get("problem_id") or "legacy"),
        problem_id=str(record.get("problem_id") or record.get("id") or "legacy"),
        dataset=str(record.get("dataset") or record.get("task") or "legacy"),
        model_name=str(record.get("model_name") or record.get("model") or "unknown"),
        prompt=prompt,
        seed=int(record.get("seed") or 0),
        temperature=float(record.get("temperature") or 0.0),
        decoding_method=str(record.get("decoding_method") or "unknown"),
        final_text=text,
        final_answer=record.get("final_answer") or record.get("produced_answer"),
        final_correct=record.get("final_correct") if "final_correct" in record else record.get("correct"),
        solution_object_id=record.get("solution_object_id"),
        steps=steps,
        metadata={"created_at": record.get("created_at", "legacy"), "repo_commit": record.get("repo_commit", "unknown"), "config_hash": record.get("config_hash", "legacy")},
    )


def convert_legacy_jsonl(path: str | Path) -> list[Trajectory]:
    trajectories = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(from_legacy_record(json.loads(line)))
    return trajectories
