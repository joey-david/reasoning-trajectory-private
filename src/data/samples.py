from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json


@dataclass(frozen=True)
class Sample:
    # A normalized prompt record, no matter how the dataset was stored.
    id: str
    prompt: str
    expected_answer: str | None = None
    metadata: dict[str, Any] | None = None


def load_samples(config: dict[str, Any]) -> list[Sample]:
    # Tiny experiments can keep prompts directly inside config.yaml.
    if "prompts" in config:
        return [_sample_from_dict(item, idx) for idx, item in enumerate(config["prompts"])]

    # Larger experiments point to a dataset file under datasets/ or elsewhere.
    dataset_path = _resolve_path(config["dataset_path"], config)
    if dataset_path.suffix == ".jsonl":
        return _load_jsonl(dataset_path)
    if dataset_path.suffix == ".json":
        # JSON holds one big list; JSONL holds one JSON object per line.
        return [_sample_from_dict(item, idx) for idx, item in enumerate(json.loads(dataset_path.read_text()))]
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def _load_jsonl(path: Path) -> list[Sample]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            # Blank lines are harmless in hand-edited jsonl files.
            if line.strip():
                samples.append(_sample_from_dict(json.loads(line), idx))
    return samples


def _resolve_path(value: str, config: dict[str, Any]) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path

    # If the script is launched outside the repo root, walk upward from the run.
    run_path = Path(config["_run_path"])
    for parent in [run_path, *run_path.parents]:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return path


def _sample_from_dict(item: dict[str, Any], idx: int) -> Sample:
    # Support common field names so new datasets need little adapter code.
    prompt = item.get("prompt") or item.get("question") or item.get("input")
    if not prompt:
        raise ValueError(f"Sample {idx} has no prompt/question/input")
    sample_id = item.get("id") or item.get("problem_id") or f"sample_{idx}"
    return Sample(
        id=str(sample_id),
        prompt=str(prompt),
        expected_answer=None if item.get("expected_answer") is None else str(item["expected_answer"]),
        # Keep extra fields available for analysis without special-casing them.
        metadata={k: v for k, v in item.items() if k not in {"prompt", "question", "input", "expected_answer"}},
    )
