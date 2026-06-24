from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


T = TypeVar("T")


def read_generation_rows(run_path: Path) -> list[dict[str, Any]]:
    path = run_path / "generation" / "generations.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_sample_records(run_path: Path) -> dict[str, dict[str, Any]]:
    sample_dir = run_path / "generation" / "samples"
    return {
        path.stem: json.loads(path.read_text()) for path in sample_dir.glob("*.json")
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def project_3d(
    x: np.ndarray,
    *,
    random_state: int | None = None,
    tsne_perplexity: int = 30,
) -> dict[str, np.ndarray]:
    return {
        "pca": PCA(n_components=3, random_state=random_state).fit_transform(x),
        "tsne": TSNE(
            n_components=3,
            perplexity=min(tsne_perplexity, len(x) - 1),
            init="random",
            learning_rate="auto",
            random_state=random_state,
        ).fit_transform(x),
    }


def evenly_capped(items: list[T], max_items: int) -> list[T]:
    if max_items <= 0 or len(items) <= max_items:
        return items
    keep = np.linspace(0, len(items) - 1, max_items, dtype=int)
    return [items[int(i)] for i in keep]
