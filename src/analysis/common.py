"""Provide shared run-record I/O, dimensionality reduction, and sampling helpers for analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.data import load_samples


T = TypeVar("T")


def read_generation_rows(run_path: Path) -> list[dict[str, Any]]:
    """Read all rollout rows from a completed run.

    Args:
        run_path: Run folder containing ``generation/generations.jsonl``.

    Returns:
        Parsed non-empty generation rows in file order.
    """
    path = run_path / "generation" / "generations.jsonl"
    return load_samples(path.resolve())


def read_sample_records(run_path: Path) -> dict[str, dict[str, Any]]:
    """Load persisted per-sample records keyed by their sanitized filename stem.

    Args:
        run_path: Run folder containing ``generation/samples``.

    Returns:
        Mapping from sample artifact stem to parsed sample record.
    """
    sample_dir = run_path / "generation" / "samples"
    return {
        path.stem: json.loads(path.read_text()) for path in sample_dir.glob("*.json")
    }


def project_3d(
    x: np.ndarray,
    *,
    random_state: int | None = None,
    tsne_perplexity: int = 30,
) -> dict[str, np.ndarray]:
    """Project a feature matrix into PCA and t-SNE three-dimensional spaces.

    Args:
        x: Two-dimensional sample-by-feature matrix with at least three rows.
        random_state: Optional deterministic seed for both projections.
        tsne_perplexity: Requested t-SNE perplexity, capped below sample count.

    Returns:
        ``pca`` and ``tsne`` coordinate arrays shaped ``[samples, 3]``.
    """
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
    """Downsample an ordered list at evenly spaced positions.

    Args:
        items: Ordered items to retain or sample.
        max_items: Maximum retained count; non-positive values disable capping.

    Returns:
        The original list when under the cap, otherwise evenly spaced items.
    """
    if max_items <= 0 or len(items) <= max_items:
        return items
    keep = np.linspace(0, len(items) - 1, max_items, dtype=int)
    return [items[int(i)] for i in keep]
