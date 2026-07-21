"""Aggregate and plot Yang causal-mediation head maps."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.experiments.layer_replications.common import read_jsonl, replication_dir
from src.experiments.layer_replications.symbolic import (
    MECHANISMS,
    cma_path,
    cma_task_key,
    cma_tasks,
    selection_path,
)
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


def fwer_threshold(
    matrices: np.ndarray, *, trials: int, seed: int
) -> tuple[float, np.ndarray]:
    """Run the paper's sign-swap permutation test with max-head FWER control."""
    rng = np.random.default_rng(seed)
    maxima = np.empty(trials, dtype=np.float64)
    for index in range(trials):
        signs = rng.choice((-1.0, 1.0), size=(matrices.shape[0], 1, 1))
        maxima[index] = np.max(np.mean(matrices * signs, axis=0))
    threshold = float(np.quantile(maxima, 0.95))
    return threshold, np.mean(matrices, axis=0) > threshold


def analyze(run_path: Path) -> dict[str, Any]:
    """Identify significant heads and write the three mechanism reports."""
    config = load_config(run_path)
    selection = read_jsonl(selection_path(run_path))
    expected = {
        cma_task_key(str(selection[task["pair_index"]]["id"]), task["mechanism"])
        for task in cma_tasks(run_path)
    }
    rows = read_jsonl(cma_path(run_path))
    missing = expected - {str(row["key"]) for row in rows}
    if missing:
        raise RuntimeError(
            f"symbolic CMA incomplete: {len(missing)}/{len(expected)} tasks remain"
        )
    trials = int(config["symbolic_mechanisms"].get("permutation_trials", 5000))
    seed = int(config["symbolic_mechanisms"]["benchmark"]["seed"])
    out = replication_dir(run_path) / "yang_symbolic"
    mechanisms: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(MECHANISMS):
        selected = [row for row in rows if row["mechanism"] == mechanism]
        matrices = np.asarray([row["scores"] for row in selected], dtype=np.float64)
        threshold, significant = fwer_threshold(
            matrices, trials=trials, seed=seed + mechanism_index
        )
        mean = matrices.mean(axis=0)
        peak = np.unravel_index(np.argmax(mean), mean.shape)
        mechanisms[mechanism] = {
            "pairs": len(selected),
            "fwer_threshold": threshold,
            "significant_heads": int(significant.sum()),
            "peak_layer": int(peak[0]),
            "peak_head": int(peak[1]),
            "peak_score": float(mean.max()),
        }
        for layer, head in np.ndindex(mean.shape):
            csv_rows.append(
                {
                    "mechanism": mechanism,
                    "layer": layer,
                    "head": head,
                    "mean_causal_score": float(mean[layer, head]),
                    "significant_fwer_0_05": bool(significant[layer, head]),
                }
            )
        _plot_heatmap(
            mean, significant, mechanism.replace("_", " "), out / f"{mechanism}.png"
        )
    with (out / "head_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "paper": "Yang et al., Emergent Symbolic Mechanisms",
        "complete": True,
        "permutation_trials": trials,
        "mechanisms": mechanisms,
    }
    write_json(out / "report.json", report)
    return report


def _plot_heatmap(
    mean: np.ndarray, significant: np.ndarray, title: str, path: Path
) -> None:
    """Render one compact causal head map with significant heads outlined."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(mean, aspect="auto", origin="lower", cmap="viridis")
    layer_ids, head_ids = np.where(significant)
    axis.scatter(head_ids, layer_ids, facecolors="none", edgecolors="white", s=26)
    axis.set(xlabel="head", ylabel="layer", title=title)
    figure.colorbar(image, ax=axis, label="causal mediation score")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
