"""Compute and plot Zhang et al. layer contribution."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiments.layer_replications.common import replication_dir
from src.experiments.layer_replications.single_layer_rl import evaluation_path
from src.runtime.artifact_store import write_json
from src.runtime.config import load_config


def analyze(run_path: Path, *, layers: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Compute the paper's contribution curve from the requested scan."""
    config = load_config(run_path)
    layer_count = int(config["model"]["layer_count"])
    selected_layers = tuple(
        layers or config["single_layer_rl"].get("core_scan_layers", range(layer_count))
    )
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("RL analysis layers must be non-empty and unique")
    if min(selected_layers) < 0 or max(selected_layers) >= layer_count:
        raise ValueError(f"RL analysis layers must lie in [0, {layer_count})")
    names = ["base", "full", *(f"layer-{index:02d}" for index in selected_layers)]
    reports = {}
    missing = []
    for name in names:
        path = evaluation_path(run_path, name)
        if not path.exists():
            missing.append(name)
        else:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise RuntimeError(f"single-layer RL evaluations missing: {', '.join(missing)}")
    base = float(reports["base"]["math_average"])
    full = float(reports["full"]["math_average"])
    denominator = full - base
    if denominator <= 0:
        raise RuntimeError(
            f"full GRPO did not improve the base model: base={base}, full={full}"
        )
    curve = []
    for layer in selected_layers:
        score = float(reports[f"layer-{layer:02d}"]["math_average"])
        curve.append(
            {
                "layer": layer,
                "math_average": score,
                "contribution": (score - base) / denominator,
            }
        )
    out = replication_dir(run_path) / "zhang_single_layer_rl"
    with (out / "layer_contribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    best = max(curve, key=lambda row: row["contribution"])
    report = {
        "paper": "Zhang et al., Is One Layer Enough?",
        "complete": True,
        "scan_complete": len(selected_layers) == layer_count,
        "scanned_layers": list(selected_layers),
        "base_math_average": base,
        "full_math_average": full,
        "best_layer": best["layer"],
        "best_contribution": best["contribution"],
        "layers_matching_full": sum(row["contribution"] >= 1.0 for row in curve),
        "curve": curve,
    }
    write_json(out / "report.json", report)
    _plot_curve(curve, out / "layer_contribution.png")
    return report


def _plot_curve(curve: list[dict[str, Any]], path: Path) -> None:
    """Render the paper's central layer-contribution plot."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(
        [row["layer"] for row in curve],
        [row["contribution"] for row in curve],
        marker="o",
        markersize=4,
    )
    axis.axhline(1.0, color="tab:green", linestyle="--", label="full GRPO")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="trained decoder layer", ylabel="layer contribution")
    axis.grid(alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
