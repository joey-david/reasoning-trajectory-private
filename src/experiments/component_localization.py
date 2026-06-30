"""Localize symbolic updates in replay-captured MLP and attention outputs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from src.analysis.common import read_generation_rows
from src.experiments.common import robust_spike_indices
from src.experiments.localized_updates import mean_or_none, shifted_null_hit_rate
from src.runtime.artifact_store import load_component_states_npz
from src.runtime.data import load_samples


def run_component_localization(
    replay_run: Path,
    h2_dir: Path,
    *,
    window: int = 2,
) -> Path:
    """Score component-output norms at verified update-completion states."""
    updates = load_samples((h2_dir / "updates.jsonl").resolve())
    by_trace: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        by_trace[(str(update["sample_id"]), int(update["seed"]))].append(update)

    rows = [
        row for row in read_generation_rows(replay_run) if row.get("hidden_states_file")
    ]
    aggregate: defaultdict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        trace_updates = by_trace[(str(row["sample_id"]), int(row["seed"]))]
        if not trace_updates:
            continue
        endpoints = [int(update["state_index"]) for update in trace_updates]
        artifact = replay_run / row["hidden_states_file"]
        for component in ("mlp_output", "attention_output"):
            try:
                states, layers = load_component_states_npz(artifact, component)
            except KeyError:
                continue
            for layer_col, layer in enumerate(layers):
                norms = np.linalg.norm(states[:, layer_col].astype(np.float32), axis=1)
                valid = [min(max(index, 1), len(norms) - 1) for index in endpoints]
                labels = np.zeros(len(norms), dtype=np.int8)
                labels[valid] = 1
                if 0 < labels.sum() < len(labels):
                    aggregate[(component, layer)]["auc"].append(
                        float(roc_auc_score(labels, norms))
                    )
                spikes = robust_spike_indices(norms)
                hit_rate = float(
                    np.mean(
                        [
                            len(spikes) > 0
                            and np.min(np.abs(spikes - endpoint)) <= window
                            for endpoint in valid
                        ]
                    )
                )
                aggregate[(component, layer)]["hit_rate"].append(hit_rate)
                aggregate[(component, layer)]["null_hit_rate"].append(
                    shifted_null_hit_rate(
                        endpoints=valid,
                        spikes=spikes,
                        token_count=len(norms),
                        window=window,
                    )
                )

    results: list[dict[str, Any]] = []
    for (component, layer), values in aggregate.items():
        hits = np.asarray(values["hit_rate"])
        nulls = np.asarray(values["null_hit_rate"])
        results.append(
            {
                "component": component,
                "layer": layer,
                "traces": len(hits),
                "mean_update_auc": mean_or_none(values["auc"]),
                "mean_hit_rate": mean_or_none(hits),
                "mean_shifted_null_hit_rate": mean_or_none(nulls),
                "mean_hit_rate_lift": mean_or_none(hits - nulls),
            }
        )
    results.sort(
        key=lambda record: (
            record["mean_update_auc"] or 0.0,
            record["mean_hit_rate_lift"] or 0.0,
        ),
        reverse=True,
    )
    report = {
        "hypothesis": "H2_component_localization",
        "replay_run": replay_run.as_posix(),
        "source_updates": h2_dir.as_posix(),
        "traces": len(rows),
        "results": results,
        "recommended_patch_target": (
            {
                "component": results[0]["component"],
                "layer": results[0]["layer"],
            }
            if results
            else None
        ),
    }
    out_dir = replay_run / "analysis" / "experiments" / "h2_component_localization"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
