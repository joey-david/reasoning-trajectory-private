"""Train H4 structural projections in the component space patched by H3."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from reasoning_trajectory.artifacts import read_generation_rows
from src.experiments.common import update_phase_bounds
from src.experiments.trajectory_dynamics.structural_contrast import fit_structural_projection
from src.runtime.artifact_store import load_component_states_npz
from src.runtime.data import load_samples


def run_component_projection(
    replay_run: Path,
    h2_dir: Path,
    out_dir: Path,
    *,
    component: str,
    layer: int,
    max_updates: int = 12000,
    max_pairs: int = 20000,
    epochs: int = 12,
    projection_dim: int = 128,
) -> Path:
    """Extract interval net vectors and fit a component-matched H4 projection.

    Args:
        replay_run: Run directory containing teacher-forced replay artifacts.
        h2_dir: Directory containing H2 update-analysis artifacts.
        out_dir: Directory in which to write the results.
        component: Activation component name.
        layer: Model layer index.
        max_updates: Maximum number of symbolic updates to retain.
        max_pairs: Maximum number of pairs to retain.
        epochs: Number of projection-training epochs.
        projection_dim: Width of the learned projection space.

    Returns:
        The path of the written or discovered artifact.
    """
    if component not in {"attention_output", "mlp_output"}:
        raise ValueError(f"Unsupported component projection: {component!r}")

    updates = load_samples((h2_dir / "updates.jsonl").resolve())
    by_trace: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for update in updates:
        by_trace[(str(update["sample_id"]), int(update["seed"]))].append(update)

    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for row in read_generation_rows(replay_run):
        trace_key = (str(row["sample_id"]), int(row["seed"]))
        trace_updates = sorted(
            by_trace[trace_key],
            key=lambda update: int(update["update_index"]),
        )
        if not trace_updates or not row.get("hidden_states_file"):
            continue
        states, layers = load_component_states_npz(
            replay_run / row["hidden_states_file"],
            component,
        )
        if layer not in layers:
            raise ValueError(
                f"Layer {layer} is unavailable for {component} in "
                f"{row['hidden_states_file']}"
            )
        layer_states = states[:, layers.index(layer)].astype(np.float32)
        for update in trace_updates:
            if int(update["token_end"]) + 1 >= len(layer_states):
                continue
            phase_start, phase_end = update_phase_bounds(
                int(update["token_start"]),
                int(update["token_end"]),
                len(layer_states),
            )
            record = {
                **update,
                "feature_row": len(vectors),
                "component": component,
                "component_layer": layer,
                "phase_start_state_index": phase_start,
                "phase_end_state_index": phase_end,
            }
            records.append(record)
            vectors.append(layer_states[phase_end] - layer_states[phase_start])

    if not vectors:
        raise ValueError("No component interval vectors were extracted")
    prefix = f"{component}_layer{layer}"
    return fit_structural_projection(
        records=records,
        vectors=np.stack(vectors),
        out_dir=out_dir,
        projection_filename=f"{prefix}_projection.pt",
        source=replay_run.as_posix(),
        layer=layer,
        component=component,
        update_vector="component-output net displacement across symbolic interval",
        max_updates=max_updates,
        max_pairs=max_pairs,
        epochs=epochs,
        projection_dim=projection_dim,
        output_prefix=prefix,
        write_pair_manifests=False,
    )
