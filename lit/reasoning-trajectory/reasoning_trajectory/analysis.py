from __future__ import annotations

from pathlib import Path

import numpy as np

from reasoning_trajectory.branching import export_basins
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.storage import load_trajectories, save_table
from reasoning_trajectory.extract.generations import extract_from_config
from reasoning_trajectory.metrics.alignment import export_alignment, first_divergence
from reasoning_trajectory.metrics.geometry import export_geometry, step_matrix
from reasoning_trajectory.visualize.trajectory_3d import export_trajectory_plot


def pca_compress(x: np.ndarray, dims: int = 2) -> dict:
    from sklearn.decomposition import PCA

    if x.shape[0] < 2 or x.shape[1] < 1:
        return {"codes": np.zeros((x.shape[0], 0)), "reconstruction": np.array(x, copy=True), "error": 0.0, "explained_variance": []}
    pca = PCA(n_components=min(dims, *x.shape))
    z = pca.fit_transform(x)
    recon = pca.inverse_transform(z)
    return {"codes": z, "reconstruction": recon, "error": float(np.mean((x - recon) ** 2)), "explained_variance": pca.explained_variance_ratio_.tolist()}


@tool(
    "compression",
    "analysis",
    "Run compact PCA compression over trajectory step hidden states.",
    "rt compression --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/compression.jsonl",
    "reasoning_trajectory.analysis.export_compression",
    "toolkit/docs/tools/compression.md",
    dashboard=True,
)
def export_compression(input_path: str | Path, out: str | Path, dims: int = 2, layer: str | None = None) -> list[dict]:
    rows = []
    for traj in load_trajectories(input_path):
        result = pca_compress(step_matrix(traj, layer), dims)
        rows.append({"trajectory_id": traj.trajectory_id, "method": "pca", "dims": dims, "reconstruction_error": result["error"], "final_correct": traj.final_correct})
    save_table(rows, out)
    return rows


def failure_report(candidate: np.ndarray, nearest_valid: np.ndarray, texts: list[str] | None = None) -> dict:
    idx = first_divergence(nearest_valid, candidate)
    text = texts[idx] if texts and 0 <= idx < len(texts) else ""
    return {"first_divergence": idx, "suspected_labels": ["needs_manual_review"], "evidence_text": text}


@tool(
    "failure-autopsy",
    "analysis",
    "Find first divergence from the nearest valid trajectory.",
    "rt failures --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/failures.jsonl",
    "reasoning_trajectory.analysis.export_failure_reports",
    "toolkit/docs/tools/failure-autopsy.md",
    dashboard=True,
)
def export_failure_reports(input_path: str | Path, out: str | Path, layer: str | None = None) -> list[dict]:
    trajectories = load_trajectories(input_path)
    valid = [(t, step_matrix(t, layer)) for t in trajectories if t.final_correct is True]
    rows = []
    for traj in trajectories:
        if traj.final_correct is not False or not valid:
            continue
        x = step_matrix(traj, layer)
        ref_traj, ref = min(valid, key=lambda item: np.linalg.norm(item[1][-1] - x[-1]))
        rows.append({"trajectory_id": traj.trajectory_id, "nearest_valid": ref_traj.trajectory_id, **failure_report(x, ref, [s.text for s in traj.steps])})
    save_table(rows, out)
    return rows


@tool(
    "run-report",
    "analysis",
    "Export a concise markdown report for one trajectory run.",
    "rt report --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/report.md",
    "reasoning_trajectory.analysis.export_run_report",
    "toolkit/docs/tools/run-report.md",
)
def export_run_report(input_path: str | Path, out: str | Path) -> Path:
    run = Path(input_path)
    trajectories = load_trajectories(run)
    artifacts = []
    for name in ["trajectories.jsonl", "metrics/geometry.jsonl", "metrics/alignment.jsonl", "compression.jsonl", "basins.json", "failures.jsonl", "trajectory.html", "dashboard.html"]:
        path = run / name
        if path.exists():
            artifacts.append(f"- `{name}` ({path.stat().st_size} bytes)")
    lines = [
        "# Reasoning Trajectory Run Report",
        "",
        f"Run: `{run}`",
        f"Trajectories: {len(trajectories)}",
        f"Correct: {sum(t.final_correct is True for t in trajectories)}",
        f"Incorrect: {sum(t.final_correct is False for t in trajectories)}",
        f"Unknown: {sum(t.final_correct is None for t in trajectories)}",
        f"Models: {', '.join(sorted({t.model_name for t in trajectories}))}",
        "",
        "## Artifacts",
        *(artifacts or ["- No derived artifacts found."]),
    ]
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze_run(input_path: str | Path, layer: str | None = None, dims: int = 2) -> dict[str, Path | None]:
    from reasoning_trajectory.dashboard.app import launch_or_export_dashboard

    run = Path(input_path)
    artifacts: dict[str, Path | None] = {
        "geometry": run / "metrics" / "geometry.jsonl",
        "alignment": run / "metrics" / "alignment.jsonl",
        "compression": run / "compression.jsonl",
        "basins": run / "basins.json",
        "trajectory": run / "trajectory.html",
        "dashboard": run / "dashboard.html",
        "report": run / "report.md",
    }
    export_geometry(run, artifacts["geometry"], layer)
    export_alignment(run, artifacts["alignment"], layer)
    export_compression(run, artifacts["compression"], dims, layer)
    export_basins(run, artifacts["basins"], layer=layer)
    trajectories = load_trajectories(run)
    if any(t.final_correct is False for t in trajectories) and any(t.final_correct is True for t in trajectories):
        artifacts["failures"] = run / "failures.jsonl"
        export_failure_reports(run, artifacts["failures"], layer)
    else:
        artifacts["failures"] = None
    export_trajectory_plot(run, artifacts["trajectory"], layer)
    launch_or_export_dashboard(run, artifacts["dashboard"])
    export_run_report(run, artifacts["report"])
    return artifacts


def run_config(config_path: str | Path, out: str | Path, analyze: bool = True, layer: str | None = None) -> Path:
    run = Path(out)
    extract_from_config(config_path, run)
    if analyze:
        analyze_run(run, layer)
    return run
