from __future__ import annotations

from pathlib import Path

import numpy as np

from reasoning_trajectory.core.storage import load_trajectories
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.core.schema import Trajectory
from .projections import project


def _points(trajectories: list[Trajectory], layer: str | None = None) -> tuple[np.ndarray, list[dict]]:
    rows, meta = [], []
    for traj in trajectories:
        visible_steps = [step for step in traj.steps if step.hidden_states]
        total = max(1, len(visible_steps) - 1)
        for step_index, step in enumerate(visible_steps):
            if not step.hidden_states:
                continue
            key = layer or sorted(step.hidden_states.keys(), key=str)[-1]
            if key not in step.hidden_states:
                continue
            rows.append(np.asarray(step.hidden_states[key], dtype=float).reshape(-1))
            meta.append(
                {
                    "trajectory_id": traj.trajectory_id,
                    "problem_id": traj.problem_id,
                    "step_id": step.step_id,
                    "step_index": step_index,
                    "step_count": len(visible_steps),
                    "progress": step_index / total,
                    "text": step.text,
                    "correct": traj.final_correct,
                    "predicted": traj.metadata.get("predicted_answer"),
                    "expected": traj.metadata.get("expected_answer"),
                    "seed": traj.seed,
                    "layer": key,
                    "labels": ",".join(step.labels),
                }
            )
    if not rows:
        raise ValueError("no hidden states found for visualization")
    width = max(len(r) for r in rows)
    arr = np.vstack([np.pad(r, (0, width - len(r))) for r in rows])
    return arr, meta


def build_trajectory_figure(
    trajectories: list[Trajectory],
    layer: str | None = None,
    color_by: str = "step",
    method: str = "pca",
    dims: int = 3,
    marker_size: int = 3,
    line_width: int = 2,
    show_arrows: bool = False,
    show_endpoints: bool = True,
):
    import plotly.graph_objects as go
    points, meta = _points(trajectories, layer)
    xyz, diag = project(points, method=method, dims=dims)
    fig = go.Figure()

    by_traj: dict[str, list[int]] = {}
    for i, row in enumerate(meta):
        by_traj.setdefault(row["trajectory_id"], []).append(i)

    for traj_id, idxs in sorted(by_traj.items()):
        idxs = sorted(idxs, key=lambda i: meta[i]["step_index"])
        first = meta[idxs[0]]
        base = _base_color(first, color_by)
        marker_colors = [_progress_color(base, meta[i]["progress"]) for i in idxs]
        hover = [_hover_text(meta[i]) for i in idxs]
        name = _trace_name(first, color_by)
        fig.add_trace(
            go.Scatter3d(
                x=xyz[idxs, 0],
                y=xyz[idxs, 1],
                z=xyz[idxs, 2],
                mode="markers+lines",
                name=name,
                legendgroup=name,
                text=hover,
                hoverinfo="text",
                marker=dict(size=marker_size, color=marker_colors, opacity=0.86, line=dict(width=0)),
                line=dict(color=_rgba(base, 0.46), width=line_width),
            )
        )
        if show_endpoints and idxs:
            start, end = idxs[0], idxs[-1]
            fig.add_trace(
                go.Scatter3d(
                    x=[xyz[start, 0], xyz[end, 0]],
                    y=[xyz[start, 1], xyz[end, 1]],
                    z=[xyz[start, 2], xyz[end, 2]],
                    mode="markers",
                    name=f"{name} endpoints",
                    legendgroup=name,
                    showlegend=False,
                    text=[f"START<br>{hover[0]}", f"END<br>{hover[-1]}"],
                    hoverinfo="text",
                    marker=dict(size=[marker_size + 2, marker_size + 4], color=[_rgba(base, 0.35), _rgba(base, 1.0)], symbol=["circle", "diamond"], line=dict(width=1, color="rgba(20,20,20,0.45)")),
                )
            )

    if show_arrows:
        for traj_id, idxs in sorted(by_traj.items()):
            idxs = sorted(idxs, key=lambda i: meta[i]["step_index"])
            base = _base_color(meta[idxs[0]], color_by)
            for a, b in zip(idxs, idxs[1:]):
                fig.add_trace(
                    go.Cone(
                        x=[xyz[a, 0]], y=[xyz[a, 1]], z=[xyz[a, 2]],
                        u=[xyz[b, 0] - xyz[a, 0]], v=[xyz[b, 1] - xyz[a, 1]], w=[xyz[b, 2] - xyz[a, 2]],
                        showscale=False, sizemode="absolute", sizeref=0.12, name=f"{traj_id} direction", hoverinfo="skip", showlegend=False,
                        colorscale=[[0, _rgba(base, 0.2)], [1, _rgba(base, 0.72)]],
                    )
                )

    variance = diag.get("explained_variance") or []
    axis_titles = [_axis_title(i, variance) for i in range(3)]
    fig.update_layout(
        title=f"Trajectory projection ({diag['method']}, trust={diag.get('trustworthiness')})",
        scene=dict(
            xaxis_title=axis_titles[0],
            yaxis_title=axis_titles[1],
            zaxis_title=axis_titles[2],
            xaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="rgb(218,218,218)", zerolinecolor="rgb(170,170,170)"),
            yaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="rgb(218,218,218)", zerolinecolor="rgb(170,170,170)"),
            zaxis=dict(backgroundcolor="rgb(250,250,250)", gridcolor="rgb(218,218,218)", zerolinecolor="rgb(170,170,170)"),
        ),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=42, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_layout(meta=diag)
    return fig, diag


def _trace_name(row: dict, color_by: str) -> str:
    if color_by == "correctness":
        label = "correct" if row["correct"] is True else "incorrect" if row["correct"] is False else "unknown"
        return f"{label} / {row['trajectory_id']}"
    if color_by == "layer":
        return f"layer {row['layer']} / {row['trajectory_id']}"
    if color_by == "verifier":
        return f"{row['labels'] or 'unlabeled'} / {row['trajectory_id']}"
    return row["trajectory_id"]


def _base_color(row: dict, color_by: str) -> tuple[int, int, int]:
    if color_by == "correctness":
        if row["correct"] is True:
            return (29, 118, 90)
        if row["correct"] is False:
            return (190, 74, 63)
        return (116, 116, 116)
    if color_by == "layer":
        return _palette(str(row["layer"]))
    if color_by == "verifier":
        return _palette(row["labels"] or "unlabeled")
    return _palette(row["trajectory_id"])


def _palette(key: str) -> tuple[int, int, int]:
    colors = [(42, 92, 170), (198, 116, 36), (74, 129, 68), (145, 79, 150), (50, 136, 160), (175, 64, 100), (110, 98, 64)]
    return colors[sum(ord(c) for c in key) % len(colors)]


def _progress_color(base: tuple[int, int, int], progress: float) -> str:
    # Early steps are pale; late steps are saturated and darker.
    mix = 0.72 - 0.5 * progress
    darken = 1.0 - 0.18 * progress
    rgb = tuple(int((channel * (1 - mix) + 255 * mix) * darken) for channel in base)
    return _rgba(rgb, 0.9)


def _rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"


def _axis_title(index: int, variance: list[float]) -> str:
    if index < len(variance):
        return f"PC{index + 1} ({variance[index] * 100:.1f}%)"
    return f"dim {index + 1}"


def _hover_text(row: dict) -> str:
    return (
        f"<b>{row['trajectory_id']}</b> {row['step_id']} / {row['step_count']}<br>"
        f"problem={row['problem_id']} seed={row['seed']} correct={row['correct']}<br>"
        f"expected={row['expected']} predicted={row['predicted']} layer={row['layer']}<br>"
        f"progress={row['progress']:.2f}<br><br>{row['text']}"
    )


@tool(
    "trajectory-3d",
    "visualize",
    "Project trajectories to 2D/3D and export interactive HTML or static images.",
    "rt plot --input experiments/runs/r1_distill_sheep30 --out figs/traj.html",
    "reasoning_trajectory.visualize.trajectory_3d.export_trajectory_plot",
    "toolkit/docs/tools/trajectory-3d.md",
    dashboard=True,
)
def export_trajectory_plot(input_path: str | Path, out: str | Path, layer: str | None = None, color_by: str = "step", method: str = "pca") -> dict:
    trajectories = load_trajectories(input_path)
    fig, diag = build_trajectory_figure(trajectories, layer=layer, color_by=color_by, method=method)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".html":
        fig.write_html(out)
    elif out.suffix in {".png", ".svg", ".pdf"}:
        try:
            fig.write_image(out)
        except Exception as exc:
            raise RuntimeError("static export requires kaleido; use .html or install kaleido") from exc
    else:
        raise ValueError("visualization output must end in .html, .png, .svg, or .pdf")
    return diag
