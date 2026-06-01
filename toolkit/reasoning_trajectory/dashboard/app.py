from __future__ import annotations

import json
from html import escape
from pathlib import Path

from reasoning_trajectory.core.storage import load_trajectories
from reasoning_trajectory.core.registry import tool
from reasoning_trajectory.metrics.geometry import trajectory_geometry


def render_dashboard_summary(input_path: str | Path) -> str:
    trajectories = load_trajectories(input_path)
    html = [
        "<html><body><h1>Reasoning Trajectory Dashboard</h1><nav>Overview | Trajectories | Metrics | Compression | Basins | Failures</nav>",
        overview_page(trajectories),
        trajectories_page(trajectories),
    ]
    if Path(input_path).is_dir():
        html.append(artifact_page(input_path, "Metrics", ["geometry.jsonl", "alignment.jsonl", "basins.json"]))
        html.append(artifact_page(input_path, "Compression Results", ["compression.jsonl"]))
        html.append(failures_page(input_path))
    html.append("<h2>Geometry Metrics</h2><pre>" + "\n".join(str(trajectory_geometry(t)) for t in trajectories if t.steps) + "</pre>")
    html.append("</body></html>")
    return "\n".join(html)


def overview_page(trajectories) -> str:
    solved = sum(t.final_correct is True for t in trajectories)
    failed = sum(t.final_correct is False for t in trajectories)
    models = sorted({t.model_name for t in trajectories})
    datasets = sorted({t.dataset for t in trajectories})
    return (
        "<section><h2>Overview</h2>"
        f"<p>Trajectories: {len(trajectories)} | Correct: {solved} | Incorrect: {failed}</p>"
        f"<p>Datasets: {', '.join(datasets)}</p><p>Models: {', '.join(models)}</p>"
        "</section>"
    )


def trajectories_page(trajectories) -> str:
    rows = ["<section><h2>Trajectory Explorer</h2><table border='1'><tr><th>id</th><th>problem</th><th>seed</th><th>temp</th><th>correct</th><th>path length</th></tr>"]
    for t in trajectories:
        try:
            path_length = f"{trajectory_geometry(t)['path_length']:.3f}"
        except Exception:
            path_length = "n/a"
        rows.append(f"<tr><td>{escape(t.trajectory_id)}</td><td>{escape(t.problem_id)}</td><td>{t.seed}</td><td>{t.temperature}</td><td>{t.final_correct}</td><td>{path_length}</td></tr>")
    rows.append("</table>")
    for t in trajectories:
        rows.append(f"<details><summary>{escape(t.trajectory_id)} reasoning</summary><pre>{escape(t.final_text)}</pre></details>")
    rows.append("</section>")
    return "\n".join(rows)


def artifact_page(run_path: str | Path, title: str, filenames: list[str]) -> str:
    run = Path(run_path)
    parts = [f"<section><h2>{title}</h2>"]
    for name in filenames:
        path = run / name
        if not path.exists():
            parts.append(f"<p>{name}: not present</p>")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                text = json.dumps(json.loads(text), indent=2)
            except Exception as exc:
                text = f"{text}\n\nJSON parse warning: {exc}"
        parts.append(f"<details open><summary>{name}</summary><pre>{escape(text[:12000])}</pre></details>")
    parts.append("</section>")
    return "\n".join(parts)


def failures_page(run_path: str | Path) -> str:
    path = Path(run_path) / "failures.jsonl"
    if not path.exists():
        return "<section><h2>Failure Autopsy</h2><p>No failure report present.</p></section>"
    return f"<section><h2>Failure Autopsy</h2><pre>{escape(path.read_text(encoding='utf-8')[:12000])}</pre></section>"


@tool(
    "dashboard",
    "dashboard",
    "Inspect run metadata, problems, trajectories, correctness, reasoning text, metrics, and optional artifacts.",
    "rt dashboard --input experiments/runs/r1_distill_sheep30 --out dashboard.html",
    "reasoning_trajectory.dashboard.app.launch_or_export_dashboard",
    "toolkit/docs/tools/dashboard.md",
    dashboard=True,
)
def launch_or_export_dashboard(input_path: str | Path, out: str | Path | None = None) -> Path | None:
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_dashboard_summary(input_path), encoding="utf-8")
        return path
    try:
        import streamlit.web.cli as stcli
    except ImportError as exc:
        raise ImportError("interactive dashboard requires streamlit; pass --out for static HTML") from exc
    import sys
    app_path = Path(__file__)
    sys.argv = ["streamlit", "run", str(app_path), "--", str(input_path)]
    stcli.main()
    return None


def streamlit_main(input_path: str | Path) -> None:
    import streamlit as st

    from reasoning_trajectory.analysis import export_compression, export_failure_reports, export_run_report
    from reasoning_trajectory.branching import export_basins
    from reasoning_trajectory.metrics.alignment import export_alignment
    from reasoning_trajectory.metrics.geometry import export_geometry
    from reasoning_trajectory.visualize.trajectory_3d import build_trajectory_figure

    st.set_page_config(page_title="Reasoning Trajectory Workbench", layout="wide")
    run = _select_run(st, Path(input_path))
    if not run.exists():
        st.error(f"Run path does not exist: {run}")
        return
    trajectories = load_trajectories(run)
    layers = _available_layers(trajectories)

    st.title("Reasoning Trajectory Workbench")
    st.caption(str(run))
    cols = st.columns(4)
    cols[0].metric("Trajectories", len(trajectories))
    cols[1].metric("Correct", sum(t.final_correct is True for t in trajectories))
    cols[2].metric("Incorrect", sum(t.final_correct is False for t in trajectories))
    cols[3].metric("Unknown", sum(t.final_correct is None for t in trajectories))

    model = st.sidebar.multiselect("Model", sorted({t.model_name for t in trajectories}), default=sorted({t.model_name for t in trajectories}), key="model_filter")
    correctness = st.sidebar.multiselect("Correctness", ["True", "False", "None"], default=["True", "False", "None"], key="correctness_filter")
    layer = st.sidebar.selectbox("Layer", layers or ["auto"], index=max(0, len(layers) - 1), key="layer_filter")
    layer_arg = None if layer == "auto" else str(layer)
    filtered = [t for t in trajectories if t.model_name in model and str(t.final_correct) in correctness]
    if not filtered:
        st.warning("No trajectories match the current filters.")
        return

    tabs = st.tabs(["Trajectories", "Projection", "Metrics", "Compression", "Basins", "Failures", "Report", "Artifacts"])
    with tabs[0]:
        selected = st.selectbox("Trajectory", [t.trajectory_id for t in filtered], key="trajectory_select")
        traj = next(t for t in filtered if t.trajectory_id == selected)
        _trajectory_panel(st, traj)
    with tabs[1]:
        color_by = st.selectbox("Color by", ["correctness", "step", "layer", "verifier"], key="plot_color")
        method = st.selectbox("Projection", ["pca", "umap", "tsne"], key="plot_method")
        plot_cols = st.columns(4)
        marker_size = plot_cols[0].slider("Point size", 1, 10, 3, key="plot_marker_size")
        line_width = plot_cols[1].slider("Line width", 1, 8, 2, key="plot_line_width")
        show_endpoints = plot_cols[2].toggle("Endpoint markers", value=True, key="plot_endpoints")
        show_arrows = plot_cols[3].toggle("Direction arrows", value=False, key="plot_arrows")
        fig, diag = build_trajectory_figure(
            filtered,
            layer=layer_arg,
            color_by=color_by,
            method=method,
            marker_size=marker_size,
            line_width=line_width,
            show_arrows=show_arrows,
            show_endpoints=show_endpoints,
        )
        st.plotly_chart(fig, width="stretch")
        st.json(diag)
        out = run / "trajectory.html"
        if st.button("Save Plot HTML", key="save_plot", help="Write the current figure to trajectory.html"):
            fig.write_html(out)
            st.success(f"Wrote {out}")
    with tabs[2]:
        _metrics_panel(st, run, filtered, layer_arg, export_geometry, export_alignment)
    with tabs[3]:
        _compression_panel(st, run, layer_arg, export_compression)
    with tabs[4]:
        _basins_panel(st, run, layer_arg, export_basins)
    with tabs[5]:
        _failures_panel(st, run, layer_arg, export_failure_reports)
    with tabs[6]:
        _report_panel(st, run, export_run_report)
    with tabs[7]:
        _artifact_panel(st, run)


def _select_run(st, initial: Path) -> Path:
    root = Path("runs")
    candidates = _run_candidates(root)
    initial = initial if initial.exists() else (candidates[0] if candidates else initial)
    with st.sidebar:
        st.header("Run")
        labels = [str(p) for p in candidates]
        if labels:
            default = labels.index(str(initial)) if str(initial) in labels else 0
            selected = st.selectbox("Run folder", labels, index=default, key="run_folder")
            run = Path(selected)
        else:
            run = initial
        if st.checkbox("Use custom path", key="use_custom_run_path"):
            run = Path(st.text_input("Custom run path", value=str(run), key="custom_run_path"))
        st.header("Config")
        configs = sorted(Path("configs").glob("*.yaml")) + sorted(Path("configs").glob("*.yml"))
        if configs:
            cfg = st.selectbox("Config preview", [str(p) for p in configs], key="config_preview")
            if st.checkbox("Show config", key="show_config"):
                st.code(Path(cfg).read_text(encoding="utf-8"), language="yaml")
    return run


def _run_candidates(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "trajectories.jsonl").exists()], key=lambda p: p.stat().st_mtime, reverse=True)


def _available_layers(trajectories) -> list[str]:
    layers = set()
    for traj in trajectories:
        for step in traj.steps:
            layers.update(step.hidden_states)
    return sorted(layers, key=lambda x: int(x) if str(x).isdigit() else str(x))


def _trajectory_panel(st, traj) -> None:
    st.write(
        {
            "trajectory": traj.trajectory_id,
            "problem": traj.problem_id,
            "correct": traj.final_correct,
            "expected": traj.metadata.get("expected_answer"),
            "predicted": traj.metadata.get("predicted_answer"),
            "seed": traj.seed,
            "temperature": traj.temperature,
            "steps": len(traj.steps),
        }
    )
    st.text_area("Generated reasoning", traj.final_text, height=260)
    st.dataframe([{"step": s.step_id, "tokens": f"{s.token_start}-{s.token_end}", "labels": ", ".join(s.labels), "text": s.text} for s in traj.steps], width="stretch")


def _metrics_panel(st, run: Path, filtered, layer: str | None, export_geometry_fn, export_alignment_fn) -> None:
    left, right = st.columns(2)
    if left.button("Run Geometry", key="run_geometry"):
        path = run / "metrics" / "geometry.jsonl"
        export_geometry_fn(run, path, layer)
        st.success(f"Wrote {path}")
    if right.button("Run Alignment", key="run_alignment"):
        path = run / "metrics" / "alignment.jsonl"
        export_alignment_fn(run, path, layer)
        st.success(f"Wrote {path}")
    st.dataframe([trajectory_geometry(t, layer) for t in filtered if t.steps], width="stretch")
    _preview_file(st, run / "metrics" / "geometry.jsonl")
    _preview_file(st, run / "metrics" / "alignment.jsonl")


def _compression_panel(st, run: Path, layer: str | None, export_compression_fn) -> None:
    dims = st.number_input("PCA dimensions", min_value=1, max_value=128, value=2, step=1, key="compression_dims")
    if st.button("Run Compression", key="run_compression"):
        path = run / "compression.jsonl"
        export_compression_fn(run, path, int(dims), layer)
        st.success(f"Wrote {path}")
    _preview_file(st, run / "compression.jsonl")


def _basins_panel(st, run: Path, layer: str | None, export_basins_fn) -> None:
    clusters = st.number_input("Clusters", min_value=1, max_value=20, value=3, step=1, key="basin_clusters")
    if st.button("Run Basins", key="run_basins"):
        path = run / "basins.json"
        export_basins_fn(run, path, int(clusters), layer)
        st.success(f"Wrote {path}")
    _preview_file(st, run / "basins.json")


def _failures_panel(st, run: Path, layer: str | None, export_failures_fn) -> None:
    if st.button("Run Failure Autopsy", key="run_failures"):
        path = run / "failures.jsonl"
        export_failures_fn(run, path, layer)
        st.success(f"Wrote {path}")
    _preview_file(st, run / "failures.jsonl")


def _report_panel(st, run: Path, export_report_fn) -> None:
    if st.button("Generate Report", key="run_report"):
        path = run / "report.md"
        export_report_fn(run, path)
        st.success(f"Wrote {path}")
    _preview_file(st, run / "report.md")


def _artifact_panel(st, run: Path) -> None:
    files = sorted([p for p in run.rglob("*") if p.is_file()])
    if not files:
        st.info("No artifacts found.")
        return
    selected = st.selectbox("Artifact", [str(p.relative_to(run)) for p in files], key="artifact_select")
    _preview_file(st, run / selected, expanded=True)


def _preview_file(st, path: Path, expanded: bool = False) -> None:
    with st.expander(str(path), expanded=expanded):
        if not path.exists():
            st.info("Not present yet.")
            return
        st.caption(f"{path.stat().st_size:,} bytes")
        if path.suffix == ".html":
            st.link_button("Open HTML artifact", str(path), help="Open this generated HTML file in a browser")
            st.code(str(path), language="text")
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix == ".json":
            try:
                st.json(json.loads(text))
                return
            except Exception:
                pass
        st.code(text[:20000], language="json" if path.suffix == ".jsonl" else "markdown" if path.suffix == ".md" else "text")


if __name__ == "__main__":
    import sys
    streamlit_main(sys.argv[1] if len(sys.argv) > 1 else "experiments/runs/r1_distill_sheep30")
