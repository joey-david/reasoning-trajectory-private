# Reasoning Trajectory Architecture

- `core/`: dataclass schema, storage, config hashing, dependency checks, fixtures.
- `extract/`: HuggingFace generation, answer extraction, and step segmentation.
- `metrics/`: geometry and alignment metrics over step hidden states.
- `analysis.py`: the standard analysis bundle: metrics, PCA compression, basins, failure rows, plots, dashboard export, and report.
- `branching/`: endpoint basin clustering used by `analysis.py`.
- `visualize/` and `dashboard/`: Plotly trajectory figures plus the Streamlit workbench.
- `cli/`: thin argument parsing.

The canonical run layout is:

```text
experiments/runs/<name>/
  trajectories.jsonl
  metrics/
    geometry.jsonl
    alignment.jsonl
  trajectory.html
  compression.jsonl
  basins.json
  failures.jsonl        # only when both correct and incorrect labels exist
  dashboard.html
  report.md
```

Preferred commands:

```bash
rt run --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30 --layer 32
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
rt dashboard --input experiments/runs/r1_distill_sheep30
```
