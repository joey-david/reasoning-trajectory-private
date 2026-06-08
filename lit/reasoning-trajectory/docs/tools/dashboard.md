# Dashboard

Purpose: inspect trajectory banks from one place.
Inputs: run directory or trajectory JSON/JSONL.
Outputs: static HTML or Streamlit app.
CLI: `rt dashboard --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/dashboard.html`
Python API: `reasoning_trajectory.dashboard.app.launch_or_export_dashboard`.
Example: `rt dashboard --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/dashboard.html`

Interactive example: `rt dashboard --input experiments/runs/r1_distill_sheep30`

Without `--out`, the command opens the Streamlit workbench for selecting run
folders/configs, filtering trajectories, previewing generated artifacts, and
rerunning dashboard-safe tools with parameter controls.
Notes: shows run metadata, problem/trajectory lists, correctness, text, metrics, and solution-object metadata.
Failure modes: interactive mode requires Streamlit.
