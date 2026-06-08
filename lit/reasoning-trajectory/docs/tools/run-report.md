# Run Report

Purpose: export a concise markdown summary of one trajectory run and its derived artifacts.
Inputs: run directory with `trajectories.jsonl` and optional metrics/figures/reports.
Outputs: markdown report.
CLI: `rt report --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/report.md`
Python API: `reasoning_trajectory.analysis.export_run_report`.
Example: `rt run --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30 --layer 32`
Notes: the full pipeline writes `report.md` automatically.
Failure modes: missing trajectory files fail through shared schema IO.
