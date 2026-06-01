# Geometry Metrics

Purpose: export path length, endpoint distance, velocity, acceleration, curvature, torsion, directional consistency, final-state distance, nearest-valid/invalid distance, commitment, drift, repetition, and branch entropy.
Inputs: trajectory JSON/JSONL/run directory.
Outputs: JSONL or Parquet table.
CLI: `rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics`
Python API: `reasoning_trajectory.metrics.geometry.trajectory_geometry`.
Example: `rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics`
Notes: synthetic curves are covered by smoke tests.
Failure modes: trajectories without hidden states raise a clear error.
