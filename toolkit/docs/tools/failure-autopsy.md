# Failure Autopsy

Purpose: find first divergence from the nearest valid path and summarize likely failure labels.
Inputs: trajectory bank with valid and invalid examples.
Outputs: JSONL failure report.
CLI: `rt failures --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/failures.jsonl`
Python API: `reasoning_trajectory.analysis.failure_report`.
Example: `rt failures --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/failures.jsonl`
Notes: labels are conservative heuristics designed to guide manual review.
Failure modes: requires at least one valid reference trajectory.
