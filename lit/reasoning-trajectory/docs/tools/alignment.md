# Alignment

Purpose: compare paths with DTW, Frechet, cosine path similarity, Procrustes, CKA, RSA, prototypes, and first divergence.
Inputs: trajectory JSON/JSONL/run directory.
Outputs: pairwise JSONL or Parquet table.
CLI: `rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics`
Python API: `reasoning_trajectory.metrics.alignment.alignment_summary`.
Example: `rt metrics --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/metrics`
Notes: supports different path lengths.
Failure modes: empty hidden-state paths fail validation.
