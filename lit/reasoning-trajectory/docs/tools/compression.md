# Compression

Purpose: measure how well a low-dimensional PCA bottleneck preserves trajectory structure.
Inputs: trajectories with hidden states.
Outputs: reconstruction-error table.
CLI: `rt compression --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/compression.jsonl --dims 2`
Python API: `reasoning_trajectory.analysis.pca_compress`.
Example: `rt compression --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/compression.jsonl --dims 2`
Notes: intentionally compact; add new bottlenecks in `reasoning_trajectory.analysis` only when an experiment needs them.
Failure modes: requires scikit-learn.
