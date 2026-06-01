# Trajectory 3D

Purpose: project hidden trajectories to 2D/3D with diagnostics and arrows.
Inputs: trajectory JSON/JSONL/run directory.
Outputs: HTML, PNG, SVG, or PDF.
CLI: `rt plot --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/traj.html --color-by correctness`
Python API: `reasoning_trajectory.visualize.trajectory_3d.export_trajectory_plot`.
Example: `rt plot --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/traj.html`
Notes: PCA, UMAP, and t-SNE are supported; diagnostics warn when projection
trustworthiness is low. The plot draws one connected line per trajectory; early
steps are lighter, late steps are more saturated, and endpoints are highlighted.
Failure modes: static image export requires `kaleido`; UMAP requires `umap-learn`.
