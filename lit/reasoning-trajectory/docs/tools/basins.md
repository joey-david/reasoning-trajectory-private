# Basins

Purpose: cluster sampled trajectories, estimate basin sizes, branch entropy, and branch-tree structure.
Inputs: trajectory bank with hidden states.
Outputs: JSON basin summary.
CLI: `rt basins --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/basins.json`
Python API: `reasoning_trajectory.branching.export_basins`.
Example: `rt basins --input experiments/runs/r1_distill_sheep30 --out experiments/runs/r1_distill_sheep30/basins.json --clusters 2`
Notes: endpoint clustering is the default compact basin estimate.
Failure modes: requires at least one hidden-state endpoint.
