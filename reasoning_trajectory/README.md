# Reasoning Trajectory

Reusable analysis for token-level hidden states captured during LLM generation.
This directory is deliberately independent of `src/experiments`,
`src/orchestration`, and project-specific datasets so it can be extracted as a
small package later.

## Input

The package reads a completed run:

```text
run/
  config.yaml
  generation/
    generations.jsonl
    samples/*.json
    hidden_states/*.npz
```

Each generation row identifies its activation artifact with
`hidden_states_file`. Arrays use shape `[tokens, layers, hidden]` and include
`layer_indices`. Float arrays and the repository's symmetric int8 encoding are
supported.

## Use

```bash
rt-analyze runs/<model>/<run>
```

or:

```python
from reasoning_trajectory import analyze_trajectories

report = analyze_trajectories(
    "runs/<model>/<run>",
    {"max_trajectories": 80, "max_tokens_per_trajectory": 64},
)
```

The command writes `analysis/trajectory_metrics.json`. It includes:

- original-space path geometry;
- DTW, Frechet, cosine-path, Procrustes, CKA, and RSA alignment;
- shared PCA compression curves;
- endpoint basin size, entropy, and silhouette;
- nearest-correct failure divergence;
- projection trustworthiness warnings.

`reasoning_trajectory.steps` also exposes structured newline, numbered, proof,
and code-span parsing plus mean, last-token, max, and attention-weighted pooling.

Caps are explicit because pairwise alignment is quadratic. The static interface
renders every report under its **Diagnostics** view.

The geometry, alignment, compression, basin, failure-autopsy, projection
diagnostics, and step parsing from the Sun et al. reference checkout are
integrated here. Its duplicate Hugging Face generator, Streamlit dashboard, and
Lean/Python/SMT verifier wrappers are intentionally not copied: generation is
already owned by `src/models/`, the static workspace replaces Streamlit, and
task verifiers are experiment concerns rather than latent-trajectory analysis.
