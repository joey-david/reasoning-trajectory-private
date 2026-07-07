# Reasoning Trajectory

Capture LLM hidden states, analyze token-level reasoning paths, and inspect them
in a static browser workspace.

## What Is Reusable

`reasoning_trajectory/` is the standalone analysis package. It owns artifact
reading, token and step segmentation, latent projections, path geometry,
alignment, compression, endpoint basins, failure divergence, and browser
payloads. It does not import the repository's experiments or orchestration.

`src/experiments/` contains this project's hypotheses, probes, interventions,
and symbolic labeling. Those are clients of the package, not part of it.

```text
reasoning_trajectory/   reusable latent-trajectory package
src/
  models/               model loading and activation capture
  runtime/              run configuration and artifact writing
  orchestration/        local and remote generation workers
  experiments/          project-specific hypotheses
scripts/
  generation/           generation entry points
  analysis/             post-processing entry points
  experiments/          experiment entry points
runs/                   self-contained configs and artifacts grouped by purpose
web/                    static analysis workspace
```

## Run Contract

```text
runs/<model>/<purpose>/<run>/
  config.yaml
  dataset.jsonl                 # optional pinned input
  generation/
    generations.jsonl
    samples/*.json
    hidden_states/*.npz
  analysis/
```

Generation is resumable. Analysis reads completed artifacts and never requires
loading the model again.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Optional: pin normalized dataset rows.
.venv/bin/python scripts/data/prepare_dataset.py runs/<model>/<purpose>/<run>

# Generate or resume traces.
.venv/bin/python scripts/generation/generate.py runs/<model>/<purpose>/<run>

# Score, segment, project, and compute trajectory diagnostics.
.venv/bin/python scripts/analysis/analyze.py runs/<model>/<purpose>/<run>

# Run only the reusable bounded metric bundle.
rt-analyze runs/<model>/<purpose>/<run>
```

The metric bundle writes `analysis/trajectory_metrics.json` with original-space
geometry, path alignment, PCA compression, endpoint basins, failure divergence,
and projection-fidelity warnings.

## Explore

```bash
python3 -m http.server 8765
```

Open <http://localhost:8765/web/index.html>. The workspace provides run
comparison, transcript inspection, interactive token/step projections, and
metric diagnostics. Projection trustworthiness is shown explicitly; a visually
clean 3D plot is not treated as scientific evidence by itself.

## Remote Generation

```bash
.venv/bin/python scripts/orchestrate.py --job generation \
  --nodes kaisertrot coktailjet \
  --devices 0,1 1 \
  --run runs/<model>/<purpose>/<run>
```

Commas create independent workers; `0+1` gives one worker two GPUs.
Orchestrable jobs implement `pending_tasks`, `setup_worker`, and `log_path`
under `src/orchestration/jobs/`.

```bash
./scripts/remote.sh push
./scripts/remote.sh pull runs/<model>/<purpose>/<run>
```

Hypothesis-specific commands remain in
[experiments/README.md](experiments/README.md). Canonical experiment status,
headline results, and durable run/report paths are indexed in
[experiments/results.md](experiments/results.md). The reusable package interface
is documented in
[reasoning_trajectory/README.md](reasoning_trajectory/README.md).

## Solution-object extraction

The controlled A-H extraction protocol has separate local-small and GPU-medium
run folders. The medium command prepares nothing implicitly outside its run
contract and does not launch remote work:

```bash
.venv/bin/python scripts/experiments/solution_object_extraction.py run \
  runs/SmolLM3-3B/interventions/solution_object_extraction_medium
```

It writes under
`analysis/experiments/solution_object_extraction/` and reuses the completed
mixed-success GSM-Symbolic activation corpus for the real trajectory and
reranking stages. See [experiments/README.md](experiments/README.md) for staged
commands and the preflight check.

The retrieval/causal follow-up has a bounded single-node runner that uses GPU
index 0 and reuses completed upstream artifacts:

```bash
scripts/experiments/run_solution_object_improvement_remote.sh
```

It sweeps dimensions, layers, token/multi-layer scopes, nonlinear encoders, a
separate causal writer, matched ablations, and a targeted low-leakage ablation
grid, then validates all medium artifacts within a hard 12-hour wall-clock
limit.
