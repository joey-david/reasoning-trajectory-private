# Reasoning Trajectory

Tools for generating many reasoning traces, capturing their hidden states, and
studying how reasoning evolves through latent space.

[Hypotheses](hypotheses.md) | [Experiment plan](experiments_plan.md)

## Pipeline

1. Define a self-contained run with `config.yaml` and, optionally, a pinned
   `dataset.jsonl`.
2. Generate sampled reasoning traces and hidden-state artifacts.
3. Analyze correctness, token trajectories, reasoning steps, and clusters
   without loading the model again.
4. Explore the results in the static web interface.

## Repository Structure

```text
runs/                   # experiment configs and artifacts
scripts/
  data/                 # dataset inspection and preparation
  generation/           # generation and remote orchestration entry points
  analysis/             # analysis and screening entry points
  experiments/          # hypothesis-specific analysis and intervention commands
src/
  datasets/             # dataset loaders and adapters
  prompting/            # prompt construction
  models/               # model loading and generation
  runtime/              # configs, paths, and artifact I/O
  orchestration/        # persistent GPU workers and task scheduling
  analysis/             # trajectory analysis
  experiments/          # symbolic updates, probes, contrastive learning, patching
web/                    # static results interface
lit/literature/         # local paper corpus and notes
```

## Quick Start

The full requirements target a CUDA 12.4 generation host:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the pipeline on any folder containing a `config.yaml`:

```bash
# Optional: pin the exact normalized dataset rows.
.venv/bin/python scripts/data/prepare_dataset.py runs/<model>/<experiment>

# Generate or resume sampled traces.
.venv/bin/python scripts/generation/generate.py runs/<model>/<experiment>

# Analyze completed generation artifacts.
.venv/bin/python scripts/analysis/analyze.py runs/<model>/<experiment>
```

Generation is resumable: existing sample, seed, and temperature combinations
are skipped.

For analysis-only work on macOS, install the lightweight dependencies instead
of the CUDA requirements:

```bash
uv pip install --python .venv/bin/python \
  "pyyaml>=6" "numpy>=1.24" "datasets>=2.19" \
  "matplotlib>=3.8" "scikit-learn>=1.4"
```

## Run Contract

Each experiment is self-contained:

```text
runs/<model>/<experiment>/
  config.yaml
  dataset.jsonl             # optional pinned normalized dataset
  generation/
    metadata.json
    generations.jsonl
    samples/
    hidden_states/          # present when capture is enabled
  analysis/
```

`config.yaml` controls the model, dataset, sampling, prompting, activation
capture, and analysis. Use `capture.enabled: false` for inexpensive screening
and `capture.layers: [-1]` to store final-layer states.

Analysis reads completed generation artifacts only. It writes correctness
labels, answer parsing, hard-question rankings, token projections, step-level
features, clusters, and browser-ready manifests under `analysis/`.

## Explore Results

Serve the repository root after analysis:

```bash
python3 -m http.server 8765
```

Open <http://localhost:8765/web/index.html> to inspect run summaries,
generations, and interactive token- or step-level latent trajectories.

## Multi-GPU and Remote Runs

For one local model replica per GPU, assign a list in the run config:

```yaml
model:
  device_map:
    "": [0, 1]
```

For dynamic scheduling across remote nodes:

```bash
.venv/bin/python scripts/orchestrate.py --job generation \
  --nodes kaisertrot boldeagle \
  --devices 0,1 1 \
  --run runs/<model>/<experiment>
```

Orchestrable jobs live in `src/orchestration/jobs/<name>.py`. Each exports
`pending_tasks`, `setup_worker`, and `log_path`; its persistent worker implements
`run_task(...) -> TaskResult`. Tasks must be JSON-serializable and outputs must
be resumable with locked writes.

Use `--nodes local --devices 0,1` to schedule across GPUs on the current host.

Sync configs and pinned datasets to the server, then pull completed artifacts:

```bash
./scripts/remote.sh push
./scripts/remote.sh pull runs/<model>/<experiment>
```

Experiment-specific remote sequences and fallback gates are kept in
[experiments_plan.md](experiments_plan.md).

Remote generation is never required for local analysis.
