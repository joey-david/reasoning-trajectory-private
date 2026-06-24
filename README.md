# Reasoning Trajectory

Research tooling for generating, storing, and analyzing language-model reasoning
trajectories.

## Research Hypothesis

We model chain-of-thought not only as textual explanation, but as a sequence of
latent transition operators over an explicit or latent solution object. Step
direction features may predict edits to that solution object, while correct
reasoning traces may show more coherent alignment between latent trajectory
updates and solution-object construction.

## Run Layout

Each experiment is self-contained:

```text
runs/<model>/<experiment>/
  config.yaml
  dataset.jsonl        # optional pinned normalized dataset
  generation/
    metadata.json
    generations.jsonl
    samples/
    hidden_states/     # present when activation capture is enabled
  analysis/
```

Generation outputs stay compact: sample-level prompt and gold fields are stored
once under `generation/samples/`, while `generations.jsonl` stores one row per
sample/seed/temperature and references hidden-state NPZ files.

## Local Setup

The checked-in requirements target the CUDA server environment. For local
analysis on macOS, install only the analysis dependencies into the existing
venv:

```bash
uv pip install --python .venv/bin/python \
  "pyyaml>=6" "numpy>=1.24" "datasets>=2.19" \
  "matplotlib>=3.8" "scikit-learn>=1.4"
```

## Prepare Datasets

Run configs can load Hugging Face or JSONL datasets directly. To pin the exact
normalized rows before generation:

```bash
python scripts/prepare_dataset.py runs/<model>/<experiment>
```

Dataset loading, normalization, deterministic shuffling, offset, and limit are
shared between preparation and generation, so both paths select identical rows.

## Generate

Run one experiment:

```bash
python scripts/generate.py runs/<model>/<experiment>
```

Run multiple experiments sequentially:

```bash
python scripts/generate.py \
  runs/Qwen3-14B/bigcodebench_hard_screen \
  runs/Qwen3-14B/bigcodebench_hard_latent
```

Existing generation rows are skipped, so restarting a partially completed run
resumes from its normalized generation index.

Activation capture is controlled in `config.yaml`:

```yaml
capture:
  enabled: true
  layers: [-1]
  activation_storage_dtype: int8_scaled
```

Use `layers: [-1]` for final-layer states and `enabled: false` for a cheap
screening run.

## Analyze

Analyze a completed run without loading the model:

```bash
python scripts/analyze.py runs/<model>/<experiment>
```

The analyzer operates only on the run's generation artifacts and produces:

- parsed answers and correctness labels;
- token step-marker alternatives;
- solution-object records;
- hard-question rankings;
- interactive token-level PCA/t-SNE data;
- step-level averaged latent vectors, directions, variance, clusters, and plots.

Static Matplotlib PNGs duplicate the interactive projection work and are
disabled by default. Enable them only when needed:

```yaml
analysis:
  static_plots: true
```

### Step Classification

`src/analysis/step_classification/` aggregates hidden states over sentence,
sentence-pair, and paragraph spans. For each step it keeps:

- the mean hidden state;
- within-step direction (`last_state - first_state`);
- latent variance and direction magnitude;
- the previous-step nudge magnitude;
- an unsupervised cluster and representative step text.

The full nudge vector is not stored because it is exactly recoverable by
subtracting consecutive mean vectors with the same `trajectory_id` and
`segmenter`. Browser projections default to 4,000 evenly sampled steps while
clustering can use up to 12,000.

See [the step-classification guide](src/analysis/step_classification/README.md)
for artifact details, interpretation, and server commands.

## Website

After analysis, serve the repository root:

```bash
python3 -m http.server 8765
```

Open:

```text
http://localhost:8765/web/index.html
```

The Analysis tab supports rotatable/zoomable token and step plots, question and
seed filtering, step granularity, cluster filtering, correctness/cluster color
modes, trajectory limits, and token-position ranges.

Entropy highlighting is available only when generation rows contain timestep
entropy diagnostics. The UI disables the control and reports when a run did not
store them.

## Recommended Next Runs

GSM-Symbolic is saturated in the current Qwen3-14B run (`100/100` correct), so
the prepared coding runs are more useful for heterogeneous step discovery:

```text
runs/Qwen3-14B/bigcodebench_hard_screen
runs/Qwen3-14B/bigcodebench_hard_latent
```

The screen config runs all 148 BigCodeBench-Hard tasks without activation
capture. The latent config runs 40 tasks with three samples each and final-layer
capture for step classification.

## Remote Workflow

Push code and run configs:

```bash
bash scripts/remote.sh push runs/Qwen3-14B/bigcodebench_hard_latent
```

On the server:

```bash
cd /home/lamsade/jdavid/reasoning
source .venv/bin/activate

python scripts/prepare_dataset.py runs/Qwen3-14B/bigcodebench_hard_latent
python scripts/generate.py runs/Qwen3-14B/bigcodebench_hard_latent
python scripts/analyze.py runs/Qwen3-14B/bigcodebench_hard_latent
```

Pull the completed run:

```bash
bash scripts/remote.sh pull runs/Qwen3-14B/bigcodebench_hard_latent
```

No inference is required locally after pulling; rerun `scripts/analyze.py` only
when analysis code or configuration changes.
