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
python scripts/data/prepare_dataset.py runs/<model>/<experiment>
```

Dataset loading, normalization, deterministic shuffling, offset, and limit are
shared between preparation and generation, so both paths select identical rows.

## Generate

Run one experiment:

```bash
python scripts/generation/generate.py runs/<model>/<experiment>
```

Run multiple experiments sequentially:

```bash
python scripts/generation/generate.py \
  runs/Qwen3-14B/bigcodebench_hard_screen \
  runs/Qwen3-14B/bigcodebench_hard_latent
```

Existing generation rows are skipped, so restarting a partially completed run
resumes from its normalized generation index.

To prefill every assistant response while keeping the prefix inside the stored
generated trace, set:

```yaml
generation:
  forced_prefix: "<think> Okay, let's see"
```

Generation also stops after the configured final-answer pattern is complete.
By default it reuses `analysis.produced_answer_regex`; set
`generation.stop_regex` only when a run needs a different terminal pattern.
Runs may also define `generation.cap_fallback`. If the primary token budget is
exhausted, its prefix is appended to the stored trace and the model receives
the configured small final-token budget.

To replicate one model per GPU and split questions between them, use:

```yaml
model:
  device_map:
    "": [0, 1]
```

This is data parallelism: each GPU receives a contiguous instance shard while
global sample indices keep seeds identical to a single-GPU run. The comma
string `"0,1"` is also accepted. After both workers finish, run analysis once
over the combined artifacts so projections and clusters share one space.

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
python scripts/analysis/analyze.py runs/<model>/<experiment>
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

The interface is static and has no separate backend process. After analysis,
serve the repository root over HTTP:

```bash
python3 -m http.server 8765
```

Open:

```text
http://localhost:8765/web/index.html
```

The interface has three workspaces:

- **Overview** summarizes run accuracy, reasoning lengths, available artifacts,
  and question-level results.
- **Generations** provides search, question/sub-run/outcome filters, step-marker
  inspection, sorting, and paginated reasoning traces.
- **Latent space** provides rotatable and zoomable token/step projections,
  question and sub-run filtering, step granularity and cluster controls,
  correctness/progress or cluster coloring, trajectory limits, token ranges,
  and point inspection.

The active workspace and filters are reflected in the URL, so a filtered view
can be bookmarked or shared. Press `O`, `G`, or `L` to switch workspaces and
`/` to focus the active search field.

Entropy highlighting is available only when generation rows contain timestep
entropy diagnostics. Colors are normalized within each generation; the UI
disables the control and reports when a run did not store them.

### Smaller-Model Screens

Three additional reasoning-capable sizes are prepared, with one math and one
code-construction dataset per model:

```text
runs/SmolLM3-3B/gsm_symbolic_p1_screen
runs/SmolLM3-3B/mbppplus_codegen_screen
runs/Qwen3-4B/polymath_medium_numeric_screen
runs/Qwen3-4B/mbppplus_codegen_screen
runs/DeepSeek-R1-Distill-Qwen-7B/polymath_medium_numeric_screen
runs/DeepSeek-R1-Distill-Qwen-7B/bigcodebench_hard_codegen_screen
```

Every run uses 20 instances and ten sampled rollouts, disables activation
capture, and explicitly asks for clear, concise reasoning that lasts only as
long as needed. MBPP+ uses the same selected tasks at 3B and 4B; PolyMath uses
the same selected tasks at 4B and 7B. See `experiments_plan.md` for the
benchmark evidence behind each pairing.

Pull completed runs after generation. With no run paths, this pulls every local
run folder that has a `config.yaml`:

```bash
bash scripts/remote.sh pull
```

After pulling, summarize the forced-prefix screen alongside the original
regime:

```bash
python scripts/analysis/summarize_screening.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_screen \
  runs/SmolLM3-3B/gsm_symbolic_p1_frontier_expand \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen
```

To identify at least 50 additional mixed/frontier questions, a separate
no-capture screen contains 300 new GSM-Symbolic-P1 instances. It starts at
offset 100 in the same deterministic shuffle, and its pinned dataset has zero
overlap with previously screened SmolLM3 items:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generation/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300
```

## Remote Workflow

Push code, run configs, and pinned datasets. Generation and analysis artifacts
under `runs/*/*/generation/` and `runs/*/*/analysis/` are excluded from push:

```bash
bash scripts/remote.sh push
```

On the server:

```bash
cd /home/lamsade/jdavid/reasoning
source .venv/bin/activate

python scripts/generation/generate.py \
  runs/Qwen3-8B/polymath_medium_numeric_screen \
  runs/Qwen3-8B/mbppplus_codegen_screen \
  runs/Qwen3-8B/bigcodebench_hard_codegen_screen

python scripts/analysis/analyze.py runs/Qwen3-8B/polymath_medium_numeric_screen
```

For dynamic rollout scheduling across several hosts, stop any generator already
using the run and launch one worker per selected GPU from lamgate:

```bash
./scripts/generation/orchestrate.py \
  --nodes kaisertrot boldeagle \
  --devices 0,1 1 \
  --run runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k
```

Workers reject GPUs that do not support the run's configured BF16 dtype.

The datasets are already pinned in each run folder, so the server does not need
Hub access for dataset preparation. Pull completed runs:

```bash
bash scripts/remote.sh pull
```

Grade the code runs with the official benchmark harnesses in an isolated
environment before running the summary command. No inference is required
locally after pulling.
