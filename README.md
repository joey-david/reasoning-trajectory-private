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

## Recommended Next Runs

The Qwen3-14B AIME 2024 pilot is unsuitable: its first five generations all
hit the 8,192-token cap. The next no-capture screens use Qwen3-8B in explicit
thinking mode on open-ended math and code-construction tasks:

```text
runs/Qwen3-8B/polymath_medium_numeric_screen
runs/Qwen3-8B/mbppplus_codegen_screen
runs/Qwen3-8B/bigcodebench_hard_codegen_screen
```

Each run contains 20 questions with ten sampled answers. PolyMath is restricted
to scalar numeric answers. Its medium-difficulty replacement has a 3,072-token
cap; the retained high-difficulty pilot is marked unsuitable because all 22
pulled generations hit 4,096 tokens. MBPP+ and BigCodeBench-Hard require the
model to construct complete Python implementations rather than predict an
existing program's output. Their generation caps are 2,048 and 3,072 tokens,
respectively. All three disable activation capture.

PolyMath can be analyzed directly. Generated programs must first be graded
against their benchmark tests in an isolated evaluation environment; do not
classify a code run from answer-string matching. After test results have been
imported into `generation/generations.jsonl`, update the dataset/model ledger:

```bash
python scripts/summarize_screening.py \
  runs/Qwen3-8B/polymath_medium_numeric_screen \
  runs/Qwen3-8B/mbppplus_codegen_screen \
  runs/Qwen3-8B/bigcodebench_hard_codegen_screen
```

The result is written to `experiments/dataset_saturation.csv`. See
`experiments_plan.md` for the selection gate and the full experiment program.
Each summarized run also gets `analysis/mixed_samples.csv`, sorted by sample ID,
with pass rate, cap count, and mixed/frontier flags.
Multiple-choice and program-output-prediction datasets are intentionally
excluded from the primary solution-object corpus.

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

The pulled DeepSeek-7B and Qwen3-4B math screens are cap-distorted, while
SmolLM3-3B on GSM-Symbolic-P1 produced clean frontier cases. Because forcing
generation to begin with `<think> Okay, let's see` changes model performance,
the next run rescreens the full 100-question P1 pool under that regime. It uses
ten rollouts per question and disables activation capture:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen
```

This wrapper uses a node-local cache under `/tmp/$USER/huggingface`, disables
Xet downloads, raises Hub timeouts, and checks the model CDN over IPv4 before
starting generation. Set `HF_LOCAL_CACHE` to override the cache directory.

Pull completed runs after generation. With no run paths, this pulls every local
run folder that has a `config.yaml`:

```bash
bash scripts/remote.sh pull
```

After pulling, summarize the forced-prefix screen alongside the original
regime:

```bash
python scripts/summarize_screening.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_screen \
  runs/SmolLM3-3B/gsm_symbolic_p1_frontier_expand \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen
```

The forced-prefix screen reached 85.7% accuracy, with 35 mixed questions and
129/1000 capped generations. Fourteen mixed questions had a pass rate strictly
below 80%; every one had at least one capped rollout. They were isolated with:

```bash
python scripts/select_mixed_samples.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen \
  --max-pass-rate 0.8 \
  --out runs/SmolLM3-3B/gsm_symb_prefixed_mixed/dataset.jsonl
```

The first latent run repeated those 14 questions ten times with an 8192-token
cap and final-layer capture:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed
```

After pulling it, build the interactive trajectory and step plots and serve
the website:

```bash
python scripts/analyze.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed
python3 -m http.server 8765
```

Open `http://localhost:8765/web/index.html`.

That run produced 10 mixed questions and still capped 12/140 generations. The
next run pins those 10 mixed questions, restores the 4096-token primary cap,
and finalizes only capped generations by appending:

```text
</think> Given what I've assessed, the answer is
Answer:
```

The model then generates between three and four additional tokens:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed_cap4096
```

To identify at least 50 additional mixed/frontier questions, a separate
no-capture screen contains 300 new GSM-Symbolic-P1 instances. It starts at
offset 100 in the same deterministic shuffle, and its pinned dataset has zero
overlap with previously screened SmolLM3 items:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300
```

The run uses two model replicas when GPUs 0 and 1 are available, ten rollouts
per question, the forced reasoning prefix, a 4096-token primary cap, and the
same cap-finalization behavior. To run on one GPU, replace `[0, 1]` with the
single integer `0` or `1`.

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

python scripts/generate.py \
  runs/Qwen3-8B/polymath_medium_numeric_screen \
  runs/Qwen3-8B/mbppplus_codegen_screen \
  runs/Qwen3-8B/bigcodebench_hard_codegen_screen

python scripts/analyze.py runs/Qwen3-8B/polymath_medium_numeric_screen
```

The datasets are already pinned in each run folder, so the server does not need
Hub access for dataset preparation. Pull completed runs:

```bash
bash scripts/remote.sh pull
```

Grade the code runs with the official benchmark harnesses in an isolated
environment before running the summary command. No inference is required
locally after pulling.
