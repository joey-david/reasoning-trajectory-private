# Step Classification

This analysis treats a reasoning step as a span of generated text, maps that span
back onto captured hidden states, and represents the step by aggregate latent
features.

## What It Computes

For each generated trajectory with hidden states, the step classifier writes:

- `analysis/step_classification/layer<L>_vectors.npz`
  - `mean_vectors`: average hidden state across tokens in each step.
  - `direction_vectors`: last token state minus first token state inside the step.
  - `variance`: mean per-dimension latent variance across tokens in the step.
  - `cluster_id`: unsupervised cluster assignment.
- `analysis/step_classification/layer<L>_steps.jsonl`
  - Metadata, text spans, token spans, variance/direction norms, cluster IDs.
- `analysis/step_classification/layer<L>_clusters.json`
  - Cluster summaries and nearest text exemplars.
- `analysis/step_classification/*_steps.json`
  - Interactive PCA/t-SNE payloads for `web/index.html`.
The current segmenters are sentence, pairs of sentences, and paragraphs. Sentence
groups are important because a single sentence can be too small: a reasoning
move often has a setup sentence and an operation sentence.

`layer<L>_steps.jsonl` contains `feature_row`, which indexes the corresponding
row in `layer<L>_vectors.npz`. The full previous-step nudge vector is intentionally
not persisted because it is recovered exactly by subtracting consecutive mean
vectors for the same trajectory and segmenter.

## How To Read The Website

Start the local server from the repo root:

```bash
python3 -m http.server 8765
```

Open:

```text
http://localhost:8765/web/index.html
```

In the Analysis tab:

- Choose a `STEP PCA` or `STEP TSNE` plot.
- Use `Step selector` to switch sentence, sentence pairs, or paragraph steps.
- Use `Color by: cluster` to inspect unsupervised step groups.
- Use `Cluster` to isolate one group and read hover examples.
- Use `Question`, `Sub-run`, and token sliders to restrict the visible trajectory.

Good clusters should have repeated local behavior: similar textual moves,
similar variance scale, and similar direction/nudge norms. A cluster is not a
semantic label yet; use exemplars to name it, then train probes later.

## Why Coding Data Next

The current `gsm_symbolic_dry` run is 100/100 correct, so it is not a useful
stress test for correctness-colored trajectories. GSM8K and MATH-style subsets
are likely too saturated for Qwen3-14B. The prepared next path uses
`bigcode/bigcodebench-hard`, which is small enough to work with this pipeline
and harder because tasks require practical Python implementations with tests.

Other useful coding datasets:

- `livecodebench/code_generation`: strong live/contamination-aware benchmark,
  but the lite dataset currently relies on legacy dataset scripts under the
  local `datasets` v5 stack.
- `BAAI/TACO` and `codeparrot/apps`: useful competitive-programming sources,
  but also rely on legacy dataset scripts here.
- `PrimeIntellect/verifiable-coding-problems`: promising and verifiable, but
  much heavier to download locally.

## Server Commands

From your Mac, push the repo and run configs:

```bash
bash scripts/remote.sh push runs/Qwen3-14B/bigcodebench_hard_screen
```

On the server:

```bash
cd /home/lamsade/jdavid/reasoning
source .venv/bin/activate
python scripts/prepare_dataset.py runs/Qwen3-14B/bigcodebench_hard_screen
python scripts/generate.py runs/Qwen3-14B/bigcodebench_hard_screen
python scripts/analyze.py runs/Qwen3-14B/bigcodebench_hard_screen
```

Pull and inspect the screen run:

```bash
bash scripts/remote.sh pull runs/Qwen3-14B/bigcodebench_hard_screen
python3 -m http.server 8765
```

For direct latent step classification, use the smaller capture config:

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

Pull it back:

```bash
bash scripts/remote.sh pull runs/Qwen3-14B/bigcodebench_hard_latent
python3 -m http.server 8765
```

The latent config captures the final layer for 40 BigCodeBench-Hard tasks with
three samples each. Increase `dataset.sample_limit` or `generation.num_samples_per_item`
after the first run if disk and runtime look acceptable.

## Probe Path

Do not train a supervised probe yet. First gather enough step examples, inspect
clusters, and assign human-readable labels to repeated clusters. Then use:

```text
analysis/step_classification/layer<L>_vectors.npz
analysis/step_classification/layer<L>_probe_examples.jsonl
```

as the feature matrix and label index. A first useful probe is a linear classifier
from `mean_vectors`, `direction_vectors`, derived previous-step nudge vectors,
and scalar variance features to manually labeled step types.
