# Minimal Trajectory Run Guide: R1-Distill Sheep30

Example to understand how to run this framework on a model and gather its trajectories for analysis.

## Goal

Run `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` thirty times on the same simple
question, collect hidden-state trajectories, infer answer correctness, and open
the dashboard.

Question:

```text
Solve step by step: A farmer has 17 sheep. All but 9 run away. How many sheep are left?
```

Expected answer: `9`

## Config

The dedicated config file is:

```bash
experiments/configs/r1_distill_sheep30.yaml
```

It sets:

- model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` # huggingface model to run
- prompt count: `1` # only one question, repeated thirty times with different seeds
- seeds: `0..29` # thirty runs with different random seeds
- temperature: `0.8`
- layers: `0, 8, 16, 24, 32` # five layers of hidden states to extract 
- output tokens: `160` # compact enough for fast repeated extraction on the sheep question
- answer key: `9` # used to label final correctness

## Local Sanity Check

```bash
python3 -m compileall toolkit/reasoning_trajectory toolkit/tests/smoke experiments/scripts/label_trajectory_tails.py
bash -n experiments/scripts/run_remote_gpu.sh
python3 toolkit/tests/smoke/test_answer_labels.py
python3 toolkit/tests/smoke/test_synthetic_curves.py
python3 toolkit/tests/smoke/test_toolkit.py
bash toolkit/tests/smoke/validate_cli.sh
```

This verifies the Python package, CLI, mock extraction, metrics, visualization,
dashboard export, verifiers, and artifact writers before using a lamsade GPU.

## Remote GPU Run

Create the local environment file once:

```bash
cp .env.example .env
$EDITOR .env
```

Set these fields in `.env`:

- `SSH_SERVER`: the first SSH hop, for example `lamgate`.
- `GPU_HOST`: the GPU machine reachable from `SSH_SERVER`, for example `ourasi`.
  Leave it empty if `SSH_SERVER` is already the GPU host.
- `REMOTE_REPO_ROOT`: where this repo should live remotely.
- `REMOTE_RUN_ROOT`: parent directory for remote experiment folders.
- `LOCAL_RUN_ROOT`: parent directory for copied-back local experiment folders.
- `REMOTE_RT_BIN`: remote `rt` executable, for example `/tmp/rt-r1-venv/bin/rt`.
- `HF_TOKEN`: Hugging Face token. `.env` is ignored by git; do not commit it.
- `CUDA_VISIBLE_DEVICES` and the HF/Transformers flags.

Then launch the experiment from the repo root:

```bash
experiments/scripts/run_remote_gpu.sh \
  --config experiments/configs/r1_distill_sheep30.yaml \
  --name r1_distill_sheep30 \
  --layer 32
```

What the wrapper does:

- syncs the repo to `$SSH_SERVER:$REMOTE_REPO_ROOT`;
- runs `rt run` on `$GPU_HOST` if set, otherwise directly on `$SSH_SERVER`;
- writes the remote run to `$REMOTE_RUN_ROOT/r1_distill_sheep30`;
- copies the result back to `$LOCAL_RUN_ROOT/r1_distill_sheep30`.

Why: `rt run` performs extraction and writes the standard analysis bundle in one
step. Host paths, token, device selection, and HF environment flags live in one
ignored `.env` file instead of being retyped in every SSH command.

## Open The Dashboard

```bash
rt dashboard --input experiments/runs/r1_distill_sheep30
```

Useful static files:

```text
experiments/runs/r1_distill_sheep30/trajectories.jsonl
experiments/runs/r1_distill_sheep30/trajectory.html
experiments/runs/r1_distill_sheep30/dashboard.html
experiments/runs/r1_distill_sheep30/report.md
```

## Rebuild Analysis Only

If labels or visual settings change, rebuild artifacts without rerunning the
model:

```bash
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
```
