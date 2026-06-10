# Reasoning Trajectory

Small, YAML-driven experiments for generating reasoning traces and running simple analysis tools.

## Layout

```text
src/       Small Python modules used by scripts.
runs/      One folder per model, then one folder per run with config.yaml.
datasets/  Local datasets in json/jsonl formats.
scripts/   Thin command-line entry points.
lit/       Papers, notes, and the imported reasoning-trajectory reference repo.
```

Run folders are named like:

```text
runs/deepseekR1DistillLlama8b/sheep_08_06_2026/config.yaml
```

## Install

Install dependencies with `uv`:

```bash
uv venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Run

Generate model outputs and save token-level activations:

```bash
python3 scripts/generate.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
```

Run analysis tools:

```bash
python3 scripts/generation_summary.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
python3 scripts/activation_norms.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
```

Outputs stay inside the selected run folder:

```text
generation/generations.jsonl
generation/activations/*.npz
analysis/generation_summary.csv
analysis/activation_norms.csv
```

## Extending

Add new rollout behavior in `src/generation`, dataset parsing in `src/data`, and analysis tools in `src/analysis`.
Expose each tool with a short script in `scripts/` that takes a run path and reads its `config.yaml`.
