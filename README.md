# Reasoning Trajectory

Small, YAML-driven experiments for generating reasoning traces and running simple analysis tools.

## Layout

```text
src/       Small Python modules used by scripts.
runs/      One folder per model, then one folder per run with config.yaml.
datasets/  Local datasets in json/jsonl formats.
scripts/   Thin command-line entry points.
lit/       Papers, notes, and the imported reasoning-trajectory reference repo.
docs/      Local implementation notes and visual style guidance.
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

See `GUIDE.md` for the full end-to-end workflow: dataset creation, run config, generation, analysis, and web visualization.

Generate model outputs and save token-level activations:

```bash
python3 scripts/generate.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
```

Run analysis tools:

```bash
python3 scripts/generation_summary.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
python3 scripts/activation_norms.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026
python3 scripts/analyze.py runs/deepseekR1DistillLlama8b/sheep_08_06_2026 --tool all
```

Outputs stay inside the selected run folder:

```text
generation/generations.jsonl
generation/activations/*.npz
analysis/generation_summary.csv
analysis/activation_norms.csv
analysis/trajectory_projection_layer32_i4_pca.json
analysis/pca_components_layer32_n24.json
```

## Web Interface

Start the modular local interface:

```bash
python3 scripts/web.py --port 8765
```

The UI can load existing runs, create a new run from a built-in Hugging Face
selector or custom model id, start generation, run one analysis tool, run all
tools, and visualize existing outputs. Tool visualizers are registered in
`src/analysis/tools.py` and rendered in `src/web_interface/static/app.js`.

For UI and analysis smoke tests without a model download:

```bash
python3 scripts/make_synthetic_run.py
python3 scripts/analyze.py runs/synthetic/half_right_half_wrong --tool all
```

The visual style is documented in `docs/visual_style.md` and is based on the
graph style of arXiv:2605.21488.

## Apple MLX Shim

Set `backend: mlx` in a run config to use the explicitly Apple-only MLX shim in
`src/generation/mlx.py`. This is for local Apple Silicon smoke generation; the
normal Linux/NVIDIA target path remains `backend: hf`. Install MLX separately
when needed:

```bash
uv pip install mlx-lm
```

## Extending

Add new rollout behavior in `src/generation`, dataset parsing in `src/data`, and analysis tools in `src/analysis`.
Expose each tool with a short script in `scripts/` that takes a run path and reads its `config.yaml`.
