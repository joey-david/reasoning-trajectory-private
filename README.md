# Reasoning Trajectory

Research sandbox for LLM reasoning analysis.

## Workflow

1. set up dataset and config.yaml
2. launch generation on an inference server (or locally)
3. pull results locally and start desired analysis tools
4. visualize in website.

### run folder config

The workflow is centered around a modular run architecture.
The runs folder contains 1 subfolder per model, which contains 1 subfolder per experiment (and 1 experiment often = 1 dataset).

They are structured as such.

```text
runs/<model>/<experiment>/
  config.yaml # see [[config.yaml]]
  generation/ # contains generated outputs and activations
  analysis/ # contains the outputs of the analysis tools on the generations
```

Start here:

```bash
source .venv/bin/activate
python scripts/generate.py runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
```

Remote sync is intentionally simple:

```bash
bash scripts/remote.sh push runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
bash scripts/remote.sh pull runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
```

Read [ARCHITECTURE.md](ARCHITECTURE.md) before implementing anything. The Python
files are mostly commented placeholders so you can fill in the pieces yourself.
