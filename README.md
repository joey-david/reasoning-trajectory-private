# Reasoning Trajectory

A small research sandbox for learning how to run LLM reasoning experiments by
hand.

The important unit is a run folder:

```text
runs/<model>/<experiment>/
  config.yaml
  generation/
  analysis/
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
