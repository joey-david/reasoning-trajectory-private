# Reasoning Trajectory Toolkit

Install from the repository root with:

```bash
python3 -m pip install -e ./toolkit
```

The main commands are:

```bash
rt run --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30 --layer 32
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
rt dashboard --input experiments/runs/r1_distill_sheep30
```
