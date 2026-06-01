# Reasoning Trajectory

This repository is now organized around three folders:

```text
literature/   Zotero export, notes, and paper-reading material.
experiments/  Configs, runs, legacy experiment scripts, and generated artifacts.
toolkit/      Installable `reasoning_trajectory` Python package, CLI, dashboard, docs, and smoke tests.
```

## Install

```bash
python3 -m pip install -e ./toolkit
rt doctor
```

## Main Example

The canonical checked-in run config is:

```bash
experiments/configs/r1_distill_sheep30.yaml
```

It runs `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` thirty times on:

```text
Solve step by step: A farmer has 17 sheep. All but 9 run away. How many sheep are left?
```

Generate trajectories and artifacts:

```bash
rt run \
  --config experiments/configs/r1_distill_sheep30.yaml \
  --out experiments/runs/r1_distill_sheep30 \
  --layer 32
```

Run the same experiment on the configured GPU host:

```bash
cp .env.example .env
$EDITOR .env
experiments/scripts/run_remote_gpu.sh \
  --config experiments/configs/r1_distill_sheep30.yaml \
  --name r1_distill_sheep30 \
  --layer 32
```

Rebuild analysis without rerunning the model:

```bash
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
```

Open the dashboard:

```bash
rt dashboard --input experiments/runs/r1_distill_sheep30
```

## Literature

Import Zotero into reusable local files:

```bash
literature/import_zotero.py --db ~/Zotero/zotero.sqlite --out literature/zotero
```

The importer is read-only and works while Zotero is open by using SQLite
read-only/no-lock access. It writes:

- `literature/zotero/items.jsonl`
- `literature/zotero/standalone_notes.jsonl`
- `literature/zotero/notes.md`
- `literature/zotero/index.md`

## Validation

```bash
python3 -m compileall toolkit/reasoning_trajectory toolkit/tests/smoke experiments/scripts/label_trajectory_tails.py literature/import_zotero.py
bash -n experiments/scripts/run_remote_gpu.sh
python3 toolkit/tests/smoke/test_answer_labels.py
python3 toolkit/tests/smoke/test_synthetic_curves.py
python3 toolkit/tests/smoke/test_toolkit.py
bash toolkit/tests/smoke/validate_cli.sh
```

More toolkit details live in `toolkit/docs/`. The short run walkthrough is
`experiments/guide.md`.
