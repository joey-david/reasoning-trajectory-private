# Repo Structure

Currently adding analysis tools.

TODO:

- [ ] Divergence as a tradeoff between originality / stability, slowness of channels/attractions
- [ ] Compute dispersion per PCA dimension
	- [ ] Number of significant components?
	- [ ] Vary the number of components
- [ ] Variation in the number of reasoning steps
- [ ] Loss as a CoT metaphor
- [X] Git for papers, ideas, etc.
- [?] Making smaller models' CoT work by tweaking attention
	- [ ] Look into literature on why it doesn't work, shallow world models, possible mitigations, etc.
- [ ] Going toward solution objects: Lean and programs
- [ ] Add better tools for dynamic divergence analysis rather than static resemblance.


```text
literature/   Zotero export, notes, and paper-reading material.
experiments/  Configs, runs, legacy experiment scripts, and generated artifacts.
toolkit/      `reasoning_trajectory` Python package, CLI, dashboard, docs, and smoke tests.
```

## Install

```bash
python3 -m pip install -e ./toolkit
rt doctor
```

## Main Example

I wrote an example config to run:

```bash
experiments/configs/r1_distill_sheep30.yaml
```

If you want to rebuild the analysis without rerunning models

```bash
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
```

Open the dashboard:

```bash
rt dashboard --input experiments/runs/r1_distill_sheep30
```


## Validation

```bash
python3 -m compileall toolkit/reasoning_trajectory toolkit/tests/smoke experiments/scripts/label_trajectory_tails.py literature/import_zotero.py
bash -n experiments/scripts/run_remote_gpu.sh
python3 toolkit/tests/smoke/test_answer_labels.py
python3 toolkit/tests/smoke/test_synthetic_curves.py
python3 toolkit/tests/smoke/test_toolkit.py
bash toolkit/tests/smoke/validate_cli.sh
```