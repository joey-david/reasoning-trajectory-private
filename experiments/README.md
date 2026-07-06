# Experiment Reproduction

Each experiment is independent. Its public command lives in
`scripts/experiments/`; its implementation lives in `src/experiments/`; model
and dataset settings live in the relevant `runs/.../config.yaml`.

Completed run artifacts are inputs, not generated source. Local analysis never
loads the model and may be rerun without repeating inference.

## Canonical Analyses

These commands reproduce the reported analyses from completed artifacts:

| Experiment | Command | Main report |
| --- | --- | --- |
| H1: prompted and natural boundaries | `scripts/experiments/boundary_comparison.py` | `runs/SmolLM3-3B/h1_freeform_replay/analysis/experiments/h1_boundaries/report.json` |
| H2: localized symbolic updates | `scripts/experiments/localized_updates.py` | `runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/report.json` |
| H4: structural projection | `scripts/experiments/structural_contrast.py` | `runs/SmolLM3-3B/h4_structural_replay/analysis/experiments/h4_structural_contrast/report.json` |
| H5: correctness prediction | `scripts/experiments/correctness_prediction.py` | `runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h5_correctness_prediction/report.json` |
| Token-level no-free-lunch | `scripts/experiments/token_segmentation.py` | `runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation*/report.json` |
| Judged semantic boundaries | `scripts/experiments/semantic_token_segmentation.py` | `runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation*/report.json` |
| H3: causal process-isomer patching | `scripts/experiments/run_h3_protocol.sh` | `runs/SmolLM3-3B/failed_hypotheses/h3_process_isomer_patching/analysis/report.json` |

Run a canonical local analysis with no arguments:

```bash
.venv/bin/python scripts/experiments/localized_updates.py
.venv/bin/python scripts/experiments/correctness_prediction.py
.venv/bin/python scripts/experiments/token_segmentation.py
.venv/bin/python scripts/experiments/semantic_token_segmentation.py
```

The token scripts run the prespecified minimum-segment sweep of 1, 4, and 8
tokens. Pass `--min-segment-tokens N` to run only one granularity. Every command
accepts explicit artifact paths for reuse on a different model or corpus; see
`--help`.

## H1: Boundaries

Inputs are the four prompting-condition runs:

```text
runs/SmolLM3-3B/h1_freeform_replay
runs/SmolLM3-3B/h1_numbered_steps_pilot
runs/SmolLM3-3B/h1_sentence_separated_pilot
runs/SmolLM3-3B/h1_paragraph_separated_pilot
```

Capture the freeform replay with `replay_capture.py`; generate the three
prompted conditions from their run configs. Then analyze:

```bash
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h1_freeform_replay
.venv/bin/python scripts/experiments/boundary_comparison.py
```

The paragraph run currently contains 21 of 60 planned traces. Existing reports
preserve that interrupted pilot rather than silently imputing missing traces.

## H2 and H4: Updates and Operator Structure

H2 reads the primary 580-trace activation corpus:

```bash
.venv/bin/python scripts/experiments/localized_updates.py
```

H4 uses a separate 300-question teacher-forced replay. Rebuild its activations,
extract its symbolic updates, then train the structural projection:

```bash
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h4_structural_replay
.venv/bin/python scripts/experiments/localized_updates.py \
  runs/SmolLM3-3B/h4_structural_replay --per-sample 5
.venv/bin/python scripts/experiments/structural_contrast.py
```

## H5 and Token-Level Segmentation

H5 needs only the primary corpus. Token segmentation additionally needs
teacher-forced gold-answer states:

```bash
.venv/bin/python scripts/orchestrate.py \
  --job gold_answer_capture --nodes local --devices 0 \
  --run runs/SmolLM3-3B/thought_units_gold_answers
.venv/bin/python scripts/experiments/correctness_prediction.py
.venv/bin/python scripts/experiments/token_segmentation.py
```

Replace `--nodes local --devices 0` with the desired orchestration workers for
remote capture.

## Judged Semantic Boundaries

Prepare overlapping labeling windows from the primary corpus and H2 updates:

```bash
.venv/bin/python scripts/experiments/prepare_solution_object_labels.py
```

Smoke-test one task, run the resumable two-GPU labeling queue, then analyze:

```bash
.venv/bin/python scripts/orchestrate.py \
  --job solution_object_labeling_smoke \
  --nodes upnquick --devices 0+1 \
  --run runs/Qwen3.5-122B-A10B-FP8/solution_object_silver
.venv/bin/python scripts/orchestrate.py \
  --job solution_object_labeling \
  --nodes upnquick --devices 0+1 \
  --run runs/Qwen3.5-122B-A10B-FP8/solution_object_silver
.venv/bin/python scripts/experiments/semantic_token_segmentation.py
```

The Qwen checkpoint path is cluster-specific and is declared in the labeling
run's `config.yaml`.

## H3: Causal Patching

H3 has independently resumable pair-mining, replay, projection, patching, and
analysis stages. Rebuild the canonical pair manifest and component projections:

```bash
.venv/bin/python scripts/experiments/mine_process_isomers.py
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h2_component_replay
.venv/bin/python scripts/experiments/component_localization.py
.venv/bin/python scripts/experiments/component_projection.py
```

The exact primary and fallback patch definitions are pinned in their run
configs. `run_h3_protocol.sh primary` validates inputs, runs a smoke gate,
resumes patching, and analyzes the primary attention-18 condition:

```bash
H3_DEVICES=0,1 scripts/experiments/run_h3_protocol.sh primary
```

Use `fallback` for the prespecified MLP-18 fallback. The component and
process-isomer preparation commands remain separate because they produce
reusable artifacts. Analysis can also be rerun without inference:

```bash
.venv/bin/python scripts/experiments/analyze_causal_patching.py \
  runs/SmolLM3-3B/failed_hypotheses/h3_process_isomer_patching
```

## Artifact Contract

- `config.yaml` pins model, dataset, capture, and intervention settings.
- `generation/generations.jsonl` stores completed traces.
- `generation/hidden_states/*.npz` stores captured activations.
- `analysis/experiments/<experiment>/report.json` is the metric source of truth.
- Analysis commands overwrite only their own report directory.
- Remote jobs are resumable and should be synchronized with
  `scripts/remote.sh`; do not commit large activation artifacts.
