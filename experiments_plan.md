# Experiment Plan

The hypotheses are implemented under `src/experiments/`, with thin commands in
`scripts/experiments/`. Heavy inference has not been started.

## Local Results

Source corpus:

```text
runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k
```

All local metrics use exactly 10 seed-sorted trajectories per question: 580
traces across 58 questions. Raw generation data remains unchanged.

- **H2:** 17,602 verified symbolic updates. Update-completion states have mean
  per-trace magnitude AUC `0.642`, but sharp-spike overlap is only `5.6%`
  against a `4.0%` shifted null. Current verdict: elevated change without
  strong sharp localization.
- **H4:** a linear 128-dimensional contrastive projection trained on strict
  lexical controls raises question-disjoint pair AUC from `0.315` to `0.920`.
  This supports learned operator structure, not natural unsupervised clustering.
- **H5:** step mean plus variance reaches ROC-AUC `0.754`, `0.741`, and `0.757`
  at 25%, 50%, and 75%. Its grouped-bootstrap gain over sentence means excludes
  zero at all checkpoints. Latent-spike features do not beat sentence means, so
  LiveCodeBench transfer is gated off for now.

Reports live under:

```text
runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k/analysis/experiments/
```

## GPU Runs

Push the prepared configs and pinned H1 datasets:

```bash
bash scripts/remote.sh push
```

Run these replays in parallel. Their uncompressed activation estimate is
`15.9 GiB`; all four H1 conditions add about `1.3 GiB` at baseline trace
lengths. The total raw estimate is `17.2 GiB`, before int8 compression.

```bash
# A100: 174 traces, layers 9/18/27/-1, residual + MLP + attention.
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h2_component_replay

# A6000: 1,500 existing traces, final-layer residual only.
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h4_structural_replay
```

After the H2 replay, capture diagnostics for the 60 matched existing freeform
traces:

```bash
.venv/bin/python scripts/experiments/replay_capture.py \
  runs/SmolLM3-3B/h1_freeform_replay
```

The prompted H1 pilot is three matched 60-generation runs:

```bash
.venv/bin/python scripts/generation/generate.py \
  runs/SmolLM3-3B/h1_numbered_steps_pilot \
  runs/SmolLM3-3B/h1_sentence_separated_pilot \
  runs/SmolLM3-3B/h1_paragraph_separated_pilot
```

## Post-Run Analysis

```bash
# H2 component localization and automatic H3 patch target.
.venv/bin/python scripts/experiments/component_localization.py \
  runs/SmolLM3-3B/h2_component_replay \
  runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates

# H1 matched comparison, using the existing freeform corpus as control.
.venv/bin/python scripts/experiments/boundary_comparison.py \
  runs/SmolLM3-3B/h1_freeform_replay \
  runs/SmolLM3-3B/h1_numbered_steps_pilot \
  runs/SmolLM3-3B/h1_sentence_separated_pilot \
  runs/SmolLM3-3B/h1_paragraph_separated_pilot

# H4 scale-up.
.venv/bin/python scripts/experiments/localized_updates.py \
  runs/SmolLM3-3B/h4_structural_replay --per-sample 5
.venv/bin/python scripts/experiments/structural_contrast.py \
  runs/SmolLM3-3B/h4_structural_replay/analysis/experiments/h2_localized_updates
```

H3 is intentionally last. Its 30 state-equivalent pairs are already prepared;
the runner reads the H2 component report and patches the recommended component
and layer automatically:

```bash
.venv/bin/python scripts/experiments/causal_patching.py \
  runs/SmolLM3-3B/h3_process_isomer_patching
.venv/bin/python scripts/experiments/analyze_causal_patching.py \
  runs/SmolLM3-3B/h3_process_isomer_patching
```
