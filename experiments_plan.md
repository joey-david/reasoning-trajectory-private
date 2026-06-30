# Experiment Plan

This file is now the live plan for the current corpus, not a history of every
screen that led here. Old model ladders, abandoned code-grading branches, and
speculative Paper 2 work are intentionally removed.

## Current State

Primary corpus:

```text
runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k
```

This is the active dataset for Paper 1 work:

- 58 GSM-Symbolic-P1 questions from the robust mixed-question pool.
- Forced reasoning prefix: `<think> Okay, let's see`.
- 10,000-token cap with forced final-answer fallback.
- Final-layer activation capture enabled with `int8_scaled` storage.
- Analysis artifacts exist, including token projections, step markers,
  solution-object records, and step-classification vectors/clusters.

Current analyzed summary:

- 631 scored rollouts across 58 questions.
- 71.95% aggregate accuracy.
- 47 mixed questions.
- 36 frontier questions.
- 11 capped rollouts, or 1.74%.
- 0 unscored rollouts.

The run has more rollouts than the configured 580 expected rows because some
questions have extra completed samples. Treat this as a corpus hygiene issue to
resolve before reporting headline statistics, not as a blocker for exploratory
analysis.

## Screening Evidence To Keep

The useful screening result is already distilled into:

```text
experiments/mixed_question_inventory.jsonl
experiments/mixed_question_inventory.csv
experiments/dataset_saturation.csv
```

Keep these facts:

- The merged mixed inventory has 65 questions.
- 58 questions remain mixed without relying on capped failures.
- The 58 robust questions define the current latent-capture corpus.
- Earlier Qwen, DeepSeek, PolyMath, MBPP+, BigCodeBench, and AIME screens are
  background calibration only. They are not active experimental branches.

## Active Questions

Focus the next work on Paper 1:

1. Are text boundaries a weak proxy for latent reasoning boundaries?
2. Do step-level transition features separate correct and incorrect traces?
3. Do sentence, sentence-pair, and paragraph segmenters produce stable and
   interpretable update clusters?
4. Which step features carry signal: mean state, within-step direction,
   variance, nudge, or cluster identity?

Do not start Paper 2 object-edit labeling, hidden-information probes, compressed
control-sequence experiments, or steering interventions until the Paper 1 signal
is clear.

## Next Steps

1. Resolve corpus hygiene.
   - Decide whether to keep all 631 analyzed rollouts or downsample to exactly
     10 rollouts per question.
   - If downsampling, do it by deterministic seed order and rerun analysis.
   - Record the final corpus row count before using any metric in a writeup.

2. Audit answer and cap behavior.
   - Inspect every capped rollout.
   - Spot-check a small random sample of correct and incorrect parsed answers.
   - Confirm that mixed/frontier labels are not created by parser mistakes.

3. Run the first Paper 1 analysis pass.
   - Compare sentence, sentence-pair, and paragraph step segmenters.
   - Report cluster composition by correctness and question.
   - Compare feature families already written by `step_classification`:
     mean, direction, variance, nudge, and cluster.
   - Prefer question-disjoint validation for any probe.

4. Decide the next experimental branch.
   - If the current corpus gives a clean signal, freeze it and write the Paper 1
     result around endogenous step quality and correctness prediction.
   - If the signal is weak, add one new focused screen rather than reviving the
     old broad model/dataset ladder.

## Useful Commands

Summarize the current corpus:

```bash
.venv/bin/python scripts/analysis/summarize_screening.py \
  runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k
```

Rerun analysis after any corpus cleanup:

```bash
.venv/bin/python scripts/analysis/analyze.py \
  runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k
```

Inspect the static UI:

```bash
python3 -m http.server 8765
```

Then open:

```text
http://localhost:8765/web/index.html
```

## Removed From The Active Plan

- Qwen3-8B screening commands.
- Smaller-model ladder execution checklists.
- Code benchmark grading as a blocker for the current math corpus.
- DeepSeek/AIME length-cap followups.
- Cap-finalization comparison as a required branch.
- Paper 2 solution-object edit labeling.
- Hidden-information, compression, and steering-intervention phases.
- Long dataset/model reference lists.
