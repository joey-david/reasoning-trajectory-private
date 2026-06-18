# Next Generations

## What is set up

- `scripts/analyze.py <run>` now post-processes completed run folders without invoking generation.
- `src/analysis/token_selectors.py` defines reusable token marker strategies:
  - `every_n`
  - `sentence_end`
  - `percentiles`
  - `reasoning_boundaries`
  - `first_last`
  - regex before/after windows
- `src/analysis/step_markers.py` writes `analysis/step_markers.json` for comparing marker strategies in the web UI.
- `src/analysis/solution_objects.py` writes `analysis/solution_objects.jsonl` with CoT text, final text, parsed answer, gold answer, numeric values, and latent anchors.
- `src/analysis/hard_questions.py` writes `analysis/hard_questions.jsonl`, ranking screen-run samples by wrong/unknown answers, answer disagreement, and generation length.
- `src/analysis/trajectories.py` writes 3D PCA and t-SNE plots from saved hidden states, with a point cap so local plotting stays usable.
- `web/index.html` can browse run generations, switch step-marker strategies, and show generated analysis plots.

## Prepared Qwen3-14B screen runs

These are first-pass no-latent runs. They are meant to identify hard questions cheaply before a second activation-capture run.

```bash
python scripts/prepare_dataset.py runs/Qwen3-14B/gsm8k_hard_screen
python scripts/prepare_dataset.py runs/Qwen3-14B/gpqa_diamond_hard_screen
python scripts/prepare_dataset.py runs/Qwen3-14B/math_algebra_hard_screen
```

`openai/gsm8k` and `EleutherAI/hendrycks_math` prepared locally without authentication.
`Idavidrein/gpqa` is gated on Hugging Face, so prepare that run only from an
environment with the repo's normal `HF_TOKEN` access.

Then, on the remote machine only when ready:

```bash
python scripts/generate.py runs/Qwen3-14B/gsm8k_hard_screen
python scripts/generate.py runs/Qwen3-14B/gpqa_diamond_hard_screen
python scripts/generate.py runs/Qwen3-14B/math_algebra_hard_screen
```

After pulling results back locally:

```bash
python scripts/analyze.py runs/Qwen3-14B/gsm8k_hard_screen
python scripts/analyze.py runs/Qwen3-14B/gpqa_diamond_hard_screen
python scripts/analyze.py runs/Qwen3-14B/math_algebra_hard_screen
```

Look first at each run's `analysis/hard_questions.jsonl`. Those rows are the intended input for building `runs/Qwen3-14B/hard_latent_template/dataset.jsonl`.

## Second latent run

`runs/Qwen3-14B/hard_latent_template/config.yaml` is configured for:

- `capture.enabled: true`
- `capture.layers: [-1]`
- `activation_storage_dtype: int8_scaled`
- `num_samples_per_item: 10`

Populate its `dataset.jsonl` from the hardest screen-run rows. Keep each row in the normalized shape:

```json
{"id": "source_sample_id", "question": "...", "gold_answer": "...", "source": "screen_run_name", "metadata": {"hardness_score": 0.0}}
```

Then push/run/pull using the existing remote workflow. Do not start this until the screen runs have been reviewed.

## What to look at

- `analysis/hard_questions.jsonl`: samples that are wrong, unparsed, unstable across seeds, or unusually long.
- `analysis/solution_objects.jsonl`: whether answer parsing and solution-object fields match each dataset.
- `analysis/step_markers.json`: whether sentence-end, percentile, or reasoning-boundary markers better align with visible reasoning moves.
- `analysis/plots/*.png`: whether final-layer trajectories separate correct/incorrect runs or reveal consistent transitions near the end of `<think>`.
- `web/index.html`: open this locally and switch the Step markers dropdown while inspecting generations.

## Expected outcome

The screen runs should identify a smaller set of hard but analyzable questions. The second run should capture last-layer hidden states at enough repeated samples per question to map latent trajectory structure and CoT text onto the same solution object.
