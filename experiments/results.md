# Experiment Registry

This is the merge-facing index for project-specific experiments. It separates
the durable experiment state from the reproduction notes in `README.md`.

Status meanings:

- `canonical`: current reported analysis.
- `pilot`: useful but incomplete or exploratory.
- `failed`: prespecified or investigated path with a negative result.
- `active`: current follow-up not yet reduced to a final headline.

## Canonical and Active Results

| ID | Status | Question | Primary runs | Command | Main report | Headline |
| --- | --- | --- | --- | --- | --- | --- |
| H1 boundaries | canonical | Do prompted/natural step boundaries explain latent trajectory structure? | `runs/SmolLM3-3B/pilots/h1_freeform_replay`; `runs/SmolLM3-3B/pilots/h1_numbered_steps_pilot`; `runs/SmolLM3-3B/pilots/h1_sentence_separated_pilot`; `runs/SmolLM3-3B/pilots/h1_paragraph_separated_pilot` | `.venv/bin/python scripts/experiments/boundary_comparison.py` | `runs/SmolLM3-3B/pilots/h1_freeform_replay/analysis/experiments/h1_boundaries/report.json` | Prompted boundary conditions are reproducible from completed artifacts; the paragraph condition is an interrupted 21/60 pilot and should be interpreted as such. |
| H2 localized updates | canonical | Are symbolic updates localized in trajectory space? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k` | `.venv/bin/python scripts/experiments/localized_updates.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/report.json` | Primary 580-trace activation corpus for localized symbolic update analysis. |
| H4 structural projection | canonical | Is there reusable operator structure in symbolic-update trajectories? | `runs/SmolLM3-3B/replay/h4_structural_replay` | `.venv/bin/python scripts/experiments/structural_contrast.py` | `runs/SmolLM3-3B/replay/h4_structural_replay/analysis/experiments/h4_structural_contrast/report.json` | Uses a separate teacher-forced replay and H2-style update extraction before structural projection. |
| H5 correctness prediction | canonical | Do trajectory features predict correctness? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k` | `.venv/bin/python scripts/experiments/correctness_prediction.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h5_correctness_prediction/report.json` | Reuses the primary mixed-success corpus without rerunning generation. |
| Token-level no-free-lunch | canonical | Do token-only segmentation rules recover semantic steps? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k`; `runs/SmolLM3-3B/replay/thought_units_gold_answers` | `.venv/bin/python scripts/experiments/token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation*/report.json` | Prespecified minimum-segment sweep over 1, 4, and 8 tokens. |
| Judged semantic boundaries | canonical | Do judged solution-object windows improve boundary evaluation? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k`; `runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver` | `.venv/bin/python scripts/experiments/semantic_token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation*/report.json` | Uses Qwen-labeled overlapping windows prepared from the primary corpus and H2 updates. |
| H3 process-isomer patching | failed | Does causal patching of process-isomer components rescue target behavior? | `runs/SmolLM3-3B/failed/h3_process_isomer_replay`; `runs/SmolLM3-3B/failed/h3_process_isomer_patching`; `runs/SmolLM3-3B/failed/h3_process_isomer_patching_mlp18` | `H3_DEVICES=0,1 scripts/experiments/run_h3_protocol.sh primary` | `runs/SmolLM3-3B/failed/h3_process_isomer_patching/analysis/report.json` | Prespecified attention-18 primary and MLP-18 fallback are kept as failed-hypothesis artifacts. |
| Solution-object extraction | active | Can latent solution objects be extracted, decoded, and used causally? | `runs/SmolLM3-3B/interventions/solution_object_extraction_small`; `runs/SmolLM3-3B/interventions/solution_object_extraction_medium` | `.venv/bin/python scripts/experiments/solution_object_extraction.py run runs/SmolLM3-3B/interventions/solution_object_extraction_medium` | `runs/SmolLM3-3B/interventions/solution_object_extraction_medium/analysis/experiments/solution_object_extraction/` | Medium validation passes the improved artifact contract; retrieval and low-leakage causal evidence are promising, but real mixed-success trajectory G/H remains the next decisive stage. |

## Supporting Manifests

These tracked files are durable inputs or compact summaries, not raw activation
artifacts:

| Path | Role |
| --- | --- |
| `experiments/mixed_question_inventory.jsonl` | Mixed-question inventory for primary corpus summaries. |
| `experiments/mixed_question_inventory.csv` | Tabular companion to the mixed-question inventory. |
| `experiments/dataset_saturation.csv` | Dataset saturation summary. |
| `experiments/h3_process_isomer_pairs.jsonl` | Pinned H3 process-isomer pair manifest. |
| `experiments/h3_process_isomer_pair_audit.json` | H3 pair audit summary. |
| `experiments/h3_projections/attention_output_layer18_report.json` | H3 attention-output layer-18 projection report. |
| `experiments/h3_projections/mlp_output_layer18_report.json` | H3 MLP-output layer-18 projection report. |

## Run Organization Policy

Keep the stable run-folder contract as
`runs/<model>/<purpose>/<run>/config.yaml` plus optional `dataset.jsonl`,
`generation/`, and `analysis/`. The purpose directory should make navigation
obvious without changing the files each run owns:

- `screening` for dataset/model screening and frontier identification.
- `replay` for teacher-forced or artifact-replay corpora.
- `interventions` for causal jobs and controlled intervention protocols.
- `labeling` for external judge or label-generation jobs.
- `pilots` for incomplete local probes.
- `failed` for retained negative-result hypotheses.

Prefer adding rows here with explicit run/report paths when introducing new
canonical experiments.
