# Experiment Registry

This is the merge-facing index for project-specific experiments. It separates
the durable experiment state from the reproduction notes in `README.md`.

Status meanings:

- `canonical`: current reported analysis.
- `pilot`: useful but incomplete or exploratory.
- `failed`: prespecified or investigated path with a negative result.
- `active`: current follow-up not yet reduced to a final headline.
- `prepared`: implemented and locally validated, but not yet run on the target GPU.

## Canonical and Active Results

| ID | Status | Question | Primary runs | Command | Main report | Headline |
| --- | --- | --- | --- | --- | --- | --- |
| H1 boundaries | canonical | Do prompted/natural step boundaries explain latent trajectory structure? | `runs/SmolLM3-3B/pilots/h1_freeform_replay`; `runs/SmolLM3-3B/pilots/h1_numbered_steps_pilot`; `runs/SmolLM3-3B/pilots/h1_sentence_separated_pilot`; `runs/SmolLM3-3B/pilots/h1_paragraph_separated_pilot` | `.venv/bin/python scripts/experiments/boundaries/boundary_comparison.py` | `runs/SmolLM3-3B/pilots/h1_freeform_replay/analysis/experiments/h1_boundaries/report.json` | Prompted boundary conditions are reproducible from completed artifacts; the paragraph condition is an interrupted 21/60 pilot and should be interpreted as such. |
| H2 localized updates | canonical | Are symbolic updates localized in trajectory space? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k` | `.venv/bin/python scripts/experiments/trajectory_dynamics/localized_updates.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/report.json` | Primary 580-trace activation corpus for localized symbolic update analysis. |
| H4 structural projection | canonical | Is there reusable operator structure in symbolic-update trajectories? | `runs/SmolLM3-3B/replay/h4_structural_replay` | `.venv/bin/python scripts/experiments/trajectory_dynamics/structural_contrast.py` | `runs/SmolLM3-3B/replay/h4_structural_replay/analysis/experiments/h4_structural_contrast/report.json` | Uses a separate teacher-forced replay and H2-style update extraction before structural projection. |
| H5 correctness prediction | canonical | Do trajectory features predict correctness? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k` | `.venv/bin/python scripts/experiments/trajectory_dynamics/correctness_prediction.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h5_correctness_prediction/report.json` | Reuses the primary mixed-success corpus without rerunning generation. |
| Token-level no-free-lunch | canonical | Do token-only segmentation rules recover semantic steps? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k`; `runs/SmolLM3-3B/replay/thought_units_gold_answers` | `.venv/bin/python scripts/experiments/token_segmentation/token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation*/report.json` | Prespecified minimum-segment sweep over 1, 4, and 8 tokens. |
| Judged semantic boundaries | canonical | Do judged solution-object windows improve boundary evaluation? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k`; `runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver` | `.venv/bin/python scripts/experiments/token_segmentation/semantic_token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation*/report.json` | Uses Qwen-labeled overlapping windows prepared from the primary corpus and H2 updates. |
| Symbolic parser audit | pilot | How faithful is the restricted-AST symbolic-update parser on real trace windows? | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k` | `.venv/bin/python scripts/experiments/symbolic/audit_symbolic_parser.py` | `experiments/symbolic_parser_audit_report.json` | 8-window hand audit: micro-recall improved 0.815→0.963 (+5 FN → 1 FN) by fixing leading-paren capture, adding prose-arithmetic detection (+/-/× as words), BIND-with-expression-RHS extension, currency-symbol support, and unit-word skipping in BIND continuations. Micro-precision 0.88→0.79 reflects 6 semantically-correct extra detections not tracked by gold labels rather than genuine regressions. Full 580-trace corpus shows +1915 updates (+10.9%), led by +953 OPERATE (+16.4%). See `src/experiments/symbolic.py`. |
| H3 process-isomer patching | failed | Does causal patching of process-isomer components rescue target behavior? | `runs/SmolLM3-3B/failed/h3_process_isomer_replay`; `runs/SmolLM3-3B/failed/h3_process_isomer_patching`; `runs/SmolLM3-3B/failed/h3_process_isomer_patching_mlp18` | `H3_DEVICES=0,1 scripts/experiments/run_h3_protocol.sh primary` | `runs/SmolLM3-3B/failed/h3_process_isomer_patching/analysis/report.json` | Prespecified attention-18 primary and MLP-18 fallback are kept as failed-hypothesis artifacts. |
| Solution-object extraction | active | Can latent solution objects be extracted, decoded, and used causally? | `runs/SmolLM3-3B/interventions/solution_object_extraction_small`; `runs/SmolLM3-3B/interventions/solution_object_extraction_medium` | `.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py run runs/SmolLM3-3B/interventions/solution_object_extraction_medium` | `runs/SmolLM3-3B/interventions/solution_object_extraction_medium/analysis/experiments/solution_object_extraction/` | Medium validation passes the improved artifact contract; retrieval and low-leakage causal evidence are promising, but real mixed-success trajectory G/H remains the next decisive stage. |
| State materialization | canonical | Does a history collapse into a reusable current state? | `runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history` | `.venv/bin/python scripts/experiments/depth_relief.py analyze-abstraction-information runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history` | `runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history/depth_relief/state_abstraction/{information_summary,interchange_summary}.json` | Read, Update, and constituent steps are perfect, but h2 Compose is 13.44% despite 76.98% Synthesize; h4 Synthesize and Compose are 12.71%. Explicit state is invariant and decodable, while implicit endpoints retain path information. The tested late-layer state subspace does not beat its random control. |
| Explicit state handoff | active | Does a rate-limited, interchangeable state contract enable reusable reasoning modules? | 32B source run above; `runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest`; `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls` | `scripts/remote/state_handoff.sh continuation-confirm-7b`; `scripts/remote/state_handoff.sh interface-final-eval-7b` | `runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest/evaluation/{comparison_summary,information_summary}.json`; `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls/evaluation/interfaces/comparison_summary.json` | Recursive decimal reuse is perfect through h32 on 9,600 cases. The final opaque adapters show an exact rate result: the 2-bit code has perfect closure and exactly 50% answer accuracy. Canonical 3-bit reaches 74.27/58.65/42.19/42.19% at h2/4/8/16; redundant 4-bit reaches 97.71/80.63/63.85/59.48%. Context-bound codes remain near chance. |
| Predicted-code equivalence | pilot | Do independently emitted tokens share causal meaning even when their surface forms differ? | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls` | `.venv/bin/python scripts/experiments/run_state_handoff_training.py compare-interfaces runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls` | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls/evaluation/interfaces/*/predicted_equivalence_summary.json` | Artifact-only predicted-donor analysis separates exact token agreement from agreement after grouping codes by downstream behavior. Redundant same-state agreement rises from 36.02% exact to 61.81% by behavior; predicted-donor preservation is 71.42%, versus 12.37% for frequency-free random codes. This is evidence for partial code equivalence, not full interchangeability. |
| Out-of-template state stress | pilot | Does recursive state use survive histories that break the matched generator's positional pattern? | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_stress` | `scripts/remote/state_handoff.sh interface-stress-7b` | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_stress/evaluation/stress/probe/comparison_summary.json` | The saved decimal adapter remains 100% accurate on all 1,600 structured, IID, shuffled, cancellation, and repeated-operation cases through h16. The 2-bit code retains perfect quotient closure and exactly 50% answers under every family. Across the four non-structured families, canonical 3-bit answers score 76.88/63.75/40.62/36.88% at h2/4/8/16; redundant 4-bit scores 96.25/76.88/56.56/38.75%. At h16 their local transition accuracies remain 78.75% and 84.26%, so valid but wrong opaque updates compound. This five-context probe selects the next run; it is not a final confidence estimate. |
| Transition closure fine-tune | active | Does spending matched supervision on reusable transitions improve long-horizon composition beyond more endpoint fitting? | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_finetune`; `runs/Qwen2.5-7B-Instruct/interventions/state_interface_endpoint_control` | `scripts/remote/state_handoff.sh interface-closure-7b` | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_finetune/evaluation/closure_comparison.json` | Running. Both runs start from the same final adapters and use byte-identical data, one epoch, 20,000 forwards, 20,000 targets, and 5,120,000 padded tokens per condition. The only change is transition-only versus encoder-only producer supervision. The gate requires at least +10 points at h8 and h16 with context-paired intervals above zero. |
| Closure stress confirmation | prepared | If closure training helps, does the gain survive non-template histories and retain the predicted-code meaning? | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_stress` | `scripts/remote/state_handoff.sh interface-closure-stress-7b` | `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_stress/evaluation/stress/probe/comparison_summary.json` | Gated follow-up comparing decimal reuse, closure-trained canonical/redundant adapters, and their endpoint-only controls on the same five stress families. Run only after the paired closure comparison finishes. |

Explicit-handoff local validation covers deterministic analysis of the completed
32B and 7B artifacts, predicted-code equivalence, the recursive continuation
bank, all four interface alphabets, the completed five-family stress probe, and
matched closure/control budgets with the pinned Qwen tokenizer. Closure
fine-tuning is running; its accuracy and gate remain unknown.

### State-interface execution order

1. `interface-stress-7b` passed its selection test: decimal recursion stayed
   perfect on all five history families through h16.
2. `interface-closure-7b` is running. Continue only if transition supervision
   beats the byte-identical endpoint control at h8 and h16 with positive
   paired intervals.
3. Run `interface-closure-stress-7b` only after step 2. Require gains on IID,
   shuffled, and cancellation histories, not only the original generator.
4. Scale to three seeds, a second operation family, and a second 7--8B model
   only if the first three steps establish closure outside the matched template.

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
| `experiments/symbolic_parser_audit_labels.jsonl` | Hand labels for the symbolic-parser fidelity audit. |
| `experiments/symbolic_parser_audit_report.json` | Summary metrics for the symbolic-parser fidelity audit. |
| `experiments/symbolic_parser_audit_matches.jsonl` | Per-window parser/gold match details for the symbolic-parser fidelity audit. |

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
