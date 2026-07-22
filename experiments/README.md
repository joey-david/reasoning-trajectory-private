# Experiment Reproduction

Each experiment is independent. Its public command lives in
`scripts/experiments/`; its implementation lives in `src/experiments/`; model
and dataset settings live in the relevant `runs/.../config.yaml`.

Completed run artifacts are inputs, not generated source. Local analysis never
loads the model and may be rerun without repeating inference.

For merge-facing experiment status, headline results, and the durable run/report
index, see [results.md](results.md). This file is limited to reproduction notes
and artifact contracts.

## Canonical Analyses

These commands reproduce the reported analyses from completed artifacts. The
result status and headline interpretation live in
[results.md](results.md#canonical-and-active-results).

| Experiment | Command | Main report |
| --- | --- | --- |
| H1: prompted and natural boundaries | `scripts/experiments/boundaries/boundary_comparison.py` | `runs/SmolLM3-3B/pilots/h1_freeform_replay/analysis/experiments/h1_boundaries/report.json` |
| H2: localized symbolic updates | `scripts/experiments/trajectory_dynamics/localized_updates.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/report.json` |
| H4: structural projection | `scripts/experiments/trajectory_dynamics/structural_contrast.py` | `runs/SmolLM3-3B/replay/h4_structural_replay/analysis/experiments/h4_structural_contrast/report.json` |
| H5: correctness prediction | `scripts/experiments/trajectory_dynamics/correctness_prediction.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h5_correctness_prediction/report.json` |
| Token-level no-free-lunch | `scripts/experiments/token_segmentation/token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation*/report.json` |
| Judged semantic boundaries | `scripts/experiments/token_segmentation/semantic_token_segmentation.py` | `runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation*/report.json` |
| H3: causal process-isomer patching | `scripts/experiments/run_h3_protocol.sh` | `runs/SmolLM3-3B/failed/h3_process_isomer_patching/analysis/report.json` |

Run a canonical local analysis with no arguments:

```bash
.venv/bin/python scripts/experiments/trajectory_dynamics/localized_updates.py
.venv/bin/python scripts/experiments/trajectory_dynamics/correctness_prediction.py
.venv/bin/python scripts/experiments/token_segmentation/token_segmentation.py
.venv/bin/python scripts/experiments/token_segmentation/semantic_token_segmentation.py
```

The token scripts run the prespecified minimum-segment sweep of 1, 4, and 8
tokens. Pass `--min-segment-tokens N` to run only one granularity. Every command
accepts explicit artifact paths for reuse on a different model or corpus; see
`--help`.

## H1: Boundaries

Inputs are the four prompting-condition runs:

```text
runs/SmolLM3-3B/pilots/h1_freeform_replay
runs/SmolLM3-3B/pilots/h1_numbered_steps_pilot
runs/SmolLM3-3B/pilots/h1_sentence_separated_pilot
runs/SmolLM3-3B/pilots/h1_paragraph_separated_pilot
```

Capture the freeform replay with `replay_capture.py`; generate the three
prompted conditions from their run configs. Then analyze:

```bash
.venv/bin/python scripts/experiments/trajectory_dynamics/replay_capture.py \
  runs/SmolLM3-3B/pilots/h1_freeform_replay
.venv/bin/python scripts/experiments/boundaries/boundary_comparison.py
```

The paragraph run currently contains 21 of 60 planned traces. Existing reports
preserve that interrupted pilot rather than silently imputing missing traces.

## H2 and H4: Updates and Operator Structure

H2 reads the primary 580-trace activation corpus:

```bash
.venv/bin/python scripts/experiments/trajectory_dynamics/localized_updates.py
```

H4 uses a separate 300-question teacher-forced replay. Rebuild its activations,
extract its symbolic updates, then train the structural projection:

```bash
.venv/bin/python scripts/experiments/trajectory_dynamics/replay_capture.py \
  runs/SmolLM3-3B/replay/h4_structural_replay
.venv/bin/python scripts/experiments/trajectory_dynamics/localized_updates.py \
  runs/SmolLM3-3B/replay/h4_structural_replay --per-sample 5
.venv/bin/python scripts/experiments/trajectory_dynamics/structural_contrast.py
```

## H5 and Token-Level Segmentation

H5 needs only the primary corpus. Token segmentation additionally needs
teacher-forced gold-answer states:

```bash
.venv/bin/python scripts/orchestrate.py \
  --job gold_answer_capture --nodes local --devices 0 \
  --run runs/SmolLM3-3B/replay/thought_units_gold_answers
.venv/bin/python scripts/experiments/trajectory_dynamics/correctness_prediction.py
.venv/bin/python scripts/experiments/token_segmentation/token_segmentation.py
```

Replace `--nodes local --devices 0` with the desired orchestration workers for
remote capture.

## Judged Semantic Boundaries

Prepare overlapping labeling windows from the primary corpus and H2 updates:

```bash
.venv/bin/python scripts/experiments/solution_object_extraction/prepare_solution_object_labels.py
```

Smoke-test one task, run the resumable two-GPU labeling queue, then analyze:

```bash
.venv/bin/python scripts/orchestrate.py \
  --job solution_object_labeling_smoke \
  --nodes upnquick --devices 0+1 \
  --run runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver
.venv/bin/python scripts/orchestrate.py \
  --job solution_object_labeling \
  --nodes upnquick --devices 0+1 \
  --run runs/Qwen3.5-122B-A10B-FP8/labeling/solution_object_silver
.venv/bin/python scripts/experiments/token_segmentation/semantic_token_segmentation.py
```

The Qwen checkpoint path is cluster-specific and is declared in the labeling
run's `config.yaml`.

## H3: Causal Patching

H3 has independently resumable pair-mining, replay, projection, patching, and
analysis stages. Rebuild the canonical pair manifest and component projections:

```bash
.venv/bin/python scripts/experiments/process_isomers/mine_process_isomers.py
.venv/bin/python scripts/experiments/trajectory_dynamics/replay_capture.py \
  runs/SmolLM3-3B/replay/h2_component_replay
.venv/bin/python scripts/experiments/process_isomers/component_localization.py
.venv/bin/python scripts/experiments/process_isomers/component_projection.py
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
.venv/bin/python scripts/experiments/process_isomers/analyze_causal_patching.py \
  runs/SmolLM3-3B/failed/h3_process_isomer_patching
```

## Artifact Contract

- `config.yaml` pins model, dataset, capture, and intervention settings.
- `generation/generations.jsonl` stores completed traces.
- `generation/hidden_states/*.npz` stores captured activations.
- `analysis/experiments/<experiment>/report.json` is the metric source of truth.
- Analysis commands overwrite only their own report directory.
- Remote jobs are resumable and should be synchronized with
  `scripts/remote.sh`; do not commit large activation artifacts.

## Explicit one-token state handoff

This follow-up asks whether a decimal state token can turn correct local updates
into reliable composition. It reuses the completed 1,920-case matched-history
run and never regenerates its inference. The local artifact-only analysis is:

```bash
.venv/bin/python scripts/experiments/depth_relief.py \
  analyze-explicit-handoff \
  runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history
.venv/bin/python scripts/experiments/depth_relief.py \
  status-explicit-handoff \
  runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history
```

At horizon 2, deterministic execution of the recorded Synthesize state scores
76.98%, versus 13.44% for one-pass Compose. The program-context-paired gain is
+63.54 points with a 95% interval of +58.33 to +68.75. At horizon 4, both paths
score 12.71% because state synthesis itself has failed. This is an oracle result,
not a passed handoff gate: no LM self, gold, or stepwise inference rows exist.

The large steps run in this order from the shared checkout on `lamgate`:

```bash
scripts/remote/state_handoff.sh phase1-32b
scripts/remote/state_handoff.sh screen-7b
scripts/remote/state_handoff.sh prepare-pilot-7b
scripts/remote/state_handoff.sh pilot-7b
```

`phase1-32b` runs history-free self, gold, and stepwise calls with
Qwen2.5-32B-Instruct revision
`5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`. `screen-7b` runs frozen
Read/Update/Synthesize/Compose and explicit handoff at h1/h2/h4/h8 with
Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. FlashAttention 2 falls back to
SDPA if unavailable. The 32B job uses one two-GPU worker; each 7B job uses one
A100. All jobs resume from append-only rows.

From the `upnquick` checkout, queue the bounded gated sequence with:

```bash
bash scripts/remote/queue_state_handoff.sh
```

It waits for both GPUs only for the 32B Phase 1 job, releases them after that
job, and then queues the one-GPU 7B screen and pilot. It stops if Phase 1 fails
or after the pilot gate; it never reserves an idle GPU.

The trainer refuses to start unless the 32B Phase 1 gate passes and the frozen
7B screen is complete. The pilot then trains `outcome_only` and
`explicit_handoff` through one shared rank-16 LoRA runner. Each condition gets
20,000 train and 2,000 validation forwards per epoch, exactly 2,281,200 active
train tokens and 5,120,000 fixed-padding compute tokens. Loss-masked control
tokens follow the only target token, so causal attention prevents them from
changing that target. Training uses h1/h2; held-out evaluation uses h2/h4/h8
over 30 unseen program contexts.

Pinned hashes are:

- 32B source dataset: `f2e02e2a4d826d7b635e8f7229fdd3abd3357fbdb6e54f0be4017d2e624d4a1d`.
- Frozen 7B screen: `2d551239e72c9bd160a05813d6895da11db86bdd57b6f2135c38947f91cc10d0`.
- Pilot train: `b27623034cabce19fe9dcea3dd047728f28a1423e0cda408a18e814de614612a`.
- Pilot validation: `63fca3627a32042ca82b8a93b92968e6632bbffaad6280416bf73e8299f7cd7b`.
- Pilot test: `8c4215684ad19ea63a6d9998bbd3fe29a4f1e6cfbf9625a1613aae0b321cb50c`.

The tiny CPU smoke covers finite state and answer losses, adapter save/reload,
resume without duplicate metrics, and evaluation without the training dataset:

```bash
.venv/bin/python scripts/experiments/run_state_handoff_training.py smoke \
  runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest
```

Primary outputs are `depth_relief/explicit_handoff/summary.json`,
`evaluation/<condition>/{cases.jsonl,summary.json}`,
`evaluation/comparison_summary.json`, `evaluation/horizon_accuracy.png`, and
`evaluation/handoff_gap.png`. Opaque codes, causal code interchange, full-scale
three-seed training, and `interchange_matrix.png` remain blocked until the pilot
gate passes.

## Causal depth relief

The controlled benchmark represents an intermediate state as a fixed-length set
of possible decimal states. Shrinking that set from `2^b` to `2^(b-r)` exposes
exactly `r` state bits without changing prompt length or checkpoint token
positions. The same case evaluates no, partial, gold, self-written,
counterfactual, and random registers. Primary depth is full-vocabulary DTR JSD
at `g=0.5`; target-position and checkpoint-position interchange provide the
causal state/read/channel checks.

The local MLX capability screen is prepared and resumable:

```bash
.venv/bin/python scripts/experiments/depth_relief.py prepare \
  runs/SmolLM3-3B/interventions/depth_relief_local_mlx
~/.venvs/mlx/bin/python scripts/experiments/depth_relief.py local-mlx \
  runs/SmolLM3-3B/interventions/depth_relief_local_mlx
```

It is below the validity gate: only one of six no/gold pairs is
unconstrained-correct, no counterfactual follows the rule-consistent branch,
and primary settling is layer 34 in every condition. This is a small-model
capability negative, not evidence for or against depth relief.

The pinned remote runs are:

```text
runs/SmolLM3-3B/interventions/depth_relief_main
runs/Qwen3-8B/interventions/depth_relief_main
runs/DeepSeek-R1-Distill-Qwen-7B/interventions/depth_relief_reasoning
runs/Qwen2.5-7B-Instruct/interventions/depth_relief_base
```

Push from the local checkout, then run the resumable queues from the shared
remote checkout reached through `lamgate`:

```bash
./scripts/remote.sh push
ssh lamgate
cd /home/lamsade/jdavid/reasoning
scripts/experiments/run_depth_relief_remote.sh
```

The runner processes Qwen3, the matched Qwen base/reasoning pair, and SmolLM3
in sequence on GPUs 0 and 1. Each queue is restart-safe through the existing
job contract, so rerunning the script resumes incomplete cases and skips
completed ones. Override the allocation with `DEPTH_RELIEF_DEVICES=0` or
`DEPTH_RELIEF_NODES=<host>` if needed. It analyzes each completed run and writes
the matched base comparison into the reasoning run folder for later pulling.

After it finishes, pull the run folders from the local checkout:

```bash
./scripts/remote.sh pull runs/SmolLM3-3B/interventions/depth_relief_main
./scripts/remote.sh pull runs/Qwen3-8B/interventions/depth_relief_main
./scripts/remote.sh pull runs/Qwen2.5-7B-Instruct/interventions/depth_relief_base
./scripts/remote.sh pull runs/DeepSeek-R1-Distill-Qwen-7B/interventions/depth_relief_reasoning
```

The first remote matrix is complete (312/312 cases). It is an informative
negative/inconclusive screen, not a positive causal-depth result. Qwen3 has a
small continuous JSD-curve-area relief of `+0.00848` (bootstrap 95% CI
`[+0.00527, +0.01137]`) and the curve area decreases monotonically as more
state bits are revealed. However, only 7/72 no/gold pairs are jointly correct,
the matched primary depth effect is zero, and only one attempted causal case
passes the gold/counterfactual behavior gate. Qwen2.5 moves in the opposite
direction (`-0.00500` curve-area relief), while DeepSeek and SmolLM saturate the
primary discrete threshold. These completed artifacts remain the v1 result and
are not reused as qualification data.

### Behavior qualification before another depth run

The follow-up separates assay validity from layer measurement. It uses pointer
programs with 2- and 3-bit states, history lengths 1/2/4/8, and five conditions:
direct one-step, no valid checkpoint, gold checkpoint, counterfactual
checkpoint, and an invalid checkpoint whose value must be ignored. Register
conditions have identical token counts and checkpoint character positions.
Only final logits are scored; no layer capture or causal intervention is run.

The local SmolLM smoke proves the diagnostic separation: the natural-language
direct control is 8/8 correct, while the small model fails the register/history
gate. All 96 cases and 480 prompts for each pinned remote model pass local
tokenizer/alignment validation. The completed runs are:

```text
runs/Qwen3-8B/interventions/depth_relief_qualification
runs/Qwen3-32B/interventions/depth_relief_qualification
runs/Qwen2.5-32B-Instruct/interventions/depth_relief_qualification
```

Run all three resumable qualifications sequentially from the shared checkout:

```bash
./scripts/remote.sh push
ssh lamgate
cd /home/lamsade/jdavid/reasoning
scripts/experiments/run_depth_relief_qualification_remote.sh
```

By default, the 8B run uses independent workers on GPUs `0,1`; each 32B run
uses one sharded `0+1` worker. Override these with
`DEPTH_RELIEF_SMALL_DEVICES` and `DEPTH_RELIEF_LARGE_DEVICES`.

The download wrapper caches models under `$HOME/.cache/huggingface` by default.
The full three-model sequence needs roughly 170 GB of free cache space. When `/tmp`
is smaller, point the existing cache boundary at a larger filesystem; the
orchestrator forwards it to every remote worker:

```bash
HF_LOCAL_CACHE=/path/with/at-least-170GB \
  scripts/experiments/run_depth_relief_qualification_remote.sh
```

The gate uses bootstrap lower bounds: direct accuracy >=0.90, gold and
counterfactual accuracy >=0.85, no/invalid history accuracy >=0.60, invalid
register invariance >=0.60, candidate probability mass >=0.80, and at least 50
jointly correct no/gold cases. All three runs fail this gate:

| Model | Direct | None | Gold | Counterfactual | Gold minus none | Joint correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B | 1.000 | 0.188 | 0.406 | 0.479 | +0.219 `[0.135, 0.313]` | 17/96 |
| Qwen3-32B | 1.000 | 0.219 | 0.667 | 0.760 | +0.448 `[0.344, 0.552]` | 18/96 |
| Qwen2.5-32B-Instruct | 0.958 | 0.198 | 0.948 | 0.906 | +0.750 `[0.667, 0.833]` | 19/96 |

The paired checkpoint effect is large and monotonic with capability, and
Qwen2.5 never loses a previously correct no-checkpoint case: 72 cases are
rescued, 19 remain correct, and five remain wrong. This is strong behavioral
evidence that an explicit state relieves computation. It is not yet admissible
for layer-depth measurement because the no-checkpoint lookup composition is
near chance and leaves fewer than 20 jointly correct pairs per model. Scaling
Qwen3 from 8B to 32B improves valid-register use but barely changes that
baseline; Qwen2.5 changes register use dramatically without fixing it.

The report's invalid-register invariance is unconditional prediction agreement
between the two flag-zero prompts; the earlier implementation accidentally
conditioned it on both answers being correct. Corrected invariance is
0.635/0.562/0.625 for Qwen3-8B/Qwen3-32B/Qwen2.5-32B, with every lower bound
below the 0.60 gate. The next assay must remove the distracting numeric value
from the absent-checkpoint condition and calibrate an easier history family to
a jointly-correct frontier before any layer capture. More scale, more samples,
or Qwen3.6-27B would not repair this contract.

### Sentinel frontier calibration

The replacement discovery grid removes both the numeric distractor and the
validity flag. Its matched prompts contain exactly one checkpoint payload token:
`state=[unknown]`, `state=[missing]`, or `state=[0..7]`. Every other prompt token
is identical across absent, alternate-absent, gold, and counterfactual
conditions. The history grid crosses add, XOR, and affine transitions with
lengths 1/2/3/4 and 2-/3-bit states; FINAL remains a pointer lookup.

The local 24-case SmolLM smoke completed through the shared scorer and report.
All 288 Qwen2.5-32B cases (1,440 prompts) pass pinned-tokenizer validation:
one checkpoint-token substitution, one-token answer candidates, and matched
prompt lengths. The discovery report can select a cell for held-out
confirmation but always writes `depth_capture_authorized: false`.

Run the prepared Qwen2.5 grid from the shared remote checkout:

```bash
./scripts/remote.sh push
ssh lamgate
ssh readycash
cd ~/reasoning
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_depth_relief_frontier_remote.sh
```

The resumable job uses one `0+1` Qwen2.5-32B worker. A cell is eligible only if
direct and valid-register accuracy are at least 0.90, both absent sentinels are
at least 0.60 accurate, absent-sentinel prediction agreement is at least 0.80,
candidate mass is at least 0.80, and at least 8/12 none/gold pairs are jointly
correct. Selection targets none accuracy near 0.75; an eligible discovery cell
must still pass a fresh held-out confirmation before layer work.

The completed grid has zero eligible cells. The nearest cell is 2-bit add with
two history steps: direct/gold are 1.000/0.917, but the two absent sentinels are
0.583/0.500, counterfactual is 0.750, and only 7/12 none/gold pairs are jointly
correct. The best fully register-valid cells still reach only 0.417 absent
accuracy. Thus replacing the numeric distractor fixes sentinel stability but
does not produce a valid computation frontier.

Case-level execution identifies the failure: among 69 one-step examples where
skipping history changes the answer, the absent condition applies FINAL to the
start state in 44 cases (63.8%), computes the history correctly in 11 (15.9%),
and emits another answer in 14. Across all lengths, 123/238 distinguishable
errors skip the entire history. This supersedes the planned zero-history closure
run: the relevant ambiguity is whether the model fails to synthesize the current
state or fails to route an available state into the final update.

After completion, pull the compact run folder:

```bash
./scripts/remote.sh pull \
  runs/Qwen2.5-32B-Instruct/interventions/depth_relief_frontier_calibration
```

After completion, pull only the qualification folders:

```bash
./scripts/remote.sh pull runs/Qwen3-8B/interventions/depth_relief_qualification
./scripts/remote.sh pull runs/Qwen3-32B/interventions/depth_relief_qualification
./scripts/remote.sh pull runs/Qwen2.5-32B-Instruct/interventions/depth_relief_qualification
```

### State-materialization factorization

The active assay evaluates the same semantic case through four prompts: Read
returns an explicit current state, Update applies FINAL to that state, Synthesize
constructs it from the history, and Compose constructs it and applies FINAL.
Each history transition is separately screened on its actual input state. Cases
are admitted to the primary taxonomy only when Read, Update, and every
constituent transition are correct.

The prepared datasets contain 216 cases and 1,368 scored prompts per model:
3-bit states, add/XOR/affine histories, horizons 1/2/4, and prose plus assignment
formats. Correct composition, history-only, final-on-start, and identity outputs
are distinct by construction. Full-vocabulary output log-probabilities are kept
for every state, so these branches are prespecified diagnostics.

Run the three resumable model evaluations in sequence from the shared checkout:

```bash
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_state_materialization_remote.sh
```

The completed factorization changes the hypothesis. Read is 1.000 for all three
models, while Update/Synthesize/Compose accuracy is 0.852/0.060/0.032 for
Qwen3-8B, 0.954/0.176/0.069 for Qwen3-32B, and 0.991/0.440/0.088 for
Qwen2.5-32B. Scaling improves synthesis, but composition remains near floor.

Qwen2.5 supplies the clean routing subset: 214/216 cases pass Read+Update, 94
also Synthesize correctly, and only 8/94 Compose correctly. Among the 86 paired
failures, 43 produce the exact final-on-start shortcut; its output log-probability
exceeds the correct composition by 7.44 nats on average. This discovery supports
a serial-integration failure, not a pure state-synthesis bottleneck.

The held-out confirmation uses those 94 cases without rerunning the discovery
conditions. It restores the full history, inserts either the gold or a
counterfactual current state immediately before FINAL, and changes exactly one
state token between conditions. The gate requires bootstrap-lower-bound accuracy
of 0.85 in both conditions, candidate mass of 0.80, and at least 50 jointly
correct factual/counterfactual pairs:

```bash
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_state_routing_remote.sh
```

The confirmation passed. Materialized and counterfactual accuracy are 0.915 and
0.936, with 80/94 jointly correct. On the same 94 cases, explicit state rescues
78/86 original Compose failures; correct-answer log-probability increases on
all 94 cases (mean +8.93 nats). This establishes serial integration failure
after successful synthesis; internal routing, retention, and use remain to be
distinguished causally.

The first causal capture deliberately stopped at its competence gate. Its new
`?`-register prompt preserved the failure regime (6/94 correct; 55
final-on-start errors), but changed the valid-state computation: factual and
counterfactual accuracy fell to 64/94 and 63/94. This is a failed prompt
contract, not a causal result; lowering the 0.85 gate would mix different
computations.

The replacement removes the synthetic register. It reuses the exact original
Compose prompt as the recipient and the exact confirmed factual/counterfactual
routing prompts as donors, capturing each at their common final `Answer=` token.
Donors share the recipient's state and surface format but must have a different
expected answer, so a successful patch cannot copy the donor answer. An
eight-dimensional materialized-state subspace is fit on 52 training cases; one
of eight layers is selected on 21 validation cases and reported once on 21
held-out cases. Factual and counterfactual transfers each have a matched random
subspace control, with a full-vector factual transfer as an upper bound. The
capture also retains the start, each history-step endpoint, the final-rule
endpoint, and `Answer=` without changing the prompt. A cross-condition decoder
selects one layer on validation and tests whether the explicit-state coordinate
is present at the held-out history and answer positions. The runner stops before
patching if the original Compose failure or confirmed explicit-state competence
does not reproduce:

```bash
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_state_transfer_remote.sh
```

If both the answer-disjoint transfer gate and history-end localization pass,
the same runner launches a final 21-case self-handoff job. It moves only the
recipient's own history-end component in the learned state subspace to its
answer anchor, with random-rank and full-vector controls. Failure of either
prerequisite gate stops the self-handoff rather than relaxing admission.

Both prerequisite gates failed on held-out cases. The validation-selected
localization layer decoded the current state at the history endpoint with 0.190
accuracy, versus 0.143 for shuffled labels and 0.190 for the initial state; the
three prespecified localization checks all failed. The validation-selected
causal layer produced a gold-target probability shift of +0.009 (95% CI
[-0.042,+0.075]) and a counterfactual-target shift of -0.066
([-0.186,+0.034]); neither beat its random-subspace control. Even the full
activation transfer shifted the gold target by only +0.031
([-0.016,+0.106]). Self-handoff was therefore inadmissible and was not run.
This rejects a compact, cross-prompt materialized-state coordinate at the
tested positions; it does not establish that no nonlinear, distributed, or
prompt-specific history representation exists.

### Matched-history state abstraction

The active confirmatory assay removes the cross-prompt coordinate assumption.
Within each randomized program context, it constructs a complete 8-state by
8-route grid. Same-state donors reach the same current state through different
histories; different-state donors share the entire history prefix and differ
only in the final history operation. The final pointer rule is shared within a
context and bijective, so the causal target always changes with donor state.

The pinned Qwen2.5-32B run has 1,920 cases across horizons 2 and 4. Its 30
independent contexts give five train, five validation, and five test contexts
per horizon. Every split is state/route balanced. All 13,440 exact-logit
conditions validate with the pinned tokenizer, and the causal history endpoint
is the same period token in every case. No answer tokens are generated.

The first job scores the existing Read/Update/Synthesize/Compose contract and
captures residuals at every history-step endpoint, the Synthesize answer
anchor, and the explicit Update state. Analysis estimates held-out variational
lower bounds on state information and route information conditional on state,
including a joint-history decoder and shuffled-label controls. Layers are
selected on validation contexts; confidence intervals resample program
contexts rather than correlated histories.

The causal continuation runs only if enough Compose failures separately pass
Read, Update, Synthesize, and every constituent transition. It patches between
matched implicit-history endpoints, not from an explicit prompt. Different-
state and same-state donors, full-vector controls, and a norm-matched random
subspace distinguish a causal state quotient from path-bound history. The state
subspace is fit on the complete balanced training grid, never a success-selected
subset. The runner caps selection at one recipient per state and eight per
context, selects
one of five fixed layers on validation, and reports test contexts once:

```bash
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_state_abstraction_remote.sh
```

The zero-inference decoder-transfer audit of the prior 94 captures is already
written under the old run. The runner executes only the matched assay and its
gated causal continuation. Primary reports are
`depth_relief/state_abstraction/information_summary.json` and
`depth_relief/state_abstraction/interchange_summary.json` in
`runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history`.

Two independent generalization assays use the same orchestrated factorization
job: the exact decimal benchmark on Mistral-Small-24B-Instruct-2501 and opaque
permutation states over eight tokenizer-native Greek symbols on Qwen2.5-32B.
Each has 216 cases and 1,368 validated prompts. A matched routing confirmation
runs only when its factorization supports the prespecified serial-integration
decision:

```bash
HF_LOCAL_CACHE="$HOME/.cache/huggingface" \
  scripts/experiments/run_state_materialization_replications_remote.sh
```

## Latent solution-object extraction

The plan-driven A-H protocol uses a controlled isomorphic bank for extraction,
decoding, and causal tests, then evaluates trajectory failure and reranking on
the completed real mixed-success corpus.

The local pilot is already materialized. Re-run its stages independently:

```bash
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py prepare \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py capture \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py analyze \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py causal \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
```

The medium bank is pinned at 24 base graphs, 2,496 edit-state records, five
layers, 12 causal pairs per condition, and all 10
available real rollouts for up to 58 mixed questions. Push code/config/datasets
from the local checkout, then run on the GPU checkout reached through
`ssh lamgate`:

```bash
./scripts/remote.sh push

# after entering the GPU checkout through the configured lamgate route
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py validate \
  runs/SmolLM3-3B/interventions/solution_object_extraction_medium
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py run \
  runs/SmolLM3-3B/interventions/solution_object_extraction_medium
```

Every model/analysis loop has `tqdm` progress. `run` executes in the foreground;
it does not detach or start a remote job. The medium remote run treats the
pre-existing mixed-trajectory corpus as optional because the normal push
intentionally excludes its 631 hidden-state files. When absent, G/H are reported
as skipped while A-F complete normally; run G/H after pulling the medium capture
to the local checkout that owns those source states. To resume after a completed
capture without loading the model again, invoke `analyze` and then `causal`.

### Increased object extraction and causal writing

The follow-up sweep reuses the bank and adds five feature views, supervised-rank
dimension/layer/scope selection, residual MLP encoders, a question-disjoint
low-rank causal writer, and matched object/lexical/answer/PCA/random ablations.
Run every local stage and validate its artifact contract with:

```bash
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py improve \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
.venv/bin/python scripts/experiments/solution_object_extraction/solution_object_extraction.py \
  validate-improvement \
  runs/SmolLM3-3B/interventions/solution_object_extraction_small
```

The medium continuation runner executes in the foreground, uses only GPU index
0 on the selected node, writes one streamed log, reuses completed upstream
artifacts, and has a hard 12-hour wall-clock limit:

```bash
./scripts/remote.sh push
ssh upnquick
cd /home/lamsade/jdavid/reasoning
scripts/experiments/run_solution_object_improvement_remote.sh
```

The final validation requires all requested patch scopes, encoder cells, writer
epoch/alpha cells, ablation controls, ablation-grid cells when present, and the
real-trajectory gate. It validates completion, not a positive scientific
outcome.

Current medium result after the writer/grid continuation:

- Validation passes with 80 causal cells, 72 nonlinear cells, 12 writer
  epoch/alpha cells, 64 standard ablation prompts, and 8 targeted ablation-grid
  cells.
- Retrieval bottleneck improved substantially under the selected nonlinear
  encoder: validation/template-validation/heldout-vocab/heldout-template top-1
  are 0.917/0.738/0.873/0.692, with lexical probe 0.790.
- The fixed learned writer has finite decreasing losses, but still does not
  beat the linear subspace: selected donor-probability delta is 0.008 versus
  0.212 for the linear control, so `recommended_method` remains
  `linear_subspace`.
- The targeted ablation grid found a low-leakage pass case:
  layer 32, patch layers `[29, 32, 35]`, rank 16, multi-layer causal cell,
  final-token ablation, lexical probe 0.610, causal strength 0.345. Object
  ablation drops correct probability by 0.0425 versus 0.0348 for compression,
  0.0049 for answer, near-zero lexical, and near-zero random, giving an
  object-minus-strongest-control margin of +0.0077.
- The gate is now `ready_for_real_trajectory` by the prespecified aggregate
  criterion. Interpret the pass as promising but not decisive: on the selected
  row, paired prompt-level bootstrap keeps object > answer/lexical/random, but
  object-vs-compression is small and crosses zero. The next meaningful stage is
  G/H: fit and evaluate real mixed-success object-trajectory
  predictors/rerankers; do not rerun the broad retrieval/causal/nonlinear sweeps
unless the run contract changes.

## Replications for the layer-computation note

The three sources cited in `layers/tex/notes.pdf` have paper-matched run
folders and one restart-safe driver:

| Paper | Replication | Run folder |
| --- | --- | --- |
| Lad et al. | all-layer drop and adjacent swap on 1M Pile tokens | `runs/Qwen2.5-1.5B/replications/lad_layer_robustness` |
| Yang et al. | ABA/ABB head-level causal mediation for three mechanisms | `runs/Qwen2.5-7B/replications/yang_symbolic_mechanisms` |
| Zhang et al. | full GRPO plus published anchor layers 1/7/10/12/24 by default | `runs/Qwen3-1.7B-Base/replications/zhang_single_layer_rl` |

Lad samples fixed 256-token sequences from deterministic random pages of the
pinned Pile revision. Yang reproduces the released code's letter-only Qwen
vocabulary construction before enforcing its exact 10-shot token layout. Both
use models and protocols present in their papers. Zhang uses the
paper's Qwen3-1.7B-Base, 50K NuminaMath-CoT, published GRPO hyperparameters,
four in-domain math benchmarks, and Average@32 for AMC. The official Lad and
Yang repositories were used as protocol references; their source trees are not
vendored. The Zhang paper does not publish a code repository, so its documented
protocol is implemented with pinned TRL 1.8.0. Because it does not publish the
indices for its filtered 50K training subset, this replication uses the first
50K rows of the pinned NuminaMath-CoT revision and records that choice in the
run config. GSM8K's worked solutions are reduced to their published `####`
final-answer field before the shared math verifier scores them.

Push the branch, enter the verified shared checkout on `upnquick`, and run:

```bash
./scripts/remote.sh push
ssh lamgate
ssh upnquick
cd /home/lamsade/jdavid/reasoning
scripts/experiments/run_layer_paper_replications_remote.sh
```

The driver validates and reuses the pushed pinned datasets (preparing them only
if absent or invalid), runs the full Lad and Yang replications concurrently,
then uses both GPUs for the Zhang base/full and independent layer settings.
The default 48-hour campaign scans the five Qwen3-1.7B anchor layers reported
in the paper: early controls 1/7, high-contribution middle layers 10/12, and
late control 24. Set `LAYER_RL_SCAN=full` to resume through all 28 layers after
the core result is established. Every stage is restart-safe.

The driver writes
`experiments/layer_replication_checklist.{md,json}` from completed artifacts.
It refreshes the checklist on startup and on every exit, including interrupted
runs. The checklist distinguishes a finished matrix from successful
reproduction of the paper's central empirical result. Default
`scripts/remote.sh pull` excludes the GRPO checkpoints while retaining those
reports. No job is detached; rerun the same command after interruption.
