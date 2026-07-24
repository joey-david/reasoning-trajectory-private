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

### Three-hour length-generalization screen

The compact follow-up uses five independent one-GPU workers: `ourasi:0,1`,
`seacove:3`, and `coktailjet:0,1`. It does not reserve GPUs or wait for them.
Start it only after those devices are free:

```bash
STATE_HANDOFF_3H_DRY_RUN=true \
bash scripts/remote/state_handoff_three_hour.sh

bash scripts/remote/state_handoff_three_hour.sh
```

Override the shared hosts without editing the script:

```bash
STATE_HANDOFF_3H_NODES="ourasi seacove coktailjet" \
STATE_HANDOFF_3H_DEVICES="0,1 3 0,1" \
bash scripts/remote/state_handoff_three_hour.sh
```

The runner prepares and validates nine run folders, dispatches 12
training-plus-evaluation tasks, then runs four length challenges and one
cross-adapter substitution. Three hours is the target runtime, not a deadline:
a healthy slow task may finish. A failed stage is recorded under
`runs/_three_hour/state_handoff/` and does not stop the later stages.

Pull these run folders after completion:

```text
state_interface_rate_sweep_3h
state_interface_rate_sweep_outcome_3h
state_interface_rate_sweep_donor_3h
state_interface_algebra_primitives_3h
state_interface_algebra_primitives_outcome_3h
state_interface_proof_actions_3h
state_interface_proof_actions_outcome_3h
state_interface_register_machine_3h
state_interface_register_machine_outcome_3h
```

### Paper confirmation: dense register execution

This run corrects the register entry contract and uses all five available
workers without reserving them:

```bash
STATE_HANDOFF_CONFIRM_DRY_RUN=true \
bash scripts/remote/state_handoff_paper_confirmation.sh

bash scripts/remote/state_handoff_paper_confirmation.sh
```

Defaults are `ourasi:0,1`, `kaisertrot:1`, and `coktailjet:0,1`. Override them
with `STATE_HANDOFF_CONFIRM_NODES` and `STATE_HANDOFF_CONFIRM_DEVICES`. The
runner prepares eight independent run folders, schedules three Qwen seeds and
one Mistral interface/control pair, retries unfinished tasks once after a
worker failure, reduces every completed run, and finishes with the small h64
active-proof-depth challenge. Logs and the append-only status ledger live
under `runs/_confirmation/state_handoff/`.

All live under
`runs/Qwen2.5-7B-Instruct/interventions/`. The screen has two independent test
contexts per main comparison; use it to choose claims and full runs, not as the
final three-seed estimate.

The 32B Phase 1 gate passed. At h2, two-call self handoff scores 76.98% versus
13.44% one-pass Compose, a context-paired +63.54 points with a 95% interval of
+57.92 to +68.75. Gold and stepwise handoff are 100% at h2 and h4. At h4,
self handoff remains at 12.71% because the first history-to-state call has
already failed.

The large steps run in this order from the shared checkout on `lamgate`:

```bash
scripts/remote/state_handoff.sh phase1-32b
scripts/remote/state_handoff.sh screen-7b
scripts/remote/state_handoff.sh prepare-pilot-7b
scripts/remote/state_handoff.sh pilot-7b
scripts/remote/state_handoff.sh continuation-probe-7b
scripts/remote/state_handoff.sh continuation-confirm-7b
scripts/remote/state_handoff.sh interface-pilot-7b
scripts/remote/state_handoff.sh interface-final-eval-7b
scripts/remote/state_handoff.sh interface-stress-7b
scripts/remote/state_handoff.sh interface-closure-7b
scripts/remote/state_handoff.sh interface-closure-stress-7b
```

`phase1-32b` runs history-free self, gold, and stepwise calls with
Qwen2.5-32B-Instruct revision
`5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`. `screen-7b` runs frozen
Read/Update/Synthesize/Compose and explicit handoff at h1/h2/h4/h8 with
Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. FlashAttention 2 falls back to
SDPA if unavailable. The 32B job uses one two-GPU worker; each 7B job uses one
A100. All jobs resume from append-only rows.

The first 7B pilot is complete and failed its OOD gate. `explicit_handoff`
reaches 100% on unseen-context h2, versus 12.08% outcome-only, but returns to
12.08% at h4 and 11.77% at h8. Gold-code continuation is 100% throughout.
The adapter therefore learned a perfect fixed-length state map and consumer,
not a closed transition interface. Exact artifact analysis finds 3.00 retained
state bits at h2, 0.28 at h4, and 0.04 at h8.

`continuation-probe-7b` reuses that saved adapter without training. It applies
the learned h2 mapping recursively at h2/h4/h8/h16 and writes local closure,
end-to-end accuracy, same-state agreement, and information measures under
`evaluation/continuation/probe/`.

`interface-stress-7b` does not train. It tests the saved decimal and opaque
adapters on structured, IID, shuffled, cancellation, and repeated-operation
histories. `interface-closure-7b` then continues the saved canonical and
redundant adapters for one matched epoch. Its transition condition and
endpoint-only control have identical program files and compute budgets.
`interface-closure-stress-7b` is the gated out-of-template comparison of those
new adapters.

`interface-pilot-7b` owns the nontrivial follow-up in
`state_interface_rate_controls`. Two A100 workers train four rank-16 adapters:

- `canonical_opaque`: one global opaque eight-code contract, exactly three bits;
- `context_bound`: eight codes with a different permutation per context;
- `compressed_2bit`: four codes, with a 50% exact-state ceiling;
- `redundant_4bit`: sixteen codes carrying state plus one path bit.

Every condition gets 10,000 semantic pairs, 20,000 forwards, 20,000 target
tokens, and 5,120,000 fixed-padding tokens per epoch. State producers use 25
training contexts and code consumers use the other 25; the 30 test contexts are
new to both. Training sees only h1/h2 blocks. Evaluation recursively composes
h2 blocks at h2/h4/h8/h16, measures global and context-conditional mutual
information, and builds same-state/different-state interchange matrices from
history-free consumer calls.

Pinned hashes are:

- 32B source dataset: `f2e02e2a4d826d7b635e8f7229fdd3abd3357fbdb6e54f0be4017d2e624d4a1d`.
- Frozen 7B screen: `2d551239e72c9bd160a05813d6895da11db86bdd57b6f2135c38947f91cc10d0`.
- Pilot train: `b27623034cabce19fe9dcea3dd047728f28a1423e0cda408a18e814de614612a`.
- Pilot validation: `63fca3627a32042ca82b8a93b92968e6632bbffaad6280416bf73e8299f7cd7b`.
- Pilot test: `8c4215684ad19ea63a6d9998bbd3fe29a4f1e6cfbf9625a1613aae0b321cb50c`.
- Interface train: `4c7c62c04ffcec8ceaa62b5347086f70b0bb372fe847ce65177a26bafccb98ec`.
- Interface validation: `2cc06eaf70955ec76941de40e439631045e5865f8d33f7a00071c153cdb8dd2d`.
- Interface test: `de35ab315505f6b6e524b1d4b4950e43309b21a23271cd1f304c71038589e781`.

The tiny CPU smoke covers finite state and answer losses, adapter save/reload,
resume without duplicate metrics, and evaluation without the training dataset:

```bash
.venv/bin/python scripts/experiments/run_state_handoff_training.py smoke \
  runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest
.venv/bin/python scripts/experiments/run_state_handoff_training.py smoke-interfaces \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls
```

New outputs are `evaluation/information_summary.json`,
`evaluation/{rate_capacity,state_information}.png`,
`evaluation/continuation/<profile>/{cases,summary}.json*`, and
`evaluation/interfaces/<condition>/{cases,summary,interchange_summary}.json*`.
The interface comparison owns `comparison_summary.json`,
`interface_accuracy.png`, and one `interchange_matrix.png` per condition. Full
three-seed confirmation remains gated on the one-seed interface result.

### State-interface generalization runs

The next suite separates five claims. `interface-joint-closure-7b` checks
whether 25% decimal-entry replay prevents the encoder damage seen after
transition-only closure. `interface-algebra-transfer-7b` holds out three
ordered pairs of add, XOR, and affine operations. The four-bit algebra run
moves the state entropy from three to four bits and compares 3-, 4-, and
5-bit channels. The proof run uses a four-fact ledger and holds out causal
two-premise rules. Seed and model-family confirmation remain separate jobs.

On a host that owns two visible A100s, launch one stage at a time with:

```bash
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-joint-closure-7b
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-algebra-transfer-7b
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-proof-transfer-7b
```

The joint-closure run needs the saved closure adapters on the same host. The
algebra and proof runs start new LoRAs from their pinned base model and can run
on another two-GPU host. Each transfer action prepares its deterministic data,
validates token and compute contracts, trains the interface and outcome arms
through one shared dynamic queue, evaluates every saved case, and writes
`evaluation/generalization_summary.json`.

For an unattended two-GPU sweep, use the resilient wrapper. It tries the
breadth runs first, then confirmation seeds and the second model. A failed or
timed-out action does not block the next action:

```bash
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
STATE_HANDOFF_ACTION_TIMEOUT=7h \
  bash scripts/remote/state_handoff_overnight.sh
```

Each action keeps its normal run folder. The wrapper also writes one log per
action plus `status.jsonl`, `session.txt`, and `run_paths.txt` under
`runs/_overnight/state_handoff/<session>/`. `Ctrl-C` or `TERM` stops the
current process group and leaves all completed checkpoints and cases in place.
Running the command again resumes incomplete training and evaluation and skips
completed tasks. Pass action names as arguments to run a smaller ordered
subset. Check the plan without starting work with:

```bash
STATE_HANDOFF_OVERNIGHT_DRY_RUN=true \
  bash scripts/remote/state_handoff_overnight.sh
```

After each stage, pull only its light artifacts from the Mac. The default pull
excludes `.pt` and `.safetensors`; use `--pt` only when adapter weights must
move:

```bash
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_joint_closure
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome
```

Inspect `.gate.status` in each generalization summary. Joint closure does not
gate the independent transfer tests. If a first transfer seed passes, run its
two extra Qwen seeds before adding breadth. Run the four-bit rate shift after
all three Qwen algebra seeds pass, and Mistral after all three Qwen proof seeds
pass:

```bash
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-algebra-confirm-7b
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-proof-confirm-7b
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-width4-transfer-7b
STATE_HANDOFF_NODES=local STATE_HANDOFF_7B_DEVICES=0,1 \
  bash scripts/remote/state_handoff.sh interface-proof-second-model-7b
```

Pull the matching paired folders after each action. The seed-2 Qwen interface
folder also owns `evaluation/replication_summary.json`. The Mistral result owns
its normal `evaluation/generalization_summary.json`:

```bash
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed2 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed2 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed3 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed3
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed2 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed2 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed3 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed3
scripts/remote.sh pull \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_algebra \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_outcome
scripts/remote.sh pull \
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_proof \
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_outcome
```

All training arms use 20,000 forwards, 20,000 supervised one-token targets,
and 5,120,000 attended padded tokens. Interface producer and consumer contexts
are disjoint; test contexts are unseen. The proof bank has 9,600 cases and
1,500 causal conjunctions. Its state labels, code symbols, and answer symbols
all pass the real Qwen and Mistral chat boundaries. Pinned semantic hashes are:

- Joint closure: train `5b0fce6736d09860f5751ede621d7fe054162e4931a59fb4a42d532f5d467b44`,
  validation `16a07f8162e5c80099a0a0280e6f4c9399427c154040ee5e899a3355f4450ef8`,
  test `1fa23d390197554857d51dd959c052fefc82f44ca29fe924e19241b4ccaead0a`.
- Algebra transfer: train `6937c2517a893a55629f0c2e392db8ee81c7144c37da430ada340cbcb299d278`,
  validation `f15d3250385197f0a0c188b96f6dafab18cc1637a6da22ad02f1e2a4336bef6c`,
  test `3ab1a416fc4cf1993a17b05d593aeb72c7fa1ac7bbada4e7ab53ea06539c276e`.
- Four-bit algebra: train `f0eb19b7a45cc7ce716bbce1f54e74eca566867678b1939fbd4fd37400e08c74`,
  validation `9368484b2c87907de599b324130b2b564a7ab5e4a998861d4e34a2bd64d86475`,
  test `0d28549199f7d8949862c70734289d490db6d2e83623c810326610564828cd9b`.
- Proof transfer: train `8d248619b36323141ce11b153632899cde0612170b55622ac5b91f3f1f2477f8`,
  validation `bc0896156c5da1188bb51143ac07fbf21d3427f630ea78f985a368b268c759af`,
  test `69ca8e8f522d13ff276b333f52c6a6813bf743c7658a317786d3ca562bd4f95e`.
  Its shared data-manifest hash is
  `d056b75924f494fbd66f79e489c33e0f1824694f614415e5047cd19a1b03d060`.

Local validation covers data generation, exact symbolic paths, one-token
boundaries, prompt lengths, matched compute, deterministic comparison, LoRA
save/reload, and resume. It does not validate A100 training speed, FlashAttention,
or any unrun model result.

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
