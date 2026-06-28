# Experiment Plan

This checklist operationalizes the hypotheses in `plan.md`. The immediate goal
is to identify model/dataset pairs that produce both correct and incorrect
traces for the same questions. Activation capture begins only after that gate.

## Success Criteria

- A **scored rollout** has an automatically extracted answer and a known gold
  answer.
- A **mixed instance** has at least one correct and one incorrect rollout.
- A **frontier instance** has a pass rate from 20% through 80%.
- A dataset is the preferred capture dataset when:
  - at least 95% of rollouts are scored;
  - the pilot has at least five frontier instances;
  - aggregate accuracy is between 15% and 85%;
  - failures are substantive rather than answer-format failures.
- Before latent capture, expand the best pilot until it contains at least 20
  frontier instances.
- `scripts/summarize_screening.py` is the authoritative classifier. Results go
  in `experiments/dataset_saturation.csv`; per-question results go in each
  run's `analysis/mixed_samples.csv`.

## Phase 0: Find The Capability Frontier

- [x] Define automatic screening metrics and CSV schema.
  - [x] Track aggregate accuracy.
  - [x] Track per-instance rollout counts.
  - [x] Track mixed-instance and frontier-instance rates.
  - [x] Classify runs as `partial`, `length_capped`, `saturated`, `frontier`,
        `middling`, `too_hard`, or `unscored`.
- [x] Record the Qwen3-14B AIME failure mode.
  - [x] Five of five completed AIME 2024 generations hit the 8,192-token cap.
  - [x] Keep the old Qwen3-14B runs for reference.
- [x] Prepare Qwen3-8B thinking-mode screens without latent capture.
  - [x] Record PolyMath high as unsuitable after 22/22 generations hit its
        4,096-token cap.
  - [x] PolyMath medium, scalar numeric subset: 20 instances x 10 rollouts.
  - [x] MBPP+ code generation: 20 instances x 10 rollouts.
  - [x] BigCodeBench-Hard code generation: 20 instances x 10 rollouts.
  - [x] Remove program-output-prediction screens from the active plan.
  - [x] Pin each selected dataset with `scripts/prepare_dataset.py` before
        inference.
- [x] Prepare a smaller-model capability ladder without latent capture.
  - [x] SmolLM3-3B on GSM-Symbolic-P1 and MBPP+.
  - [x] Qwen3-4B on PolyMath medium and MBPP+.
  - [x] DeepSeek-R1-Distill-Qwen-7B on PolyMath medium and
        BigCodeBench-Hard.
  - [x] Use 20 instances x 10 rollouts for every screen.
  - [x] Tell every model to think only as long as needed and reason clearly
        and concisely.
  - [x] Reuse the same MBPP+ instances at 3B/4B and the same PolyMath instances
        at 4B/7B.
  - [x] Pin every model and dataset revision.
- [ ] Run the six smaller-model screens.
  - [x] Generate SmolLM3-3B GSM-Symbolic-P1 rollouts.
  - [x] Generate SmolLM3-3B MBPP+ solutions.
  - [x] Generate Qwen3-4B PolyMath-medium rollouts.
  - [x] Generate Qwen3-4B MBPP+ solutions.
  - [x] Generate DeepSeek-7B PolyMath-medium rollouts.
  - [x] Generate DeepSeek-7B BigCodeBench-Hard solutions.
  - [x] Analyze DeepSeek-7B PolyMath-medium after pulling.
  - [x] Update `experiments/dataset_saturation.csv` with pulled runs.
  - [x] Analyze SmolLM3-3B GSM-Symbolic-P1 after pulling.
  - [x] Analyze Qwen3-4B PolyMath-medium after pulling.
  - [x] Update `experiments/dataset_saturation.csv` with the pulled 3B/4B runs.
  - [ ] Grade generated code with the official benchmark tests in isolation.
  - [ ] Re-run screening summaries after code grading and remaining runs.
- [ ] Expand the clean numeric frontier candidate.
  - [x] SmolLM3-3B GSM-Symbolic-P1 is the first clean frontier run: 200/200
        scored, 2/200 capped, 77.0% accuracy, and 4/20 frontier instances.
  - [x] Qwen3-4B PolyMath-medium is cap-distorted: 181/200 capped, while
        uncapped rollouts were 19/19 correct.
  - [x] Prepare `runs/SmolLM3-3B/gsm_symbolic_p1_frontier_expand` with 80 new
        GSM-Symbolic-P1 instances and 10 rollouts each.
  - [x] Verify the expansion has no question overlap with the first 20-item
        pilot.
  - [x] Generate the 80-item expansion screen
  - [x] Analyze and summarize the expansion screen.
  - [x] The expansion added 30 frontier instances at 69.13% accuracy, with
        22/800 rollouts capped at the pulled 4,096-token limit.
  - [x] Identify at least 20 clean frontier instances across the original and
        expanded SmolLM3-3B GSM-Symbolic-P1 screens before activation capture.
- [ ] Resolve the DeepSeek-7B length-cap finding before capture.
  - [x] DeepSeek-7B PolyMath-medium completed all 200 rollouts, but 172/200
        hit the 2,048-token cap.
  - [x] The 28 uncapped PolyMath rollouts were scored and mostly correct
        27/28, so the aggregate 20.5% accuracy is cap-distorted.
  - [x] DeepSeek-7B BigCodeBench-Hard completed all 200 rollouts, but 134/200
        hit the 2,048-token cap and still requires test-based grading.
  - [x] Prepare DeepSeek-7B AIME 2024 long-cap diagnostic: 6 instances x 3
        rollouts, 16,384-token cap.
  - [x] Prepare DeepSeek-7B AIME 2025 long-cap diagnostic: 6 instances x 3
        rollouts, 16,384-token cap.
  - [x] Analyze DeepSeek-7B AIME 2024 diagnostic: 18/18 scored, 5/18 capped,
        13/13 uncapped correct, 2 nominal frontier items caused by capped failures.
  - [x] Analyze DeepSeek-7B AIME 2025 diagnostic: 18/18 scored, 11/18 capped,
        5/7 uncapped correct, 1 clean frontier item.
  - [ ] Do not accept cap-created wrong answers as frontier evidence.
  - [ ] Prefer a smaller/easier model-dataset pair whose traces finish under the
        cap, or use the small AIME diagnostics only if the higher cap resolves the
        length artifact.
  - [ ] Do not expand either AIME diagnostic for latent capture without a cleaner
        follow-up screen; AIME 2024 is too close to saturation when uncapped, while
        AIME 2025 remains length-dominated at 16,384 tokens.
- [ ] Run all three screens.
  - [ ] Generate PolyMath medium rollouts.
  - [ ] Generate MBPP+ code solutions.
  - [ ] Generate BigCodeBench-Hard code solutions.
  - [ ] Grade generated code with the official benchmark tests in isolation.
  - [ ] Import test pass/fail results before screening classification.
  - [ ] Run `scripts/analyze.py` for the completed PolyMath run.
  - [ ] Update `experiments/dataset_saturation.csv`.
- [ ] Audit answer extraction before accepting the measurements.
  - [ ] Manually inspect 25 random scored outputs per dataset.
  - [ ] Manually inspect every unscored output, up to 50 per dataset.
  - [ ] Confirm PolyMath numeric formatting variants compare exactly.
  - [ ] Confirm code correctness comes from tests, not string matching.
  - [ ] Confirm no run has a high generation-cap rate.
  - [ ] Treat `<think>` token boundaries as tokenizer-template metadata, not as
        guaranteed semantic reasoning boundaries.
  - [ ] When the span between start-of-thinking and end-of-thinking is empty,
        inspect the post-`</think>` text as the visible reasoning trace for
        segmentation and solution-object analysis.
- [ ] Select the activation-capture dataset.
  - [ ] Prefer the dataset with the most frontier instances, not merely 50%
        aggregate accuracy.
  - [x] Require at least five frontier instances in the pilot.
  - [x] Expand the unforced P1 screen to establish a 100-question candidate
        pool and a paired baseline.
  - [x] Prepare `runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen` over all
        100 candidates with ten rollouts each and no activation capture.
  - [x] Force each screened generation to start with
        `<think> Okay, let's see`.
  - [x] Generate and summarize the forced-prefix screen: 85.7% accuracy,
        35/100 mixed questions, and 129/1000 capped generations.
  - [x] Select the 14 mixed questions with pass rate strictly below 80% using
        only forced-prefix results; all 14 have cap-affected rollouts.
  - [x] Record the selected instances in the pinned long-cap dataset.
  - [x] Generate, pull, and analyze the 8192-token latent-state run: 60.0%
        accuracy, 10/14 mixed questions, and 12/140 capped generations.
  - [x] Pin those 10 latest mixed questions in
        `runs/SmolLM3-3B/gsm_symb_prefixed_mixed_cap4096`.
  - [x] Prepare ten fresh rollouts per question with a 4096-token primary cap.
  - [x] On cap only, append the closing-thinking answer prompt and generate
        three to four final tokens.
  - [ ] Generate, pull, and analyze the cap-finalization run.
  - [x] Prepare `runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300` with 300 new
        P1 questions, five rollouts each, and no activation capture.
  - [x] Verify the pinned 300-question pool has no overlap with prior SmolLM3
        screening or latent-run samples.
  - [x] Configure two model replicas on GPUs 0 and 1 with deterministic
        contiguous instance sharding.
  - [x] Generate and summarize the 300-question screen.
  - [x] Require at least 50 mixed/frontier questions across the new screen:
        the merged inventory contains 65 mixed questions.
  - [x] Select the final activation-capture pool after comparing capped and
        uncapped outcomes: 58 questions remain mixed without capped rollouts.
  - [x] Prepare `runs/SmolLM3-3B/gsm_symb_pure_mixed_latents_10k` with ten
        rollouts per question, last-layer capture, and forced answers at the
        10,000-token cap.
  - [ ] Generate, pull, and analyze the 58-question latent-state run.
  - [ ] If no dataset passes, add one harder/easier screen rather than capturing
        weak data.

### Phase 0 Commands

```bash
python scripts/prepare_dataset.py runs/Qwen3-8B/polymath_medium_numeric_screen
python scripts/prepare_dataset.py runs/Qwen3-8B/mbppplus_codegen_screen
python scripts/prepare_dataset.py runs/Qwen3-8B/bigcodebench_hard_codegen_screen

python scripts/generate.py \
  runs/Qwen3-8B/polymath_medium_numeric_screen \
  runs/Qwen3-8B/mbppplus_codegen_screen \
  runs/Qwen3-8B/bigcodebench_hard_codegen_screen

python scripts/analyze.py runs/Qwen3-8B/polymath_medium_numeric_screen

# Run the official MBPP+/BigCodeBench test harnesses and import their results.
python scripts/summarize_screening.py \
  runs/Qwen3-8B/polymath_medium_numeric_screen \
  runs/Qwen3-8B/mbppplus_codegen_screen \
  runs/Qwen3-8B/bigcodebench_hard_codegen_screen
```

### Smaller-Model Commands

The original six datasets and the SmolLM3-3B expansion are pulled. The next
run rescreens the paired 100-question P1 pool under forced reasoning
initiation:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen
```

After pulling the run-agnostic artifact set, summarize the forced-prefix
screen, isolate its mixed questions below 80% success, and launch the prepared
long-cap capture run:

```bash
bash scripts/remote.sh pull
python scripts/summarize_screening.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_screen \
  runs/SmolLM3-3B/gsm_symbolic_p1_frontier_expand \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen
python scripts/select_mixed_samples.py \
  runs/SmolLM3-3B/gsm_symbolic_p1_forced_think_screen \
  --max-pass-rate 0.8 \
  --out runs/SmolLM3-3B/gsm_symb_prefixed_mixed/dataset.jsonl
bash scripts/remote.sh push
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed
```

After analyzing that run, isolate all newly mixed questions and launch the
cap-finalization comparison:

```bash
python scripts/select_mixed_samples.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed \
  --max-pass-rate 1.0 \
  --out runs/SmolLM3-3B/gsm_symb_prefixed_mixed_cap4096/dataset.jsonl
bash scripts/remote.sh push
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_mixed_cap4096
```

The independent 300-question frontier expansion can run separately:

```bash
bash scripts/run_with_hf_download_fix.sh python scripts/generate.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300
```

After both GPU workers finish, pull and summarize the combined run:

```bash
bash scripts/remote.sh pull runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300
python scripts/summarize_screening.py \
  runs/SmolLM3-3B/gsm_symb_prefixed_frontier_300
```

Grade MBPP+ and BigCodeBench-Hard with their official test harnesses, import
the pass/fail results, and then summarize all six runs.

### Smaller-Model Selection Evidence

- **SmolLM3-3B + GSM-Symbolic-P1:** SmolLM3 reports 83.4% on GSM-Plus in
  thinking mode. GSM-Symbolic-P1 adds one reasoning clause beyond the base
  templates, making it a plausible mixed-success step up without moving to
  long olympiad traces.
- **SmolLM3-3B + MBPP+:** SmolLM3 reports 52.91% MBPP+ for its base model and
  30.0% on LiveCodeBench in extended-thinking mode. MBPP+ provides compact
  solution construction with substantially expanded tests.
- **Qwen3-4B + PolyMath medium:** Qwen3-4B reports 54.1% on MATH as a base
  model while its thinking-mode GSM-Plus score is 88.2%. PolyMath medium spans
  university exercises, entrance exams, and low-difficulty competitions,
  placing it between those capability points.
- **Qwen3-4B + MBPP+:** Qwen3-4B reports 63.75% on MBPP+ as a base model and
  52.9% on LiveCodeBench in thinking mode, both well below saturation.
- **DeepSeek-7B + PolyMath medium:** DeepSeek-R1-Distill-Qwen-7B reports 92.8%
  on MATH-500 but 55.5% on AIME 2024. PolyMath medium is deliberately below
  AIME-heavy PolyMath high while remaining harder than grade-school math.
- **DeepSeek-7B + BigCodeBench-Hard:** The model reports 37.6% on
  LiveCodeBench. BigCodeBench-Hard contains 148 practical tasks whose observed
  solve rate is below 50%, with richer requirements and object structure than
  short algorithm-only exercises.

## Phase 1: Capture A Mixed-Success Corpus

- [ ] Freeze the selected frontier subset.
  - [ ] Include every frontier instance if there are at most 200.
  - [ ] Otherwise sample 200, stratified by pass-rate bin and subject.
  - [ ] Preserve all screening seeds for reproducibility.
- [ ] Create the primary latent run.
  - [ ] Use the selected full-precision model on the GPU server.
  - [ ] Capture the final layer to begin with.
  - [ ] Store 10-16 rollouts per instance.
  - [ ] Enable token diagnostics only on a representative 20-instance subset.
- [ ] Verify corpus balance.
  - [ ] At least 50 correct traces.
  - [ ] At least 50 incorrect traces.
  - [ ] At least 10 questions with both outcomes.
  - [ ] No single question contributes more than 5% of all traces.
  - [ ] No answer-extraction failure is counted as model failure.
- [ ] Freeze train/validation/test splits by question ID.
  - [ ] 60% train questions.
  - [ ] 20% validation questions.
  - [ ] 20% held-out test questions.
  - [ ] Never split rollouts from one question across partitions.

## Phase 2: H1 - Discover Reasoning Boundaries

- [ ] Generate matched prompt-condition traces.
  - [ ] Freeform chain of thought.
  - [ ] Forced numbered steps.
  - [ ] Sentence-separated instructions.
  - [ ] Paragraph-separated instructions.
  - [ ] Reuse the same question IDs and seed schedule.
- [ ] Implement latent boundary features.
  - [ ] Hidden-state displacement norm.
  - [ ] Adjacent displacement cosine.
  - [ ] Curvature or turning angle.
  - [ ] Token entropy.
  - [ ] Token log probability.
  - [ ] Optional logit-lens change.
- [ ] Implement change-point baselines.
  - [ ] Sentence boundaries.
  - [ ] Paragraph boundaries.
  - [ ] Numbered-step boundaries.
  - [ ] Fixed-token intervals.
  - [ ] Univariate threshold detector.
  - [ ] Multivariate change-point detector.
- [ ] Evaluate segmentation quality.
  - [ ] Boundary agreement.
  - [ ] Within-segment directional coherence.
  - [ ] Between-segment separation.
  - [ ] Bootstrap confidence intervals by question.
  - [ ] Ablate every latent feature family.
- [ ] H1 decision gate.
  - [ ] Continue only if at least one learned/latent segmentation beats sentence
        segmentation on held-out questions.

## Phase 3: H2 - Identify Functional Step Types

- [ ] Define a domain-specific functional label guide.
  - [ ] Introduce or bind a variable/concept.
  - [ ] Add a constraint or fact.
  - [ ] Derive an intermediate result.
  - [ ] Substitute, simplify, or calculate.
  - [ ] Split into cases.
  - [ ] Verify or check.
  - [ ] Backtrack or revise.
  - [ ] Extract/finalize the answer.
- [ ] Build a labeled corpus.
  - [ ] Sample 500-1,000 segments.
  - [ ] Balance correctness, position, question, and segment length.
  - [ ] Double-label at least 20%.
  - [ ] Report inter-annotator agreement.
- [ ] Train step-type baselines and probes.
  - [ ] Majority and token-position baselines.
  - [ ] Bag-of-words baseline.
  - [ ] Text embedding baseline.
  - [ ] Mean hidden-state probe.
  - [ ] Mean + variance probe.
  - [ ] Direction + nudge probe.
  - [ ] Full combined feature probe.
- [ ] Evaluate cross-question and cross-dataset transfer.
  - [ ] Macro-F1 and per-label F1.
  - [ ] Calibration.
  - [ ] Probe complexity control.
  - [ ] Layer ablation.
  - [ ] Segmenter ablation.

## Phase 4: H3 - Predict Correctness Early

- [ ] Construct prefix examples at 25%, 50%, 75%, and 100% of each trace.
- [ ] Train question-disjoint predictors.
  - [ ] Token-state baseline.
  - [ ] Sentence-mean baseline.
  - [ ] Step-mean baseline.
  - [ ] Step mean + variance.
  - [ ] Step direction + nudge.
  - [ ] Learned boundary/object-update features.
- [ ] Evaluate.
  - [ ] ROC-AUC and PR-AUC.
  - [ ] Expected calibration error.
  - [ ] Early-warning AUC by prefix.
  - [ ] Accuracy at fixed false-positive rates.
  - [ ] Cross-prompt and cross-dataset transfer.
- [ ] Demonstrate operational value.
  - [ ] Correctness-aware trace reranking.
  - [ ] Best-of-N selection.
  - [ ] Optional early stopping simulation.

## Phase 5: H4 - Test Reusable Update Types

- [ ] Cluster only on training questions.
  - [ ] Mean-state features.
  - [ ] Direction/nudge features.
  - [ ] Combined features.
  - [ ] Text-embedding control.
- [ ] Test cluster stability.
  - [ ] Bootstrap adjusted mutual information.
  - [ ] Seed stability.
  - [ ] Prompt-condition stability.
  - [ ] Layer stability.
  - [ ] Dataset transfer.
- [ ] Interpret clusters.
  - [ ] Blind-label exemplars.
  - [ ] Measure label purity.
  - [ ] Test cluster-to-label transfer on held-out questions.
  - [ ] Reject clusters explained mainly by token position or length.

## Phase 6: H5-H6 - Align Steps To Solution-Object Edits

- [ ] Choose the first explicit object domain.
  - [ ] Prefer coding once isolated test-based grading is available.
  - [ ] Otherwise use math with variables, constraints, equations, and derived
        quantities.
- [ ] Define object states and edit vocabulary.
  - [ ] Specify representation and granularity.
  - [ ] Define deterministic extraction where possible.
  - [ ] Keep human/LLM labels separate from objective artifacts.
- [ ] Build aligned examples.
  - [ ] At least 100 traces.
  - [ ] Include matched correct/incorrect traces for the same questions.
  - [ ] Label edit type and object state after every step.
- [ ] Train step-to-edit models.
  - [ ] Text-only baseline.
  - [ ] Position-only baseline.
  - [ ] Mean-state baseline.
  - [ ] Transition-feature model.
  - [ ] Shuffled-alignment control.
- [ ] Analyze divergence.
  - [ ] First object-state divergence index.
  - [ ] Missing/spurious component rates.
  - [ ] Distance to reference object over time.
  - [ ] Earliest reliable latent warning.

## Phase 7: H7-H8 - Hidden Information And Compression

- [ ] Probe partial object state from:
  - [ ] Visible text only.
  - [ ] Hidden states only.
  - [ ] Hidden states + text.
  - [ ] Final answer only.
- [ ] Predict:
  - [ ] Requirements/constraints already covered.
  - [ ] Next object edit.
  - [ ] Final object skeleton.
  - [ ] Recoverability of the partial solution.
- [ ] Test compressed control sequences.
  - [ ] All token states.
  - [ ] Sentence means.
  - [ ] Learned-step transitions.
  - [ ] High-impact steps only.
  - [ ] Cluster IDs plus scalar features.
- [ ] Report signal retained versus representation size.

## Phase 8: H9 - Causal Intervention

- [ ] Identify candidate directions on training questions only.
  - [ ] Verify/check direction.
  - [ ] Backtrack/revise direction.
  - [ ] Case-split direction.
  - [ ] Finalize direction.
- [ ] Run dose-response steering.
  - [ ] Positive and negative coefficients.
  - [ ] Random-direction controls matched by norm.
  - [ ] Off-boundary intervention controls.
  - [ ] Multiple layers.
- [ ] Measure:
  - [ ] Target edit occurrence.
  - [ ] Final correctness.
  - [ ] Semantic drift.
  - [ ] Generation length.
  - [ ] Unintended edit rates.
- [ ] Limit claims to diagnostic evidence if interventions are not selective.

## Reproducibility And Paper Gates

- [ ] Register all run configs and pinned datasets.
- [ ] Record package versions, model revisions, and quantization status.
- [ ] Use question-level bootstrap intervals for every headline metric.
- [ ] Correct for multiple comparisons in broad ablation tables.
- [ ] Keep one untouched final test split.
- [ ] Reproduce headline results with at least two generation seed schedules.
- [ ] Reproduce the main Paper 1 result on a second dataset.
- [ ] Compare at least two model sizes to test model-specificity.
- [ ] Publish negative results and failed segmentation/object definitions.
- [ ] Paper 1 gate:
  - [ ] Learned steps beat textual segmentation.
  - [ ] Transition features add signal beyond text.
  - [ ] Correctness prediction or reranking improves.
- [ ] Paper 2 gate:
  - [ ] Step transitions predict objective object edits.
  - [ ] Incorrect traces diverge earlier in object space.
  - [ ] Hidden states add information beyond visible text.

## Dataset References

- [AIME 2024 dataset](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)
- [AIME 2025 dataset](https://huggingface.co/datasets/MathArena/aime_2025)
- [OlympiadBench dataset](https://huggingface.co/datasets/Hothan/OlympiadBench)
- [PolyMath dataset](https://huggingface.co/datasets/Qwen/PolyMath)
- [GSM-Symbolic dataset](https://huggingface.co/datasets/apple/GSM-Symbolic)
- [MBPP+ dataset](https://huggingface.co/datasets/evalplus/mbppplus)
- [BigCodeBench-Hard dataset](https://huggingface.co/datasets/bigcode/bigcodebench-hard)
- [SmolLM3-3B model](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- [Qwen3-4B model](https://huggingface.co/Qwen/Qwen3-4B)
- [DeepSeek-R1-Distill-Qwen-7B model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [Qwen3-8B model](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3-14B model](https://huggingface.co/Qwen/Qwen3-14B)
