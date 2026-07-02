# No Free Lunch on the Sentence Lattice: Objective-Relative CoT Thought Units

## Aim

We first test whether chain-of-thought has a **canonical segmentation over the
sentence lattice**, or whether useful boundaries are selected by an objective.
The claim is not that CoT lacks structure: it is that no context-free grouping
of sentences is simultaneously privileged for answer information, symbolic
updates, correctness prediction, latent-trajectory compression, and causal
influence.

Let \(S\) partition a trace at sentence boundaries and let
\(\mathcal{U}=\{U_{\mathrm{answer}},U_{\mathrm{object}},
U_{\mathrm{correctness}},U_{\mathrm{compression}},U_{\mathrm{causal}}\}\).
The empirical no-free-lunch claim is:

\[
\nexists S^\star \text{ that is near-optimal for every } U\in\mathcal{U}
\text{ across tasks, prompts, and models.}
\]

Evidence should take the form of **rank reversals, cross-objective regret, and
transfer failure**, not merely weak performance by one heuristic.

## Status

The locally feasible first pass is implemented in
[`src/experiments/thought_units.py`](../src/experiments/thought_units.py), with
exact fixed-budget partitioning and metrics in
[`src/experiments/sentence_lattice.py`](../src/experiments/sentence_lattice.py)
and a thin
[`scripts/experiments/thought_units.py`](../scripts/experiments/thought_units.py)
entry point. It streams the completed last-layer activations, fits PCA and all
probes on training questions only, and evaluates held-out questions for both
SmolLM3-3B (580 traces) and Qwen3-14B (232 traces).

Primary artifacts:

- [full report](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/report.json),
  [score matrix](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/objective_matrix.csv),
  and [regret plot](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/regret_matrix.png);
- [all primary partitions](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/partitions.jsonl)
  and [text examples](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/boundary_examples.jsonl);
- [supervised transfer matrix](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/supervised_transfer.csv);
- [question-disjoint prompt-transfer report](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/prompt_transfer.json);
- Qwen3-14B [full report](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/thought_units/report.json),
  [score matrix](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/thought_units/objective_matrix.csv),
  [H2 report](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/h2_localized_updates/report.json),
  and [H4 report](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/h4_structural_contrast/report.json);
- teacher-forced gold-solution captures for
  [SmolLM3-3B](../runs/SmolLM3-3B/thought_units_gold_answers/gold_answers/manifest.jsonl)
  and [Qwen3-14B](../runs/Qwen3-14B/thought_units_gold_answers/gold_answers/manifest.jsonl);
- [completed causal-intervention report](../runs/SmolLM3-3B/thought_units_boundary_interventions/analysis/report.json);
- small prompt pilots:
  [freeform](../runs/SmolLM3-3B/h1_freeform_replay/analysis/experiments/thought_units/report.json),
  [numbered](../runs/SmolLM3-3B/h1_numbered_steps_pilot/analysis/experiments/thought_units/report.json),
  and [sentence-separated](../runs/SmolLM3-3B/h1_sentence_separated_pilot/analysis/experiments/thought_units/report.json).

## Scope

**Sentence-lattice segmentation.** Given a fixed sentence parser, use sentences
as the only atomic units. Segmenters may select boundaries between consecutive
sentences or merge adjacent sentences. Represent each sentence from last-layer
token activations using a common feature family: mean, endpoint, transition,
and covariance/Gram spectrum.

This is a sentence-lattice no-free-lunch result, not a token-level theorem. A
negative result at this human-legible granularity motivates, but does not
establish, a broader claim about token-level thought units.

Evaluate each candidate against five objectives:

- **Answer information:** HSIC or cross-validated probe dependence on the gold
  answer from a fixed-dimensional segment representation.
- **Object updates:** coverage of exactly one symbolic graph update, or boundary
  alignment with \(G_k\rightarrow G_{k+1}\).
- **Correctness:** improvement in held-out final-correctness prediction from a
  segment-prefix representation.
- **Latent-trajectory compression:** reconstruction error for token activations
  from fixed-dimensional segment summaries, penalized by segment count.
- **Causal influence:** continuation changes caused by matched interventions at
  selected boundaries versus non-boundary controls. This is an evaluation
  column, not an oracle objective, in the first pass.

The implemented local objectives are exact fixed-segment dynamic programs:
answer/correctness/compression minimize within-segment squared error in their
respective curves or representations, while the object objective penalizes
segments containing anything other than one verified symbolic update.
Normalized utility is \(0\) for matched random boundaries and \(1\) for that
objective's oracle; negative values are worse than random.

Compare textual sentence boundaries, answer-information peaks, Gram-state
transitions, objective-specific dynamic-programming oracles, and a supervised
change-point model. Report a segmenter-by-objective score matrix, boundary
agreement, normalized regret, Pareto dominance, and transfer across prompts
and held-out questions.

## Experimental Controls

### Boundary Budget

- Give every method the same number of boundaries in the primary comparison.
- Select objective weights on train questions and score held-out questions.
- Sweep segment counts or MDL penalties and require conclusions to be stable.
- Compute segment scores only from the segment or its immediate boundary; do
  not leak full-trace or held-out gold-answer information into boundary
  selection.
- Give every objective the same representation feature family and dimension.

Diagonal oracle performance is expected and is not evidence. The result of
interest is cross-objective regret and non-dominance under matched budgets.

### Projection Spaces

Evaluate every segmenter in:

1. raw last-layer activation space;
2. PCA-whitened last-layer space;
3. Gram-spectrum space;
4. the H4 operation-supervised projection space.

Where feasible, add an answer-information projection fitted only on training
questions. This separates “no useful segmentation” from “useful structure is
hidden by the coordinate system.”

The primary answer signal now uses normalized linear-kernel HSIC against the
mean teacher-forced gold-solution state. The captures start from a lone BOS/EOS
token, not the question prompt. They therefore measure alignment with a
canonical solution text; they are **not** a Gaussian-HSIC or mutual-information
replication of Qian et al. The older cross-rollout answer proxy is retained as
a diagnostic.

### Baselines

- whole trace, with no internal segmentation;
- every sentence as a segment;
- fixed-size consecutive sentence windows;
- random matched-budget boundaries;
- unmerged sentence-parser boundaries;
- answer-information/HSIC peaks;
- Gram-state transitions;
- objective-specific dynamic-programming oracles;
- supervised change-point detection.

## Results and Running Notes

The primary \(20\%\) boundary-budget matrix shows clear rank reversals:

| Segmenter | Answer | Object | Correctness | Compression |
| --- | ---: | ---: | ---: | ---: |
| Fixed windows | 0.004 | 0.379 | 0.307 | 0.215 |
| Answer peaks | **0.734** | -0.285 | -0.320 | 0.292 |
| Gram transitions | -0.010 | -3.370 | -0.161 | -0.484 |
| Answer oracle | **1.000** | -0.411 | -0.306 | 0.401 |
| Object oracle | -0.046 | **1.000** | -0.174 | -0.267 |
| Correctness oracle | -0.001 | -0.482 | **1.000** | -0.114 |
| Compression oracle | 0.415 | -0.297 | -0.044 | **1.000** |

Exact SmolLM oracle-boundary Jaccard is \(0.10\)–\(0.27\); one-sentence
tolerance raises F1 only to \(0.34\)–\(0.55\). Fixed windows are the best
maximin method, but their worst utility is only \(0.004\), nowhere near a
joint optimum. The same qualitative reversals survive \(10\%\), \(20\%\), and
\(30\%\) budgets.

Qwen3-14B independently reproduces the result. Its answer peaks score \(0.626\)
on answer but \(-0.603\) on objects; its object oracle scores \(1.000\) on
objects but \(-0.128\) on answer and \(-0.211\) on correctness. Oracle Jaccard
is \(0.09\)–\(0.17\), and the best maximin utility is only \(0.030\). Qwen is
225/232 correct, so its held-out correctness AUC is undefined and correctness
claims should rest on SmolLM rather than this ceiling-limited run.

The strong adversary does recover its training ontology. In PCA-whitened
space, held-out diagonal ROC-AUC is \(0.889\) answer, \(0.884\) object,
\(0.786\) correctness, and \(0.861\) compression. Transfer is poor or
anticorrelated. Qwen gives the same pattern: diagonal AUC is
\(0.830/0.878/0.774/0.844\), while mean cross-objective performance is only
\(56\%\)–\(75\%\) of in-domain AUC in PCA space. This supports “learnable
ontology” rather than “no latent structure.”

The Yu-style five-state accumulated-Gram abstraction is strongly
position-ordered: cluster mean positions span \(0.03,0.16,0.36,0.56,0.74\),
with only \(3.1\%\) of sentence transitions changing state. On this much finer
sentence lattice, its matched-budget boundaries are worse than random on
answer, object, and compression. This does not refute its use on prompted
coarse steps; it shows that the abstraction is not a general sentence boundary
rule here.

The BOS-only gold-solution target is not sharply peaked. Score-position
correlation is \(0.049\) for SmolLM and \(0.277\) for Qwen; the IQR peak rate is
effectively zero for both. The old proxy and new curves correlate
\(0.564/0.785\), but their top-boundary Jaccard is only \(0.378/0.372\).
Target construction therefore changes which boundaries count as peaks. This
supports target relativity, but the capture limitation prevents a direct
claim against Qian-style prompt-conditioned MI peaks.

Parser audit: all 580 traces align exactly to tokens; 84,338 sentences produce
83,758 candidate boundaries. Short fragments are \(3.3\%\) of sentences and
standalone list markers \(0.7\%\). They remain unmerged in the primary result.
A secondary lattice merges all one- and two-token fragments, reducing the
lattice to 81,554 sentences. The result is stable: oracle Jaccard remains
\(0.11\)–\(0.26\); answer peaks score \(0.741\) on answer but \(-0.376\) on
objects and \(-0.099\) on correctness; no method is near-optimal across all
objectives.

Prompt pilots reproduce the major oracle disagreements for freeform, numbered,
and sentence-separated prompts, but each has only 12 questions and three
held-out questions. The interrupted paragraph run has five questions and an
all-correct held-out split, so no correctness result is reported.

The stronger prompt-transfer test excludes those 12 questions from training,
leaving 46 source questions and 66,818 boundaries in the shared H4 space.
In-domain ontology transfer survives prompt changes: answer/object/correctness/
compression AUC is respectively \(0.640/0.730/0.752/0.659\) on freeform,
\(0.611/0.738/0.753/0.653\) on numbered, and
\(0.645/0.711/0.722/0.639\) on sentence-separated traces. Cross-ontology scores
remain much weaker, commonly \(0.35\)–\(0.59\). Prompt variation therefore does
not erase the learned ontology, but neither does it make one ontology recover
the others.

Qwen also replicates the earlier latent-structure pattern. Across 3,711
symbolic updates, path length is elevated (78th matched-window percentile),
but peak share is only \(0.131\), effective width is \(0.983\), and net/path
ratio is \(0.110\): updates are distributed rather than point-like. Operation
identity rises from raw cosine AUC \(0.433\) to \(0.995\) after a supervised
projection. Structure is highly decodable, but not naturally organized as
universal boundaries in the raw space.

The causal run is complete: 1,160 rows give 580 paired interventions over all
58 questions. Zeroing layer-18 attention output changes the extracted answer
in \(32.9\%\) of all pairs, confirming broad behavioral sensitivity. However,
183 pairs have at least one unfinished 10k-token continuation; numeric fallback
extraction makes their apparent correctness unreliable. Among the 397 complete
pairs, answer changes fall to \(9.3\%\). With the same completion filter applied
to each position-matched random control, answer-boundary accuracy specificity
is \(-0.011\), 95% CI \([-0.076,0.054]\); compression and correctness are also
near zero, while object specificity is \(-0.073\), CI
\([-0.159,0.000]\). The object direction is consistent with greater causal
importance, but remains weak (two-sided question-level sign-flip \(p=0.157\);
one-sided \(p=0.078\)) and is driven by only six nonzero question effects.
The other families are indistinguishable from random.

The answer manifest also predates gold-solution rescoring: only 53/116 answer
points remain in the current answer oracle. The 28 completed, random-matched
points in that overlap have exploratory specificity \(+0.038\), CI
\([0.000,0.115]\), but this is a small post-hoc subset. A confirmatory rerun
must rebuild the manifest from the current partition and reject unfinished
continuations during scoring.

## H1: Information Peaks Are Target-Relative

**Claim:** Transitions between sentences that peak for one target do not
consistently peak for another.

**Experiment:** For the boundary after sentence \(i\), compute
\(\Delta\mu_i=\mu_{i+1}-\mu_i\), target-score changes such as
\(\Delta\mathrm{HSIC}_i\), and
\(1-\cos(\Delta\mu_{i-1},\Delta\mu_i)\). Estimate transition scores for the
gold answer, symbolic operation, and final correctness, then compare peak sets
under matched boundary budgets.

**Metrics:** Boundary F1/Jaccard, rank correlation, peak overlap, and
cross-target regret.

**Null:** The same sentence transitions are informative across targets.

**Positive result:** Each target yields reproducible peaks, but peak locations
and rankings disagree across targets.

**Current evidence:** Supported for BOS-only gold-solution alignment versus
object, correctness, and compression targets on both models. A
prompt-conditioned Gaussian-HSIC replication remains pending.

## H2: No Sentence Segmentation Dominates Across Objectives

**Claim:** The best sentence partition depends on its utility.

**Experiment:** Fit objective scoring functions on training questions, then
apply penalized dynamic programming to held-out traces using only local
features. Optimize separate partitions for object alignment, answer
information, correctness, and latent-trajectory compression. Evaluate each
oracle under every other objective, including causal influence where
interventions are available, using equal boundary and representation budgets.

**Metrics:** Cross-objective regret matrix, variation of information, boundary
Jaccard, and Pareto dominance.

**Null:** One partition is near-optimal for all objectives.

**Positive result:** Oracles disagree structurally and exhibit stable rank
reversals; no partition is Pareto-dominant.

**Current evidence:** Supported on SmolLM3-3B and independently replicated on
Qwen3-14B across all three tested boundary budgets. Task transfer remains open.

## H3: Learned Boundaries Recover Ontologies, Not Universal Steps

**Claim:** Supervision can recover a useful sentence-boundary scheme, but that
scheme transfers poorly when the target ontology changes.

**Experiment:** Train a strong change-point adversary on one objective using
last-layer sentence features, then test question-, prompt-, objective-, and
eventually model/task-transfer without retraining.

**Metrics:** In-domain and transfer boundary F1, calibration, target regret,
and degradation relative to objective-specific retraining.

**Null:** A detector trained for one objective recovers the others and
transfers without material loss.

**Positive result:** High in-domain performance coexists with objective- or
domain-specific degradation, showing recoverable structure without canonical
steps.

**Current evidence:** Supported for objective transfer on held-out questions
and for question-disjoint transfer across three prompt conditions. Model/task
transfer is not yet tested.

## H4: Projection Reveals Structure but Not Canonical Segmentation

**Claim:** Learned projections can expose objective-relevant structure without
producing a segmentation that is privileged across objectives.

**Experiment:** Evaluate all segmenters in raw, PCA-whitened, Gram-spectrum,
answer-information, and operation-supervised spaces using identical splits and
boundary budgets.

**Metrics:** Segment coherence and separation, cross-objective regret, transfer
F1, and Pareto dominance.

**Null:** One projection reveals a segmentation that remains near-optimal
across objectives.

**Positive result:** Projections improve their target objective but do not
yield a transferable, general-purpose segmentation.

**Current evidence:** PCA is the strongest general detector space. The H4
operation projection improves object-boundary decoding over Gram space
(\(0.783\) versus \(0.671\) AUC) but is weaker than PCA (\(0.884\)) and does
not yield broadly coherent partitions. On Qwen, operation identity itself is
almost perfectly decodable after projection (AUC \(0.995\)), while
operation-space object-boundary AUC \(0.813\) still trails PCA \(0.878\).

## Interpretation and Failure Mode

The decisive result is not “all segmenters fail,” but “different defensible
utilities select incompatible sentence partitions.” Conversely, agreement
among independent objectives, low cross-objective regret, and robust transfer
would reject the sentence-lattice no-free-lunch claim. One partition that stays
near-optimal across objectives, projections, prompt conditions, and held-out
questions becomes a positive candidate canonical segmentation rather than a
failed experiment.

Existing results motivate both outcomes: symbolic updates are distributed;
operation type moves from raw cosine AUC \(0.394\) to \(0.957\) after supervised
projection; and sentence summaries outperform proposed symbolic and
sustained-change segments for correctness prediction
([current results](hypotheses_summary.md)).

## Related Work

- Yu et al., *Explainable Chain-of-Thought Reasoning: An Empirical Analysis on
  State-Aware Reasoning Dynamics* ([arXiv:2509.00190](https://arxiv.org/abs/2509.00190)).
  It embeds text-defined CoT steps through accumulated Gram spectra, clusters
  them into latent states, and models transitions as a Markov chain. We test
  whether those transitions remain useful across objectives.
- Qian et al., *Demystifying Reasoning Dynamics with Mutual Information:
  Thinking Tokens are Information Peaks in LLM Reasoning*
  ([arXiv:2506.02867](https://arxiv.org/abs/2506.02867)). It estimates
  dependence between intermediate representations and the gold answer with
  HSIC and identifies answer-relative information peaks. We test whether such
  peaks define general boundaries or target-specific importance.

## Todo

For the SmolLM3-3B pass, every item below operates on sentences or consecutive
sentence windows; prompted-step and token-level variants remain deferred.

- [x] Capture teacher-forced gold-solution states and compare their linear-HSIC
  curves with the cross-rollout proxy.
- [ ] Capture prompt-conditioned gold-answer states and reproduce Gaussian
  HSIC before claiming an MI-peaks replication.
- [x] Add accumulated Gram-spectrum states on sentences.
- [x] Add exact fixed-budget dynamic-programming oracles for object, answer,
  correctness, and compression objectives.
- [x] Add nonlinear supervised change-point adversaries with held-out-question
  objective transfer.
- [x] Produce the segmenter × objective matrix and Pareto/regret plots.
- [x] Repeat coherence and separation tests in raw, PCA-whitened,
  Gram-spectrum, and H4 operation-supervised spaces.
- [x] Match boundary budgets, split by question, and sweep 10/20/30% budgets.
- [x] Run small freeform, numbered, and sentence-separated prompt pilots and a
  question-disjoint H4 detector-transfer test.
- [x] Add a short-fragment parser-robustness condition without replacing the
  primary fixed parser.
- [x] Run causal interventions at independently selected boundary families
  (1,160/1,160 rows).
- [ ] Rebuild causal points from the current gold-solution partition and use
  strict completion-aware scoring.
- [x] Replicate the sentence-lattice matrix on Qwen3-14B.
- [ ] Test detector transfer across models and segmentation transfer across
  task families.
- [ ] Attempt token-level units only after the sentence-lattice result survives
  the missing controls above.

## Remote Confirmation Runs

Pulled run status:

| Run | Status |
| --- | ---: |
| `SmolLM3-3B/thought_units_gold_answers` | 58/58 |
| `Qwen3-14B/thought_units_gsm_symbolic` | 232/232 |
| `Qwen3-14B/thought_units_gold_answers` | 58/58 |
| `SmolLM3-3B/thought_units_boundary_interventions` | 1,160/1,160 |

The causal queue uses two position-matched points from each answer, object,
correctness, compression, and random partition on all 58 questions. Each point
gets a deterministic baseline and a zero-ablation of attention output at layer
18. It tests boundary-family causal specificity without process-isomer or
single-spike assumptions. All listed queues are complete.
