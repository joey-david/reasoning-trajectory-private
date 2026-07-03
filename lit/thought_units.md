# Token-Level Thought Units

## Claim

We test an empirical no-free-lunch claim at token resolution:

> CoT contains useful structure, but no objective-independent token partition is
> near-optimal for answer alignment, symbolic object updates, correctness
> transitions, and latent-trajectory compression.

Every generated-token transition is eligible. Sentences are one baseline, not
the segmentation lattice. This is an empirical non-dominance result; a formal
theorem would require a constrained utility family.

## Method

The implementation is
[`src/experiments/token_segmentation/`](../src/experiments/token_segmentation/)
with a thin
[`token_segmentation.py`](../scripts/experiments/token_segmentation.py) CLI.
It streams existing last-layer activations and uses a question-disjoint split.

Each trace receives the same boundary budget: its number of verified symbolic
updates, capped at 64. Primary partitions require at least four tokens per
segment; one- and eight-token constraints are sensitivity checks. Utility is
normalized so random selection has expected score 0 and the target-specific
oracle has score 1.

Objectives:

- **Answer:** positive increase in cosine alignment with the mean
  teacher-forced gold-solution state. This is not an MI/HSIC replication.
- **Object:** completion token of a deterministically verified arithmetic
  update.
- **Correctness:** absolute change in a train-only linear final-correctness
  probe.
- **Compression:** local left/right mean-state change in train-only PCA space.

Baselines include random tokens, raw magnitude/cosine/curvature peaks, sentence
boundaries, each objective oracle, and one supervised token-boundary detector
per objective.

## Results

Primary objective utility:

| Model / method | Answer | Object | Correctness | Compression |
|---|---:|---:|---:|---:|
| Smol raw magnitude | .324 | -.009 | .175 | .107 |
| Smol raw cosine | .304 | .006 | .070 | .116 |
| Smol sentences | -.051 | -.015 | .066 | .210 |
| Smol answer oracle | 1.000 | -.004 | .105 | -.034 |
| Smol object oracle | .020 | 1.000 | .061 | .089 |
| Smol correctness oracle | .093 | .008 | 1.000 | .100 |
| Smol compression oracle | .005 | .019 | .086 | 1.000 |
| Qwen raw magnitude | .139 | .080 | .157 | .134 |
| Qwen raw cosine | .299 | .038 | .156 | .070 |
| Qwen sentences | -.014 | -.018 | -.014 | .136 |
| Qwen answer oracle | 1.000 | -.016 | .246 | .084 |
| Qwen object oracle | .019 | 1.000 | .000 | .403 |
| Qwen correctness oracle | .049 | -.009 | 1.000 | .056 |
| Qwen compression oracle | .029 | .112 | .011 | 1.000 |

The result covers 310,044 held-out SmolLM tokens and 61,691 held-out Qwen
tokens. Objective-oracle F1 within four tokens is only .049-.161 on SmolLM and
.071-.187 on Qwen without a length constraint. Under the primary four-token
constraint, the ranges are .098-.210 and .130-.294. The strongest
worst-objective utility is still only .020 on SmolLM and .080 on Qwen.
Across 1/4/8-token constraints it stays within .020-.027 and .079-.085,
respectively. No tested segmentation approaches a joint optimum.

Supervised detectors recover their own ontology: held-out diagonal ROC-AUC is
.999/.984/.849/.868 for SmolLM answer/object/correctness/compression and
.998/.992/.859/.825 for Qwen. Objective scores themselves are excluded from
the detector features. This rejects “there is no latent structure.”
Their cross-objective performance and utility remain substantially lower.

Question-level bootstrap intervals, full matrices, and examples are in:

- SmolLM [report](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation/report.json),
  [matrix](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation/objective_matrix.csv),
  and [examples](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation/boundary_examples.jsonl).
- Qwen [report](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/token_segmentation/report.json),
  [matrix](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/token_segmentation/objective_matrix.csv),
  and [examples](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/token_segmentation/boundary_examples.jsonl).
- Minimum-length sensitivity:
  SmolLM [1 token](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation_gap1/report.json)
  and [8 tokens](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/token_segmentation_gap8/report.json);
  Qwen [1 token](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/token_segmentation_gap1/report.json)
  and [8 tokens](../runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/token_segmentation_gap8/report.json).

## What Survived

- Objective-specific token partitions show strong rank reversals.
- Symbolic object completions remain nearly unrelated to answer and
  correctness peaks.
- Sentence boundaries are not privileged when arbitrary token cuts are
  allowed.
- Strong supervision recovers useful target-specific structure without
  producing one generally useful partition.
- The result replicates across SmolLM3-3B and Qwen3-14B.

## What Did Not

- “There are no general latent peaks” is too strong. Raw cosine peaks recover
  .303/.299 answer utility, and raw magnitude has nontrivial answer,
  correctness, and compression utility.
- Objectives are not wholly independent. On Qwen, object-oracle compression
  utility is .403; object-to-compression and compression-to-object supervised
  AUC are .654 and .856.
- Qwen correctness evidence is weak because its held-out set has only one
  incorrect trace. Correctness conclusions should rely on SmolLM.
- The current answer objective is gold-solution alignment, not mutual
  information with the answer. It cannot directly refute MI-peak results.

The defensible claim is therefore **no universal optimum**, not absence of
shared transition signals.

## Semantic Labels

Qwen3.5-122B labeled 415 overlapping token windows from one concise correct
SmolLM trace per question. It returned complete quoted intervals with one of
nine semantic edit labels; exact matching recovered global token offsets.
The implementation is in
[`solution_object_labeling.py`](../src/experiments/solution_object_labeling.py)
and the reconciliation/evaluation code is in
[`semantic_labels.py`](../src/experiments/token_segmentation/semantic_labels.py)
and
[`semantic_evaluation.py`](../src/experiments/token_segmentation/semantic_evaluation.py).

Preparation:

```bash
.venv/bin/python scripts/experiments/prepare_solution_object_labels.py \
  runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k \
  runs/Qwen3.5-122B-A10B-FP8/solution_object_silver/token_windows.jsonl \
  --updates runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/updates.jsonl
```

The FP8 checkpoint uses vLLM's Ampere-compatible W8A16 Marlin path. Run the
one-record end-to-end smoke before scheduling the full queue. vLLM 0.21 must
come from its CUDA 12.9 index; its default wheel requires CUDA 13.

```bash
uv pip install --python .venv/bin/python vllm \
  --extra-index-url https://wheels.vllm.ai/0.21.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

.venv/bin/python scripts/orchestrate.py \
  --job solution_object_labeling_smoke \
  --nodes local --devices 0+1 \
  --run runs/Qwen3.5-122B-A10B-FP8/solution_object_silver
```

### Semantic Results

The judge returned all 415 windows; 406 (97.8%) passed exact alignment. After
dropping 648 intervals touching internal window edges, reconciliation retained
1,761 unique intervals over 58 traces, covering 75.8% of tokens. Median interval
length is 29 tokens. The 27 exact intervals repeated across windows received
the same label 88.9% of the time; this is useful but limited redundancy.

Most boundaries remain sentence-related: 84.7% lie within four tokens of a
sentence end. However, 57.7% of intervals combine material across at least one
sentence boundary, and 21.8% of held-out boundaries are not sentence-aligned.
A question-disjoint linear detector reaches ROC-AUC .982 for all semantic
boundaries (question-bootstrap 95% CI [.976, .987]) and .969 for only the
non-sentence subset ([.958, .980]). Raw magnitude/cosine reach .791/.804
overall but only .595/.684 off sentence boundaries. A matched text-only
character n-gram detector reaches .956 overall and .775 off sentence
boundaries. In paired question-level comparisons, latent features exceed text
by .026 AUC overall (95% CI [.015, .040]) and .193 off sentence boundaries
([.115, .301]).

Eight semantic operation types are decodable from held-out span activations:
macro one-vs-rest ROC-AUC is .891 and macro-F1 is .487 over 374 spans, versus
.070 majority macro-F1. This is meaningful but not latent-exclusive: lexical
TF-IDF reaches .892 AUC, their combination .901, and unsupervised PCA-cosine
only .623. The activations contain the ontology, but supervision and textual
cues select it.

The new ontology strengthens rather than overturns non-dominance. Its oracle
has utility -.012/.043/.036/.324 on answer/object/correctness/compression;
the first three 95% intervals stay near zero, while compression is
[.248, .398].
The four existing oracles obtain only .018-.166 semantic utility. The best
worst-objective utility is .011 with a four-token minimum and remains
.024/.020 at one/eight tokens. The learned semantic detector recovers .651
semantic utility but does not transfer to answer or symbolic-object utility.

Artifacts: [report](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation/report.json),
[matrix](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation/objective_matrix.csv),
[reconciled spans](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation/reconciled_spans.jsonl),
and minimum-length sensitivity at
[one](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation_gap1/report.json)
and
[eight](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/semantic_token_segmentation_gap8/report.json)
tokens.

## Next

- Manually audit a stratified semantic-label sample; Qwen labels are silver,
  not ground truth.
- Replace the answer proxy with prompt-conditioned HSIC/MI estimation.
- Repeat labeling on a non-arithmetic task and another activation model.

The semantic result has only 12 held-out traces, all from GSM-Symbolic and all
selected for correct concise reasoning. It supports an arithmetic,
sentence-adjacent token-level result, not yet a task-general theorem.

## References

- Yu et al., [State-Aware Reasoning Dynamics](https://arxiv.org/abs/2509.00190).
- Qian et al., [Thinking Tokens Are Information Peaks](https://arxiv.org/abs/2506.02867).
