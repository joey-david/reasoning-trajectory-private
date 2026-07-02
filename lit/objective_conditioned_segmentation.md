# From Universal Steps to Useful Steps

## Goal

Chain-of-thought does not appear to have one privileged sentence partition.
The positive claim is that a useful partition is defined by an objective,
representation, and boundary budget:

$$
S^\star_{U,B}=\arg\max_{S\in\mathcal S_B} U(S;x,h,y).
$$

We call this **Objective-Conditioned Segmentation (OCS)**. For additive
sentence-segment utilities, OCS uses exact fixed-budget dynamic programming.
Every candidate is reported by oracle-relative regret, where zero is the
objective oracle and one is a matched-budget random partition. The immediate
application is solution-object construction: segments should isolate typed
edits to a partial mathematical solution rather than generic textual steps.

This remains a sentence-lattice result. Token/clause hierarchies and task-family
transfer are extensions, not evidence already established here.

## Contributions

1. **OCS:** a small reusable API for exact objective-conditioned partitions and
   normalized regret, backed by the existing sentence-lattice DP.
2. **SO-GSM:** a compact benchmark built from completed GSM-Symbolic traces.
   Bronze edits are deterministic; silver fields may be proposed but must pass
   deterministic arithmetic/state validation; gold is a sampled audit queue,
   not an unaudited label claim.
3. **Objective-specific causality:** rebuild intervention points from current
   partitions and score each family on its own outcome, with strict completion
   filtering and position-matched random controls.
4. **Transfer:** test one non-GSM task family after the local method and
   benchmark contracts are stable.

## Method

For local segment cost $c(i,j)=-u(i,j)$, OCS computes

$$
F(k,j)=\min_{i<j} F(k-1,i)+c(i,j)
$$

with exactly $k$ nonempty segments. For candidate $S$, lower cost is better
and normalized regret is

$$
\widetilde R_U(S)=
\frac{C_U(S)-C_U(S^\star_U)}
{\mathbb E[C_U(S_{\rm random})]-C_U(S^\star_U)}.
$$

Thus the oracle has regret $0$, matched random has expected regret $1$, and
values above one are worse than random. All methods share the same boundary
budget and held-out questions.

### Solution-object edits

The deterministic object state contains bindings, derived relations,
verification events, and terminal extraction. Each accepted edit records its
type, arithmetic operation, before/after graph signatures, added/removed
relations, numeric quantities, textual/token span, and verification status.
Object utility rewards edit purity and coverage while penalizing mixed edits
and splits. Entity and semantic-role fields remain silver annotations until
verified; deterministic arithmetic labels remain bronze.

## Progress

- [x] Existing exact fixed-budget DP and objective matrix established on
      SmolLM3-3B and Qwen3-14B.
- [x] Extract the OCS/regret contract from the monolithic experiment code.
- [x] Build and inspect the compact deterministic SO-GSM benchmark.
- [x] Add silver proposal validation and a reproducible gold audit sample.
- [x] Train one objective-conditioned boundary model and compare it with four
      separately trained models.
- [x] Rebuild causal manifests from current partitions and add
      objective-specific outcomes.
- [ ] Run the clean confirmatory causal intervention.
- [x] Prepare a pinned 50-question Qwen3-14B MATH-algebra transfer run.
- [ ] Run and analyze the MATH-algebra transfer.

## Running Notes

- Existing results already establish rank reversal: objective oracles score
  $1.0$ on their own normalized utility but incur substantial cross-objective
  regret. The contribution here is the reusable positive method, not another
  restatement of non-dominance.
- The current arithmetic extractor is intentionally conservative. Missing an
  edit lowers benchmark coverage; accepting an invalid edit corrupts the
  objective. Precision therefore takes priority for bronze labels.
- The earlier answer intervention manifest is stale relative to gold-solution
  rescoring. It must not be reused for confirmatory claims.

### Local implementation

- OCS now lives in
  [`objective_segmentation.py`](../src/experiments/objective_segmentation.py).
  It exposes exact fixed-budget optimization, normalized regret, and objective
  identity features without duplicating the sentence-lattice DP.
- The 2,381-line thought-unit implementation was split by ownership into cache,
  features, signals, partitions, probes, outputs, and shared types. Generation
  and causal patching were similarly split. Every Python file is below 1,000
  lines; the new modules are below 500.
- The local [SO-GSM report](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/solution_object_benchmark/report.json)
  contains 50 questions, 50 concise correct traces, 3,218 sentence candidates,
  and 761 deterministic edits. Only 576 sentences have bronze edits. A targeted
  audit found 215 unlabelled word-form equation candidates and 137 unlabelled
  answer statements, confirming that bronze is a high-precision anchor rather
  than adequate full coverage.
- The initial benchmark uses one trace per question rather than the proposed
  four to eight. These freeform traces are unusually long; four traces would
  exceed the intended compact labeling budget before adding semantic labels.
  Route diversity should be added only after the one-trace silver/gold audit
  establishes label quality.
- Bronze variable bindings now preserve variable identity, so `N=5` and
  `cost=5` are distinct object edits. Unit-shaped regex matches such as
  `kg=75` are rejected. The
  [gold queue](../runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/solution_object_benchmark/gold_audit_queue.jsonl)
  contains 120 stratified records explicitly marked pending.
- The richer typed-edit cost is already consistent with the previous object
  objective: its exact oracle has regret $0$, while the old object oracle has
  mean regret $0.009$. Fixed windows reach $0.601$, random $0.965$,
  answer peaks $1.244$, and the answer oracle $1.331$. Thus typing validates
  and sharpens the object objective without manufacturing a different result.
- One shared nonlinear model conditioned on objective ID nearly matches four
  separately trained models. In PCA space its answer/object/correctness/
  compression AUCs are $0.876/0.861/0.774/0.854$, versus
  $0.889/0.884/0.786/0.861$ separately. Gram-space differences are at most
  $0.004$; H4-space differences are $0.001$–$0.044$. This is positive
  evidence that one callable segmenter can select different useful units by
  objective. The raw-space multi-head model reaches
  $0.891/0.877/0.743/0.840$, but its heads do not share decision weights and
  should be treated as an architecture-matched upper bound, not compression.
- Re-scoring the old intervention run with objective-specific outcomes remains
  negative: target-minus-random effects are $-0.011$ for answer change,
  $+0.006$ for object-state Jaccard disruption, $+0.013$ for correctness
  harm, and $-0.036$ for continuation disruption; every confidence interval
  crosses zero. This is exploratory because the old answer manifest is stale.
- A fresh 580-point manifest is prepared under
  `runs/SmolLM3-3B/thought_units_objective_causality/`. It uses current
  partitions, strict complete-only analysis, the H2-supported layer-18
  attention component, and a 2,048-token cap.
- The transfer run is pinned at
  `runs/Qwen3-14B/thought_units_math_algebra/`: 50 MATH algebra questions, two
  rollouts each, last-layer int8 capture, and a 16k generation ceiling.

### Silver-labeling decision

Use `deepseek-v4-flash` for **proposals**, not ground truth. It supports JSON
output and is cheap enough to label all 3,218 candidates. Every proposed
quantity must occur in the sentence, arithmetic equalities must pass the
restricted evaluator, and rejected rows remain visible. The labeler is
resumable and receives adjacent-sentence context.

Alternatives:

- deterministic-only labels have the strongest precision and reproducibility
  but demonstrably miss common word-form relations and answer statements;
- local SmolLM3-3B avoids external data transfer, but at roughly 20 tokens/s a
  full pass takes hours and its schema/semantic consistency is the weakest;
- DeepSeek V4 Pro may improve difficult semantic roles, but costs roughly three
  times more and should be reserved for disagreements found in the gold audit;
- human labeling is the authority for the 120-record gold subset, not a
  scalable first pass.

The API integration is ready in
[`solution_object_labeling.py`](../src/experiments/solution_object_labeling.py),
but no API call has been made because `DEEPSEEK_API_KEY` is not currently set.

## Next Runs

Silver proposals:

```bash
export DEEPSEEK_API_KEY=...
.venv/bin/python scripts/experiments/solution_object_benchmark.py \
  runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k \
  --updates runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/updates.jsonl \
  --partitions runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/thought_units/partitions.jsonl \
  --label-silver
```

The remote queue already ends with the MATH transfer generation and fresh
objective-causality intervention:

```bash
./scripts/remote.sh push
./scripts/experiments/run_thought_units_remote.sh
```

After pulling those two runs:

```bash
.venv/bin/python scripts/experiments/thought_units.py \
  runs/Qwen3-14B/thought_units_math_algebra \
  --projection runs/Qwen3-14B/thought_units_gsm_symbolic/analysis/experiments/h4_structural_contrast/layer-1_projection.pt

.venv/bin/python scripts/experiments/prepare_boundary_interventions.py \
  runs/SmolLM3-3B/thought_units_objective_causality --analyze
```

## References

- Yu et al., _Explainable Chain-of-Thought Reasoning: An Empirical Analysis on
  State-Aware Reasoning Dynamics_, [arXiv:2509.00190](https://arxiv.org/abs/2509.00190).
- Qian et al., _Demystifying Reasoning Dynamics with Mutual Information:
  Thinking Tokens are Information Peaks in LLM Reasoning_,
  [arXiv:2506.02867](https://arxiv.org/abs/2506.02867).
- DeepSeek, [API quick start](https://api-docs.deepseek.com/) and
  [model/pricing table](https://api-docs.deepseek.com/quick_start/pricing).
- Background experiments and numerical evidence:
  [`thought_units.md`](thought_units.md).
