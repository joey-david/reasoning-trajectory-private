Yes — the split is strategically correct. But I would **not** make Paper 1 a mostly-theory paper. I would make it a **definition + operationalization + small empirical win** paper. Pure definitions are hard to sell unless they immediately clarify a live confusion in the literature.

Important warning: the term **“solution object” is no longer pristine**. A June 2026 paper, _Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces_, already uses “solution object” in a structured optimization setting: the model emits JSON actions, a parser/checker validates them, and a transition function updates the structured state; they explicitly say the model “never directly edits the full solution object.” ([arXiv][1]) That is useful, not fatal. It means your novelty should be framed as **implicit/latent solution-object construction from natural CoT**, not merely “we introduce solution objects.”

My suggested split:

## Paper 1: Define reasoning steps as solution-object updates

Working title:

**Reasoning Steps as Solution-State Updates: A Framework for Segmenting Chain-of-Thought**

Core claim:

> A CoT step is not a sentence, newline, or numbered item. It is a minimal contiguous generation interval that performs one coherent update to a represented solution state.

This paper should attack the current weakness in the field: trajectory papers show that CoT traces traverse structured representation spaces, but the “step” boundary is usually inherited from formatting or heuristics. The 2026 trajectory paper characterizes CoT as structured movement through representation space and finds ordered step-specific subspaces, but it still largely treats steps as already-given units of analysis. ([arXiv][2]) The state-aware dynamics paper clusters step embeddings into latent states and Markov transitions, but again starts from segmented reasoning steps rather than solving the step-definition problem itself. ([arXiv][3])

So Paper 1’s contribution is:

[
\text{reasoning step} \neq \text{text span}
]

but rather:

[
\text{reasoning step}\_i = [t_a,t_b]
]

such that:

[
\Delta h_i = h_{t_b} - h_{t_a}
]

predicts an update:

[
\Delta s_i = s_i - s_{i-1}
]

to a partial solution object.

The paper can define three levels:

**Textual step:** sentence, newline, paragraph, or prompted `Step k`.

**Latent step:** a contiguous interval with coherent hidden-state motion and detectable boundaries via direction/curvature/logit-lens/entropy changes.

**Solution-object step:** a latent step that corresponds to one coherent update to the partial solution state: add a constraint, derive a quantity, open a case, close a case, implement a branch, verify a condition, revise a prior claim, extract final answer.

For empirical content, Paper 1 should not try to solve the full mapping. It should show:

1. Prompt-forced steps, sentence steps, and latent-change-point steps are measurably different.
2. Latent/object-update segmentation predicts correctness better than naive sentence/newline segmentation.
3. Step features such as mean, direction, nudge, curvature, and variance cluster into reusable update types.
4. Maybe a small intervention or selection result: traces with cleaner solution-update decomposition are more likely to be correct.

A small performance improvement could be something like: use solution-step segmentation to improve process reward modeling, correctness prediction, trace reranking, or early stopping. It does not need to beat frontier systems. It just needs to show the definition has operational bite.

## Paper 2: Full mapping from CoT to solution-object construction

Working title:

**Chain-of-Thought as Latent Solution-Object Construction**

Core claim:

> Natural-language CoT is a lossy projection of an internal solution-construction process; latent step transitions align with edits to explicit or latent solution objects.

This is the bigger paper. It should actually build the map:

[
\Delta h_i \rightarrow \Delta s_i
]

where (s_i) can be:

For math: equation graph, variable binding graph, constraint set, derived quantity set, proof-state-like object.

For coding: AST, test-coverage vector, requirement-coverage graph, control-flow skeleton, branch/case set.

For spatial/optimization tasks: structured state/action objects, closer to the June 2026 optimization-like reasoning paper’s setup, except you infer the construction from natural reasoning rather than forcing valid JSON actions. ([arXiv][1])

This paper asks stronger questions:

Can we reconstruct partial solution objects from hidden states?

Can we align natural CoT steps to object edits without prompting `Step 1`, `Step 2`, etc.?

Do correct and incorrect CoTs diverge because they construct different latent solution objects?

Can the final solution representation be compressed into a short sequence of latent updates?

Can interventions on latent update directions change which solution object gets constructed?

This is where your “compression/optimization” idea belongs. I would phrase it less as “compression of inner representations via CoT” and more as:

> CoT induces a low-dimensional, temporally ordered basis for constructing solution representations.

Or:

> The visible CoT trace acts as a sparse control sequence over a richer latent solution object.

That is stronger and cleaner.

## Why two papers is the right decomposition

The one-paper version is too large because it would need to prove all of these at once:

1. define what a reasoning step is;
2. show that current step heuristics are inadequate;
3. introduce solution objects;
4. learn mappings from latent trajectories to object edits;
5. show correctness/faithfulness/optimization implications;
6. maybe intervene causally.

That is too much. Reviewers would attack whichever component is weakest.

The two-paper split lets Paper 1 win on ontology and measurement:

> “What is a step, and how can we detect one?”

Then Paper 2 wins on mechanistic alignment:

> “Do these steps correspond to construction of solution objects?”

## I would avoid calling Paper 1 “theory” too strongly

Use “framework” or “formalism.” In ML, a theory paper without theorems or hard empirical consequences can look speculative. You can include formal definitions and propositions, but the paper should have an empirical core.

Good Paper 1 theorem-style results could be modest:

**Existence:** for domains with an explicit state-transition system, a solution-object step decomposition exists whenever the final answer is generated by a finite valid action sequence.

**Non-identifiability:** textual segmentation alone cannot identify solution-object steps, because multiple tokenizations/sentence boundaries can implement the same object update.

**Refinement:** prompted `Step k` segmentation is a coarse partition; latent/object segmentation may refine or merge prompted steps.

**Minimality:** a solution-object step is minimal only relative to an object representation and update vocabulary.

That last point matters. There is no universal atomic step. “Step” is representation-relative. In algebra, “substitute (x=3)” is atomic under one object vocabulary, but decomposable under another.

## The actual thesis across both papers

I would state the program as:

> Existing CoT analysis treats reasoning steps as text spans and then studies their latent representations. We instead define reasoning steps by their role in constructing a solution object. This separates surface segmentation from computational function: a step is a coherent update to a partial solution state, whether or not the model marks it as a step in text.

That is a good research program. It is not just “milking.” It is a natural staged decomposition: first define and validate the unit of analysis, then use that unit to study the full latent construction process.

[1]: https://arxiv.org/html/2606.05464v1 "Step-by-Step Optimization-like Reasoning in LLMs over Expanding Search Spaces"
[2]: https://arxiv.org/abs/2604.05655 "[2604.05655] LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals"
[3]: https://arxiv.org/abs/2509.00190 "[2509.00190] Explainable Chain-of-Thought Reasoning: An Empirical Analysis on State-Aware Reasoning Dynamics"

Run this as a sequence of increasingly hard hypotheses. Do **not** start with the full “CoT maps onto latent solution-object construction” claim. First prove that your unit of analysis is better than current step heuristics.

Your current codebase is already set up for the first layer: it stores step mean vectors, direction vectors, nudge vectors, variance, clustering artifacts, and probe-ready examples, and you prepared BigCodeBench-Hard as the harder coding direction rather than relying on saturated GSM/MATH runs.

## Paper 1: “What is a reasoning step?”

### H1 — Prompted/sentence steps are weak approximations of real reasoning steps

**Claim:** Step boundaries imposed by `Step 1`, newlines, or sentence splitting do not reliably align with coherent latent transitions.

**Experiment:** Generate CoTs under four conditions:

1. forced numbered steps;
2. freeform CoT;
3. sentence-separated CoT;
4. paragraph-separated CoT.

For each output, compute candidate boundaries using:

[
\cos(\Delta h_t,\Delta h_{t+1}),\quad
|\Delta h_t|,\quad
\text{entropy}_t,\quad
\text{logprob}_t,\quad
\text{curvature}_t
]

Then compare boundary types.

**Metrics:** boundary agreement, latent jump magnitude at boundaries, within-segment coherence, between-segment separation.

**Null:** textual boundaries are as good as latent change-point boundaries.

**Positive result:** latent change-point boundaries produce segments with higher internal directional coherence and sharper between-step separation than sentence/newline/numbered boundaries.

This directly attacks the weakness of existing work: trajectory papers study structured movement through representation space, but mostly assume step units instead of deriving them. The 2026 trajectory paper, for example, treats CoT as structured representation-space trajectories with step-specific subspaces and correctness signals, but it does not fully solve endogenous step discovery. ([arXiv][1])

### H2 — A step is better defined by update function than by textual form

**Claim:** A reasoning step is a contiguous interval that performs one coherent update to a partial solution state.

Create a small manually labeled dataset of 500–1,000 segments. Labels should be functional, not grammatical:

- introduce variable;
- add constraint;
- derive intermediate quantity;
- substitute/simplify;
- case split;
- verify/check;
- backtrack/revise;
- extract final answer;
- for code: identify requirement, choose algorithm, add branch, handle edge case, construct return value, test mentally.

Use three segmentations: sentence, latent change-point, and forced numbered step.

Train simple probes:

[
x*i = [\mu_i,\ h*{\text{end}}-h*{\text{start}},\ \mu_i-\mu*{i-1},\ \mathrm{Var}(h_i)]
]

to predict the functional label.

**Metrics:** macro-F1, NMI/ARI for unsupervised clusters, label purity, cross-problem generalization.

**Null:** labels are mostly lexical; bag-of-words or sentence embeddings do as well as hidden-state transition features.

**Positive result:** latent transition features classify update type better than lexical/text baselines, and direction/nudge features add predictive power beyond mean pooling.

This distinguishes your contribution from the Gram-matrix/state-aware paper. That paper clusters spectral embeddings of already segmented reasoning steps into latent states and Markov transitions; your stronger claim is that transition features identify _functional solution updates_, not merely latent state categories. ([arXiv][2])

### H3 — Object-update segmentation improves correctness prediction

**Claim:** If your step definition is real, it should help predict whether a trace is going wrong.

For each partial trace prefix, train predictors of final correctness using:

1. token-level hidden states;
2. sentence mean states;
3. step mean only;
4. step mean + variance;
5. step direction/nudge;
6. latent/object-update segmentation features.

Use mixed-success datasets. Avoid GSM8K/MATH500 if Qwen3-14B saturates. Use BigCodeBench-Hard or LiveCodeBench-style code tasks because they have richer failure modes. BigCodeBench is designed around diverse function calls and complex instructions, and LiveCodeBench continuously collects coding problems from contest platforms to reduce contamination. ([arXiv][3])

**Metrics:** ROC-AUC, calibration, early-warning AUC at 25/50/75% of trace, cross-task transfer.

**Null:** segmentation choice does not matter.

**Positive result:** object-update or latent-change-point segments predict final correctness earlier/better than sentence/newline baselines.

This is probably the cleanest Paper 1 empirical win.

### H4 — The same update types recur across problems

**Claim:** Reasoning steps form reusable move types.

Cluster step features across many problems, then test whether clusters have stable functional meanings.

**Protocol:**

- cluster on train problems only;
- label cluster exemplars;
- evaluate label consistency on held-out problems;
- repeat across model seeds, prompts, datasets, and layers.

**Metrics:** cluster stability across bootstrap samples, adjusted mutual information, exemplar coherence, transfer of cluster-to-label mapping.

**Null:** clusters are problem-specific or lexical.

**Positive result:** clusters correspond to reusable update functions such as “derive quantity,” “verify,” “case split,” or “handle edge case.”

Your current step-classification artifact already supports this direction: it stores dense research vectors and lightweight interactive JSON, with mean/direction/nudge/variance as probe features.

## Paper 2: “Does CoT map onto solution-object construction?”

This paper needs a real target object. Do coding first; it gives more objective labels.

### H5 — CoT step transitions predict edits to explicit solution objects

**Claim:** Latent step transitions correspond to edits in a solution object.

For coding, define solution objects as:

[
s_i = {\text{AST sketch},\ \text{requirements covered},\ \text{branches/cases},\ \text{tests likely passed}}
]

Build edit labels:

- add import/dependency;
- define helper;
- add loop;
- add conditional;
- handle empty input;
- handle malformed input;
- compute aggregate;
- return final object;
- revise algorithm.

For math, define solution objects as:

[
s_i = {\text{variables},\ \text{constraints},\ \text{equations},\ \text{derived quantities},\ \text{goal residual}}
]

Then align each CoT step to an object edit.

**Models:**

[
g(x_i) \to \Delta s_i
]

where (x_i) is the step transition feature.

**Baselines:**

- text-only sentence embedding;
- final-answer-only;
- token position;
- prompted step number;
- random/shuffled edit alignment.

**Metrics:** edit-type F1, Recall@K for matching step to edit, contrastive accuracy between true and shuffled step-edit pairs.

**Positive result:** hidden transition features predict object edits significantly better than text-only or position baselines.

### H6 — Correct and incorrect traces construct different solution objects

**Claim:** Wrong answers are not just bad final decoding; they often reflect wrong object construction.

For each problem, sample multiple traces. Label or infer object states at each step. Compare correct vs incorrect traces.

Look for failure modes:

- missing required object component;
- spurious constraint;
- wrong variable binding;
- wrong case split;
- premature final-answer extraction;
- algorithm skeleton mismatch;
- edge case omitted.

**Metrics:** divergence step index, edit-distance to reference object, missing-component rate, object-state accuracy over time.

**Positive result:** incorrect traces diverge in object space before final answer generation, and latent trajectory features predict the divergence point.

This is where your “latent solution object” idea starts to become concrete.

### H7 — CoT is a lossy projection of latent solution construction

**Claim:** Hidden states contain object-construction information that is not fully present in the visible CoT text.

Train probes from:

1. visible CoT text only;
2. hidden states only;
3. hidden states + text;
4. final answer only.

Tasks:

- predict which requirements are already satisfied;
- predict next object edit;
- predict final AST skeleton;
- predict which tests will pass;
- predict whether the current partial solution is recoverable.

**Positive result:** hidden states predict object state/edit information beyond visible text.

This would be a strong mechanistic result. It directly supports the claim that CoT is not the whole computation but a projection of an internal construction process.

### H8 — Step transitions form a compressed control sequence for solution construction

I would avoid saying “compression” too early. Say **control sequence** or **low-dimensional update basis**.

**Experiment:** Learn a small latent transition model:

[
z_i = F(z_{i-1}, x_i)
]

where (z_i) predicts the partial solution object embedding.

Compare:

- all token hidden states;
- sentence means;
- step transition sequence;
- only selected “high-impact” steps;
- compressed cluster IDs plus scalar features.

**Metrics:** reconstruction of final object, next-edit prediction, final correctness prediction, number of steps/features needed.

**Positive result:** a short sequence of step transitions preserves most of the object-construction signal.

That would justify the claim:

> CoT induces a sparse, temporally ordered control sequence over latent solution-object construction.

### H9 — Intervention test: changing step direction changes object construction

This is high-risk but high-payoff.

Find a direction associated with an edit type, e.g.:

[
v_{\text{verify}},\quad v_{\text{edge-case}},\quad v_{\text{substitute}},\quad v_{\text{finalize}}
]

At generation time, add/subtract that direction at candidate step boundaries.

Test whether the generated continuation changes object construction:

- more verification;
- more edge-case handling;
- different branch;
- delayed final answer;
- improved test-pass rate.

**Metrics:** edit occurrence rate, correctness, length, semantic drift, output quality.

**Null:** directions are diagnostic only, not causal.

**Positive result:** steering a direction increases the corresponding object edit without destroying generation quality.

Be cautious: this is not needed for Paper 1. It belongs in Paper 2 or a follow-up, because intervention results are easy to overclaim.

## Minimal experiment plan I would actually run first

Do this in order:

**Experiment 1: Mixed-success data collection.**
Use Qwen3-14B or a slightly weaker model on BigCodeBench-Hard / LiveCodeBench. Generate 8–16 traces per problem at temperature 0.6–0.8. Need both correct and incorrect traces.

**Experiment 2: Segmenter comparison.**
Compare sentence, paragraph, prompted step, and latent change-point segmentation on within-step coherence and correctness prediction.

**Experiment 3: Manual step-type labels.**
Label 500 segments. Train probes from mean/direction/nudge/variance. Report whether direction/nudge matter.

**Experiment 4: Object-edit labels for coding.**
For 100 coding traces, label requirement/AST/test-related object edits. Train step-to-edit matching.

**Experiment 5: Divergence analysis.**
For problems with both correct and incorrect traces, identify where object construction diverges. Test whether latent features detect the divergence earlier than text.

That sequence gives you two paper-shaped results:

Paper 1 result:

> Reasoning steps are better modeled as latent transition units than as textual segments; this improves clustering, step-type prediction, and early correctness prediction.

Paper 2 result:

> These latent transition units align with edits to explicit solution objects; correct and incorrect reasoning differ by object-construction trajectories.

## What would falsify the project?

Take these seriously.

If sentence segmentation performs as well as latent/object segmentation on all metrics, your “step” definition is mostly conceptual, not empirically useful.

If text-only embeddings predict step type and object edits as well as hidden states, the latent trajectory angle is weaker.

If clusters are unstable across prompts/models/datasets, then “reusable step types” may be an artifact.

If object-edit prediction works only with LLM-labeled edits and not with objective code artifacts, the result may be circular.

If interventions fail, you can still claim diagnostics, but not causal control.

## My strongest recommendation

Use **coding tasks** for Paper 2. Math is seductive but hard to objectify. Coding gives you ASTs, branches, requirements, unit tests, and execution outcomes. BigCodeBench contains 1,140 software-engineering-oriented tasks involving diverse libraries and domains, while LiveCodeBench is explicitly designed as contamination-free and continuously updated; those are better substrates than saturated arithmetic benchmarks. ([GitHub][4])

The core test should be:

[
\Delta h_i \quad \text{predicts} \quad \Delta s_i
]

better than text, position, and naive segmentation baselines.

If that works, the research program is real.

[4]: https://github.com/bigcode-project/bigcodebench?utm_source=chatgpt.com "[ICLR'25] BigCodeBench: Benchmarking Code Generation ..."
