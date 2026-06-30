[1]: https://arxiv.org/pdf/2605.14619

## Paper 1: “What is a reasoning step?”

<!-- CODEX_NOTES -->
**Execution constraints.** Assume one A6000 and one A100. Prefer exploratory
runs under 12 hours and keep all remote-run artifacts under 20 GB total. Capture
only layer `-1` by default. For mechanistic symbolic-update experiments, use
approximately `L/4`, `L/2`, `3L/4`, and `-1`. Store heavy local artifacts only
when they are necessary.

Order of work: H2 on the balanced existing corpus, H5 locally, the 180-generation
H1 pilot, H4 teacher-forced replay, then H3 intervention.
<!-- /CODEX_NOTES -->

### H1 — Prompted/sentence steps are weak approximations of real reasoning steps

**Claim:** Step boundaries imposed by `Step 1`, newlines, or sentence splitting do not reliably align with coherent latent transitions.

**Experiment:** Generate CoTs under four conditions:

1. forced numbered steps;
2. freeform CoT;
3. sentence-separated CoT;
4. paragraph-separated CoT.

For each output, compute candidate boundaries using:

$$
\cos(\Delta h_t,\Delta h_{t+1}),\quad
|\Delta h_t|,\quad
\text{entropy}_t,\quad
\text{logprob}_t,\quad
\text{curvature}_t
$$

Then compare boundary types.

**Metrics:** boundary agreement, latent jump magnitude at boundaries, within-segment coherence, between-segment separation I guess? Use Silhouette Score to measure how similar a token's hidden state is to its own assigned latent step versus the adjacent latent step. We can also use variance ratio criterion (Calinski-Harabasz index)..

**Null:** textual boundaries are as good as latent change-point boundaries.

**Positive result:** latent change-point boundaries produce segments with higher internal directional coherence and sharper between-step separation than sentence.

This directly the weakness of current trajectory/CoT papers, that mostly assume/force unnatural step units instead of deriving them. ([arXiv][1])

<!-- CODEX_NOTES -->
**Primary objective.** Study naturally occurring boundaries. The current
`<think> Okay, let's see` prefix is natural/freeform for this purpose because it
does not impose structural syntax. Use the existing sentence parser on this
corpus.

Prompt-induced boundaries are a secondary friction baseline: forcing visible
structure may consume latent capacity or place textual boundaries away from
natural latent spikes. Use in-context demonstrations plus these instructions:

- Numbered: `Break your reasoning down into explicit numbered steps. Start each
  step with 'Step X:' on a new line.`
- Sentence-separated: `Express your reasoning with exactly one complete
  sentence per paragraph. Use a double newline after every single sentence.`
- Paragraph-separated: `Group your thoughts into coherent paragraphs of
  multiple sentences. Use a double newline only when moving to a completely new
  phase of the solution.`

First run a matched pilot of 12 questions x 5 seeds x 3 prompted conditions
(180 generations). Scale to 58 x 10 x 3 only if the pilot is informative.
Capture `-1` with diagnostics; use four depth anchors only if layer localization
is needed.
<!-- /CODEX_NOTES -->

### H2: Latent updates are not "smooth", solution-object updates are highly localized within a step

**Claim**: Within, say, a 15-token step, solution-object updates are highly localized. The actual operator $\Delta s_i$ is computed in a sharp, high-magnitude latent spike (high variance, possible directional shift), while the surrounding tokens are merely decoding the state change into linguistic syntax.

The only caveat to anticipate is the interference of attention heads specialized in syntax/grammar. We might need to isolate the specific MLP or attention blocks (typically in the middle-to-late layers) that handle the actual mathematical state updates (see and use [this paper](https://arxiv.org/pdf/2502.20332)), rather than just taking the mean residual stream, which might be noisy with next-token syntax.

Plot the magnitude $||d_t||$ of the token-to-token derivative $d_t = h_t - h_{t-1}$. We can the cluster only the high-magnitude spikes. If the theory is correct, these spikes should cluster cleanly into distinct operator types (e.g. binding, calculating, verifying), regardless of the words around them.

<!-- CODEX_NOTES -->
A **solution-object update** is the minimal token interval
$I=[t_{\mathrm{start}},t_{\mathrm{end}}]$ in which a new valid relation, value
assignment, or terminal answer becomes textually complete and symbolically
verifiable, changing the deterministic GSM-Symbolic execution graph from
$G_k$ to $G_{k+1}$.

Mechanistic claim: the residual stream is a continuous bus, while MLP outputs
act as discrete write operators. Measure the norm of the gated MLP contribution
after projection back into the residual stream,
$\lVert W_{\mathrm{down}}(\sigma(W_{\mathrm{gate}}h)\odot
W_{\mathrm{up}}h)\rVert_2$. If spikes cluster, map graph changes to:

- `BIND`: introduce a variable or constant node.
- `OPERATE`: connect existing nodes with an arithmetic edge.
- `VERIFY`: re-evaluate an expression or check a constraint.
- `EXTRACT`: map an internal node to the terminal answer.

Start with zero-GPU spike localization on the balanced existing corpus. If the
signal is not distinctly non-uniform, revise the step definition before new
generation. Otherwise, capture MLP outputs for 100-200 representative traces at
the four depth anchors. Avoid human or LLM-judge labels.
<!-- /CODEX_NOTES -->

### H3 — Causal Verification of Process Isomers

**Claim:** if two distinct latent trajectories, say $\Delta h_A$ and $\Delta h_B$, are truly process isomers constructing the same solution object, then their resulting internal states should be functionnaly interchangeable.

**Experiment**: Activation Patching:

- Find two structurally different `gsm_symb` traces that reach the same intermediate mathematical state (e.g. trace A uses substitution, trace B uses elimination, but both isolate variable x). Extract the latent state $h_{A, post-update}$ and inject it/patch it into Trace B at the equivalent step, overwriting $h_{B, post-update}$.
- if the solution object is invariant enough to the path taken, Trace B should still successfuly complete the problem or at least come to the same outcome as A. If patching causes catastrophic collapse, then the latent state remains too hard-coupled to the textual/path dependency.

**Quirks**: we must account for positional misalignment. we may need to patch specific attention head outputs rather than the entire residual stream, or use a technique like path patching (as seen in causal tracing) to isolate the value of the isolated variable rather than the entire latent state.

<!-- CODEX_NOTES -->
States in traces A and B are equivalent only when their symbolic execution
graphs are isomorphic at the relevant token: identical bindings and evaluated
sub-results, allowing variable renaming, despite different histories.

Do not patch the full residual stream. Patch the attention output projection or
MLP output at the layer immediately preceding the H2 spike. Stop trace B just
after the equivalent symbolic step, inject A's activation, and preserve B's KV
cache and positional history.

Begin with 20-30 matched pairs and five continuations for each of: baseline,
equivalent-state patch, position-matched random patch, and mismatched-state
patch (roughly 400-600 continuations). Attempt this only after H2 identifies a
stable component and layer.
<!-- /CODEX_NOTES -->

### H4 - unsupervised structural discovery

Instead of building an arbitary smooth top-down supervised dataset via Deepseek, use contrastive learning over `gsm_symb` to supervise the latent operations..

**Experiment**:

- Use the existing thousand of CoT traces for `gsm_symb` in all of the already generated runs of SmolLM3-3B.
- Identify traces that feature the same mathematical operation applied to different numbers, and traces with different operations applied to the same numbers.
- Use a contrastive objective a la SimCLR, strictly on the latent transition vectors ($\Delta h_i$).
- Do the ($\Delta h_i$) vectors of structurally identical mathematical updates naturally cluster together in the manifold, or does any other meaningful structure emerge? If we can extract a clean transition matrix WITHOUT textual JSON labels, we'll have mathematically proven that the model possesses an implicit, non-linguistic solution object schema.

<!-- CODEX_NOTES -->
Mine pairs from GSM-Symbolic graph changes with strict lexical controls:

- Positive: same symbolic operator type across different problems, with no
  overlap in literal tokens, variable names, or numerical values.
- Negative: high textual or numerical overlap, preferably within one problem,
  but different operator types such as `BIND` versus `OPERATE`.

The target is a learned contrastive representation, not only natural
clustering. Success means structurally equivalent, lexically distinct updates
align while lexical controls do not. This is evidence for a non-linguistic
schema, but requires question- and surface-form-disjoint evaluation.

Start locally with existing captured vectors. For scale, add teacher-forced
capture over 1,500-3,000 existing generations instead of regenerating them,
subject to the 20 GB remote-artifact limit.
<!-- /CODEX_NOTES -->

### H5 - Latent solution object steps are better predictors of correctness than standard fragmentation

For each partial trace prefix, train predictors of final correctness using:

1. token-level hidden states;
2. sentence mean states;
3. step mean only;
4. step mean + variance;
5. step direction/nudge;
6. latent/object-update segmentation features.

**Metrics:** ROC-AUC, calibration, early-warning AUC at 25/50/75% of trace, cross-task transfer.

**Null:** segmentation choice does not matter.

**Positive result:** object-update or latent-change-point segments predict final correctness earlier/better than sentence/newline baselines.

<!-- CODEX_NOTES -->
Prediction unit: a prefix checkpoint containing step representations through
25%, 50%, or 75% of the trace. Use grouped, question-disjoint splits.

Keep probes deliberately simple: L2 logistic regression or ridge classification
as the primary model, with a shallow random forest as a secondary nonlinear
check. Compare identical probes over token states, sentence means, step means,
mean plus variance, direction/nudge, and latent-change-point segments.

Run the first study locally on current artifacts. Confirm on a held-out
GSM-Symbolic slice. Only after internal validation, test cross-task transfer on
LiveCodeBench, retaining categorical execution failures such as syntax error,
assertion failure, and timeout.
<!-- /CODEX_NOTES -->

<!-- ## Paper 2: “Does CoT map onto solution-object construction?”

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
-->
