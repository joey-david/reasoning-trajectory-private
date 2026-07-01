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

### NEW: H3 UPDATE

# H3 Protocol: Process-Isomer Causal Patching (Full-Vector + Subspace-Restricted)

## Rationale (for context, not to re-litigate)

H1 (null silhouette) and H4 (raw cosine AUC 0.350-0.394, below chance) both show
that solution-object structure is not geometrically salient in raw hidden-state
coordinates. It only became decodable in H4 after passing through a supervised
contrastive projection. This means a naive full-vector activation patch is
likely to fail for an uninformative reason: it imports trace A's surface/
positional/stylistic content along with whatever task-relevant content exists,
and we cannot tell collapse-from-noise apart from collapse-from-no-object.

We therefore run H3 as **two parallel patch variants** on the same matched
pairs, not one:

1. **Full-vector patch** — the originally specified H3 design. Serves as the
   baseline/reference result and as a check on whether raw-space noise alone
   is sufficient to cause collapse.
2. **Subspace-restricted patch** — replace only the components of B's
   activation that lie along directions identified by H4's learned projection
   as operator-relevant; leave the orthogonal complement as B's own. Tests
   whether task-relevant content alone suffices to carry the object forward.

Interpretation key:

- Both fail -> no portable object at this site; construction is genuinely
  path-dependent.
- Full-vector fails, subspace succeeds -> object exists but is buried under
  nuisance directions (the MoE-syntax-routing analogue).
- Both succeed -> path-independence is robust even to raw-space noise.
- Full-vector succeeds, subspace fails -> treat as an implementation bug and
  re-check projection fidelity before drawing conclusions.

Report both variants side by side. Do not present the subspace variant as a
stronger result than it is: success there is partially underwritten by H4's
projection being right about what's relevant, so it's a claim about
decodability within our existing operator ontology, not proof of an
unconditioned object.

## Pair selection

- Source: existing `gsm_symb_pure_mixed_latents_10k` corpus (and 1,500-trace
  replay corpus if pair yield is too low locally).
- A valid pair (trace A, trace B) requires: same question or same underlying
  symbolic state, structurally different derivation paths (e.g., substitution
  vs. elimination), reaching an **isomorphic** symbolic execution graph state
  at the patch point — identical bindings/evaluated sub-results, variable
  renaming allowed.
- Reject pairs where the only difference is superficial phrasing of the same
  derivation; we want path-distinct constructions of the same object, not
  paraphrases.
- Target: 20-30 matched pairs. Log actual yield and rejection reasons
  (insufficient path diversity, no isomorphic state found, etc.) — if yield is
  far below 20, flag this back before proceeding rather than silently
  loosening the isomorphism criterion.

## Alignment

- Never align to an H2 magnitude/path spike. Always derive completion indices
  from symbolic `token_end`.
- Stop trace B's generation/replay exactly after its own $t_{end}$.
- Source activation: trace A's component output at $t_{end} + 1$ (completed-
  state index, one token past the symbolic update), not at $t_{end}$ itself.
- Preserve trace B's preceding KV cache and full positional history; only the
  single component output at the aligned index is overwritten.

## Components and layers

- Primary target: **attention output projection, layer 18**.
- Prespecified sensitivity target: **MLP output, layer 18** — use this if and
  only if the attention-18 intervention is negative (collapse or no
  improvement over random-position control) on the full-vector variant. Do
  not run MLP-18 preemptively in parallel with attention-18 on the first
  pass; treat it strictly as the fallback the plan already specifies.
- Run both variants (full-vector, subspace-restricted) at whichever
  layer/component is active for that stage.

## Subspace-restricted patch — implementation detail

1. Load the trained H4 contrastive projection (`h4_structural_replay`
   artifacts — the linear projection that took raw cosine AUC 0.350-0.394 to
   0.945-0.957 question-disjoint).
2. Project both $h_{A,t_{end}+1}$ and $h_{B,t_{end}+1}$ into the projection's
   subspace.
3. In that subspace, replace B's projected coordinates with A's projected
   coordinates.
4. Reconstruct: take B's original activation, subtract B's component along
   the projection's basis directions, add back A's component along those same
   directions. Leave B's activation unchanged in the orthogonal complement.
5. Use this reconstructed vector as the patch, in place of A's raw activation,
   for the subspace-restricted condition only. The full-vector condition still
   uses A's raw, unprojected activation.

If the projection is not full-rank/orthogonal (e.g., it's a learned linear map
to a lower-dimensional space rather than an orthogonal projection in the
original basis), use the pseudoinverse to map back, and log the reconstruction
residual (||B_reconstructed_in_subspace_only - B_original|| restricted to the
subspace) so we can verify the swap is actually happening as intended before
trusting downstream results.

## Conditions (run for each of the two patch variants)

1. **Baseline** — no patch, B continues normally.
2. **Equivalent-state patch** — A's (sub)activation at $t_{end}+1$, as defined
   by the active variant, patched into B.
3. **Position-matched random patch** — same position, same variant
   (full-vector or subspace-restricted as appropriate), source activation
   drawn from an unrelated, non-equivalent trace. Controls for "any patch at
   this position disrupts generation."
4. **Mismatched-state patch** — a different symbolic state's completed
   activation, same variant. Controls for "any plausible solution-object
   vector works regardless of which state it encodes" — tests specificity.

This gives 2 variants x 4 conditions = 8 cells per pair.

## Scale

- 20-30 matched pairs x 5 continuations per cell x 8 cells ≈ 800-1,200 total
  continuations (roughly double the original single-variant estimate, since
  we're running both patch variants on the same pairs).
- If compute-constrained, prioritize completing all 8 cells on a smaller pair
  set (e.g., 15 pairs) over running fewer cells on more pairs — the
  within-pair full-vector-vs-subspace contrast is the primary comparison of
  interest.

## Metrics — log separately, do not collapse into one score

For every continuation:

- Collapse / non-collapse (degenerate or incoherent output).
- Valid-answer rate (produces _a_ well-formed final answer, regardless of
  correctness).
- Correctness (matches expected answer under the patched/implied symbolic
  state — for equivalent-state, this is A's answer; for mismatched-state,
  there is no "correct" answer to target, log whether it follows the patched
  state's implied answer or B's original unpatched trajectory's answer or
  neither).
- Reconstruction residual (subspace variant only, per above).

Aggregate per cell (mean + question-grouped bootstrap CI, consistent with H2's
reporting convention) before any cross-cell comparison.

## Decision rule

- Full-vector equivalent-state vs. full-vector random/mismatched: is there a
  meaningfully higher collapse/correctness gap specific to equivalent-state?
- Subspace equivalent-state vs. subspace random/mismatched: same question,
  restricted variant.
- Full-vector equivalent-state vs. subspace equivalent-state: this is the
  primary new contrast — does restricting to the H4-relevant subspace rescue
  a result that the full-vector patch destroys?

If attention-18 is negative on the full-vector variant for both equivalent-
state and subspace-restricted conditions, proceed to MLP-18 fallback before
concluding path-coupling. If both attention-18 and MLP-18 fail on the
subspace-restricted variant specifically, treat that as the strongest
available evidence against a portable solution object at this depth, and
note in the writeup that this does not rule out the object existing at a
different layer (per the earlier flag: peak write-activity layer is not
guaranteed to be the layer where the object stabilizes into a portable form).

## Prepared implementation

- The pair miner found 24 strict pairs spanning 12 questions and 32 source
  trajectories. Every pair has an exact completed graph-state match, at least
  two graph-changing symbolic-history edits, and normalized path distance at
  least 0.286. Every target source trace answered correctly and has at most 891
  tokens after the patch point, leaving at least 133 tokens of headroom under
  the 1,024-token continuation cap. Rejections and selection criteria are
  recorded in `experiments/h3_process_isomer_pair_audit.json`.
- H4 projections were retrained in the actual patch spaces rather than reusing
  the final-residual projection. Question-disjoint projected AUC is 0.958 for
  attention output at layer 18 and 0.972 for MLP output at layer 18; both
  128-dimensional maps have full row rank.
- The subspace swap uses the Moore-Penrose pseudoinverse. On locally available
  matched activations, maximum coordinate-reconstruction residual was
  `1.96e-6` for attention and `1.42e-6` for MLP; mean retained update norm was
  0.135 and 0.153 respectively.
- `h3_process_isomer_replay` captures only the 32 required trajectories and
  both layer-18 components. Estimated raw activation storage is 0.26 GiB.
  The primary attention run contains 24 pairs x 8 cells x 5 continuations =
  960 continuations. The MLP run is prepared but remains a gated fallback.
- Preflight validates pair evidence, activation indices, projection metadata,
  numerical reconstruction, and all eight cells before inference. Analysis
  reports each cell separately with question-grouped bootstrap intervals.
- Local SmolLM3-3B MLX inference is suitable for text-generation smoke tests
  and pilots. It does not validate component hooks, KV-cache alignment, or
  activation replacement, so H3 itself remains an HF/CUDA remote run.
  Exact-token-prefix pilots on both smoke pairs produced valid answers in 7
  and 180 continuation tokens.

## Implementation pointers

- `scripts/experiments/run_h3_protocol.sh primary` runs targeted replay,
  strict preflight, the two-pair smoke gate, resumable inference, and analysis.
- `causal_patching.py` exposes `--patch-mode full|subspace|both`; each run
  config pins the matching component-space H4 projection.
- Derive all donor/target indices from symbolic `token_end`, per existing
  convention; there is no separate spike-based indexing path.
- Use the prespecified component/layer directly (attention-18 primary, MLP-18
  fallback).

## Attention-18 result

- All 960 continuations completed with no degenerate outputs and a valid
  extracted answer in every row.
- Equivalent-state target accuracy did not separate reliably from controls.
  Full-vector differences were `+6.9` points versus position-random (95% CI
  `-3.6` to `+16.9`) and `+3.3` versus mismatched (`-5.0` to `+12.5`).
  Subspace differences were `+5.8` (`-6.7` to `+18.9`) and `+2.8`
  (`-7.2` to `+13.9`). Subspace versus full equivalent differed by only
  `+2.2` points (`-3.3` to `+7.8`).
- Equivalent donor and target gold answers are identical by construction, so
  donor-answer matching does not demonstrate state transfer. Mismatched patches
  often matched or beat equivalent patches, and gains concentrated in pairs
  with weak random controls rather than increasing with symbolic path distance.
- Duplicated no-patch baselines had identical token sequences in only 66/120
  cases and identical correctness in 105/120. The mixed A100/A40 run did not
  record worker provenance, leaving hardware-sensitive sampling noise
  unquantified.
- About half the rows reached the 1,024-token cap after emitting an answer
  because H3 omitted regex stopping. The reported first-answer endpoint is what
  normal stopping would retain; a last-answer sensitivity check removes the
  subspace effect. Answer-based stopping is restored for subsequent runs.

**Conclusion.** Attention output at layer 18 is a null result for portable,
state-specific solution objects and triggers the prespecified MLP-18 fallback.
Run that fallback on GPUs of one model type and require equivalent patches to
separate from both random and mismatched controls.

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
natural latent transition phases. Use in-context demonstrations plus these
instructions:

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

Pilot status: numbered and sentence-separated conditions completed 60 matched
traces each; the paragraph condition was interrupted after 21. Preliminary
behavioral checks show weak literal format compliance. Sentence separation
reduced matched trace length to about 44% of freeform, while its accuracy change
was small and uncertain. Thus the prompt changes generation dynamics even when
the model does not obey the requested delimiter syntax. Latent comparison
finds no clean discrete boundary family: magnitude spikes recover only 3-5% of
symbolic updates, while sentence boundaries recover 65-82% but with only
14-21% precision. Segment Silhouette scores remain near zero or negative.
Sentence prompting improves textual coverage of symbolic updates without
creating more coherent latent clusters. Sentences are therefore coarse
high-recall envelopes, not faithful atomic reasoning steps.

Matched interval dynamics do not support the secondary architectural-friction
claim. None of the prompted conditions detectably increases path length or
temporal width. Completion-state net displacement does increase relative to
freeform for numbered (+0.034), sentence-separated (+0.083), and the partial
paragraph condition (+0.066); sentence-separated net-to-path ratio also rises
by 0.024. Together with its much shorter traces, the current evidence is more
consistent with prompt-induced compression or regularization than extra latent
work spent forcing textual boundaries.

<!-- /CODEX_NOTES -->

### H2: Solution-object updates form localized transition phases, not point spikes

**Claim:** A solution-object update is localized to a short token interval, but
its latent change is distributed across that interval as a wave-like transition
band. It is not generally concentrated in one high-magnitude token. "Wave-like"
describes broad temporal support, not a smooth or directionally coherent path:
the surrounding tokens participate in constructing and stabilizing the update
rather than merely decoding an instantaneous state change into syntax.

**Experiment:** For each symbolically verified update interval
$I=[t_{\mathrm{start}},t_{\mathrm{end}}]$, measure:

$$
\sum_{t\in I}\lVert h_t-h_{t-1}\rVert_2,\qquad
\lVert h_{t_{\mathrm{end}}+1}-h_{t_{\mathrm{start}}}\rVert_2,
\qquad
\sum_{t\in I}\left(1-\cos(h_t,h_{t-1})\right).
$$

Also measure peak share, effective temporal width, temporal centroid, and
net-to-path ratio. Compare every interval with same-length non-update windows.
Run the same analysis on residual, MLP-output, and attention-output paths at
the selected depth anchors.

**Positive result:** Update intervals have elevated integrated path length or
net displacement, while their peak share is low and effective width spans
multiple tokens. Operator types should be tested using the interval's net
transition vector, not a single endpoint derivative.

<!-- CODEX_NOTES -->

A **solution-object update** is the minimal token interval
$I=[t_{\mathrm{start}},t_{\mathrm{end}}]$ in which a new valid relation, value
assignment, or terminal answer becomes textually complete and symbolically
verifiable, changing the deterministic GSM-Symbolic execution graph from
$G_k$ to $G_{k+1}$.

Mechanistic claim: the residual stream is a continuous bus, while MLP outputs
contribute a sequence of partial writes over the update interval. Integrate the
gated MLP contribution after projection back into the residual stream,
$\lVert W_{\mathrm{down}}(\sigma(W_{\mathrm{gate}}h)\odot
W_{\mathrm{up}}h)\rVert_2$, and measure the path traced by successive outputs.
Map graph changes to:

- `BIND`: introduce a variable or constant node.
- `OPERATE`: connect existing nodes with an arithmetic edge.
- `VERIFY`: re-evaluate an expression or check a constraint.
- `EXTRACT`: map an internal node to the terminal answer.

The initial last-layer result rejects the strong point-spike version: update
endpoints have elevated derivative magnitude, but sharp-spike overlap is only
slightly above shifted controls. Treat this as evidence for a distributed
transition phase and test interval-integrated metrics on the component replay.
Avoid human or LLM-judge labels.

On 17,602 updates from 580 traces, interval path length ranks at approximately
the 73rd percentile of matched non-update windows by question
(95% bootstrap interval: 71st-75th). Mean peak share is 0.19 and effective
width covers 0.98 of the interval. Net displacement is only near the 52nd
percentile and the net-to-path ratio is 0.17. The supported claim is therefore
distributed, meandering update activity; calling it a coherent directional
wave would currently be too strong.

The operator split is not uniform. Question-balanced path length ranks at the
81st matched-window percentile for `VERIFY`, 72nd for `OPERATE`, 65th for
`BIND`, and only 26th for `EXTRACT`. This suggests that the terminal answer is
usually a low-change readout of an object constructed during earlier
computation and checking, rather than the point where that object is formed.

The 174-trace component replay localizes the strongest distributed transition
to layer 18. Attention output has a question-balanced combined score of 0.739
(path 0.829, net displacement 0.631), while MLP output is close at 0.732
(path 0.842, net displacement 0.583). Their peak shares are 0.22 and 0.19,
effective widths are 0.93 and 0.98, and temporal centroids are 0.53 and 0.51.
The update is therefore centered across the interval rather than end-loaded.
Both components participate; the evidence does not support an MLP-only
discrete-write account.

The larger replay replicates the residual result on 27,035 updates from 1,500
traces and 300 questions: question-balanced path percentile is 0.700
(95% CI: 0.690-0.709), peak share is 0.192, effective width is 0.979, and
net-to-path ratio is 0.170.

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

Do not align a patch to an H2 spike. That would inject an incomplete partial
update and make a failure uninterpretable. Stop trace B exactly after
$t_{\mathrm{end}}$, the final token of the equivalent symbolic step. Inject
Trace A's component activation from its corresponding completed-state index
$t_{\mathrm{end}}+1$, while preserving Trace B's preceding KV cache and
positional history.

Patch only the attention output projection or MLP output at the component and
layer with the strongest interval-integrated H2 signal. The implementation must
derive completion indices from `token_end`; it must not trust a previously
stored spike or midpoint index.

The primary automatic target is attention output at layer 18. Because MLP
output at layer 18 is nearly tied and has the stronger path signal, treat MLP-18
as the prespecified sensitivity target if the attention-only intervention is
negative.

Begin with 20-30 matched pairs and five continuations for each of: baseline,
equivalent-state patch, position-matched random patch, and mismatched-state
patch (roughly 400-600 continuations). Attempt this only after H2 identifies a
stable component and layer.

A negative H3 result is not evidence against process-isomer equivalence unless
the completed-state alignment and component choice pass their controls. Even
with end alignment, one terminal MLP/attention output may expose only the final
piece of a distributed update rather than the accumulated solution object.
Treat H3 as exploratory and report collapse/valid-answer rates separately from
correctness.

<!-- /CODEX_NOTES -->

### H4 - unsupervised structural discovery

news: apparently already found in <https://arxiv.org/html/2605.13772>.

Instead of building an arbitary smooth top-down supervised dataset via Deepseek, use contrastive learning over `gsm_symb` to supervise the latent operations..

**Experiment**:

- Use the existing thousand of CoT traces for `gsm_symb` in all of the already generated runs of SmolLM3-3B.
- Identify traces that feature the same mathematical operation applied to different numbers, and traces with different operations applied to the same numbers.
- Use a contrastive objective a la SimCLR on interval transition vectors
  $\Delta h_I=h_{t_{\mathrm{end}}+1}-h_{t_{\mathrm{start}}}$.
- Do the interval vectors of structurally identical mathematical updates
  cluster together despite lexical controls, or does another meaningful
  structure emerge?

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

Start locally with interval net-displacement vectors from existing captures.
For scale, add teacher-forced capture over 1,500-3,000 existing generations
instead of regenerating them, subject to the 20 GB remote-artifact limit.

Local result: replacing the endpoint delta with full-interval net displacement
gives projected question-disjoint controlled-pair AUC 0.945 on the initial
corpus (raw cosine AUC: 0.350). The 1,500-trace replay scales this to 10,421
updates over 278 questions and reaches 0.957 projected AUC, while raw cosine AUC
remains only 0.394. Structural operator information is robustly recoverable by
a symbolically supervised linear projection and does not depend on a point
spike. The low raw AUC rejects the stronger natural-clustering or unsupervised
discovery claim; this is decodability evidence, not proof of an autonomous
non-linguistic schema.

<!-- /CODEX_NOTES -->

### H5 - Latent solution object steps are better predictors of correctness than standard fragmentation

For each partial trace prefix, train predictors of final correctness using:

1. token-level hidden states;
2. sentence mean states;
3. step mean only;
4. step mean + variance;
5. step direction/nudge;
6. latent transition-phase and object-update interval features.

**Metrics:** ROC-AUC, calibration, early-warning AUC at 25/50/75% of trace, cross-task transfer.

**Null:** segmentation choice does not matter.

**Positive result:** object-update or latent transition-phase segments predict
final correctness earlier/better than sentence/newline baselines.

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

Current corrected result: sentence-segment mean plus variance reaches AUC
0.757, 0.740, and 0.762 at 25%, 50%, and 75%. Symbolic interval mean plus
variance reaches only 0.631, 0.652, and 0.700; sustained-change bands reach
0.639, 0.616, and 0.680. All deficits against the equal-dimensional sentence
baseline have question-grouped bootstrap intervals below zero. An earlier
`step_mean_variance` label referred to sentence aggregates and was misleading.
H5 is rejected on the current corpus: latent or symbolic transition
segmentation does not improve correctness prediction over sentence-level
variation.

<!-- /CODEX_NOTES -->
