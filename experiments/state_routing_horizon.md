# State routing and long-reasoning consistency

## Question and claim

Why does a transformer lose consistency in long reasoning when the facts and
intermediate values it needs still occur in its context and activations?

The working claim is:

> Long-run consistency depends on how each read gathers prior results. A
> value-weighted routing margin predicts whether a model will reuse a correct
> written value beyond the lengths used to fit it. The same sparse heads
> causally select state at final and intermediate reads; redirecting them can
> repair the read without changing the values stored in the trace.

This splits failures into four testable types: the value is no longer carried
by the write (**storage**); a new write is not made usable (**update**); the
read selects a stale or irrelevant write (**retrieval**); or the read selects
the right value for the wrong variable, goal, or operation (**binding**).
Circuit discovery is a measurement tool, not the claim. We do not assume that
the model maintains an incremental state. Tang et al. find parallel read-time
aggregation rather than incremental tracking on entity updates; the first test
decides whether arithmetic reasoning reuses its own writes or re-reads the
problem.

## Six tests

1. **Storage over time.** Decode value, variable, and operation separately from
   value vectors at writes, the read query before attention, head outputs, and
   the post-attention residual. Train on short histories; test on longer ones.
   Decoding a literal value token alone does not count as state storage.
2. **Routing over time.** At each marked read, measure every head's attention to
   the latest valid write, stale writes of that variable, writes of other live
   variables, and matched irrelevant tokens. Record the log attention ratio
   between the valid write and its strongest competitor.
3. **Update closure.** After each overwrite, test whether the same functional
   head set switches from the old write to the new write and remains switched
   through later work. Separate a missing usable write from a failed switch.
4. **Capacity.** Cross the number of live variables, overwrites, delay tokens,
   and value collisions. Fit the routing rule on short, low-load histories and
   predict the first failed read and full-trajectory success out of range.
5. **Semantic stability.** Hold the state and dependency graph fixed while
   changing names, prose, event order where causally valid, and arithmetic
   form. Compare functional read-write edges and causal effects, not raw map
   identity.
6. **Causal repair.** On failures where the needed value remains available,
   patch only query/key routing signals from a successful structural twin whose
   numeric answer differs. Test recovery of the next value and final answer.
   Match against value, residual, stale-write, wrong-variable, random-head,
   layer, norm, and intervention-size controls.

## Test 0: locate the causal source

Before head selection, compare three possible sources for each later read: the
original problem mentions, earlier computed results in the ordinary solution,
and the read token where information is gathered. Use clean/corrupt matched
pairs and residual patching at each source across layers. Score normalized
indirect effects on the next value token. This test may show that there is no
persistent write interface: if problem mentions dominate, all later tests use
read-time aggregation as their mechanism and “write” means a candidate source,
not a mutable state slot.

### Locked pilot result

The 200-case pilot passed on both models. Among clean/corrupt pairs that changed
the preferred answer, restoring the prior computed result had mean normalized
effects of 0.895 on Qwen2.5-7B-Instruct and 0.896 on Mistral-7B-Instruct-v0.3.
Restoring the changed problem mention gave 0.093 and 0.053. The paired gaps were
0.802 (95% interval 0.782--0.822; 197 cases) and 0.844 (0.833--0.854; 195
cases). The next stage therefore treats the model-written result as the causal
source. This does not show that a hidden mutable state exists.

Free-generation accuracy was 0.76 on two-event and 0.78 on ten-event Qwen
cases. Mistral fell from 0.84 to 0.27. The latter supplies long-run failures for
the head test; Qwen tests whether the same measure separates errors without a
mean length collapse.

### Head and prediction result

A 64-case screen found sparse late read heads. The frozen 16-head sets restored
0.689 of Qwen's and 0.922 of Mistral's clean/corrupt answer margin; layer-matched
heads restored -0.056 and -0.075. On the model's own trace, fit on 4--8 events
and tested on 12--20, their value-weighted valid-write margin reached AUC 0.904
(105 reads) and 0.901 (65 reads). Matched heads reached 0.609 and 0.627; length
reached 0.594 and 0.497. Direct answer-token confidence reached about 0.999, so
the pre-set best-baseline advantage gate failed. Gold-trace attention reached
only 0.585 and 0.568 AUC, confirming that a different trace cannot stand in for
the state the model actually wrote.

Cross-context Q/K transfer also failed its controls. Successful-donor Q+K
repaired 0.125 of aligned Qwen failures and 0.308 of Mistral failures, while
preserving only 0.500 and 0.486 of correct reads; a failed donor repaired 0.196
on Qwen. We therefore reject semantically stable donor Q/K as the repair
interface.

The replacement intervention redirects half of each frozen head's output mass
to a source value already present in the same trace. Qwen development selected
this strength; it transferred unchanged to Mistral. On held-out 12--20-event
traces, valid-source steering repaired 45/54 Qwen errors (0.833; exact 95%
interval 0.707--0.921) and 25/25 Mistral errors (1.000; 0.863--1.000).
Layer-matched heads repaired 0/54 and 1/25; the stale target write repaired
0/79. The intervention preserved 45/45 and 35/35 correct reads (pooled exact
lower bound 0.955). All cases had a correct literal target value before the
wrong answer, valid and stale values differed, and reconstructed baseline
logits had to match the saved answer.

This is an oracle-source result: it identifies where the valid write is, but it
does not replace its value, alter any prior token, or change model weights. For
a head output `y = sum_t a_t v_t`, the intervention
`y' = (1-alpha)y + alpha v_s` is exactly the attention distribution
`a'_t = (1-alpha)a_t + alpha 1[t=s]`. It therefore changes routing while
holding the available value vectors fixed. The result supports a causal
stored-but-misrouted failure class at the final read. It does not yet establish
repair of a full free-running trajectory.

### Intermediate-read result

We then froze the same head sets and Qwen-selected strength and moved the
intervention from the answer to repeated-variable operands inside held-out
12--20-event traces. We retained every wrong read backed by a correct earlier
model write and 100 fixed correct reads per model. Valid-source steering
repaired 9/15 Qwen and 12/12 Mistral reads (pooled 21/27, 0.778; exact 95%
interval 0.577--0.914). Layer-matched heads repaired 2/27. The paired read
advantage was 20 versus 1 discordant cases (exact McNemar `p=2.1e-5`).

The repaired operand was then placed in the same generated prefix and the
unmodified model computed the next arithmetic result. Both the read and next
write were correct in 6/15 Qwen and 8/12 Mistral failures (pooled 14/27,
0.519; interval 0.319--0.713), versus 2/27 for matched heads (13 versus 1
discordant cases, `p=0.0018`). Valid steering preserved 199/200 correct reads;
ordinary read-plus-update accuracy changed from 165/200 to 164/200.

For the semantic control, we redirected the causal heads to a correct stored
value of another variable. This repaired 0/27 failures and made the model emit
that wrong variable's exact value in 27/27. Thus the intervention does not just
raise digit probability or disrupt the network: the selected source determines
which stored state the model uses. This establishes causal one-step reuse and
update closure on the controlled task. Full autoregressive closure and natural
tasks remain open.

## Experiment 1: controlled causal map

Create ordinary word problems with several named quantities. Events read one
or two quantities and write a new value; some overwrite an existing quantity.
Filler events preserve the same surface length but do not affect the queried
value. A normal worked solution states each calculation; it does not ask the
model to report, label, or monitor an internal state. A single teacher-forced
forward pass scores every gold step. Free generation measures the model's true
failure point on the same problem.

Use single-token integer answers and retain exact character spans for every
variable mention, operand value, result, operation, and read/write link. Cross:

- live variables: 2, 4, 8;
- writes per variable: 1, 2, 4;
- delay after the last relevant write: 0, 8, 32 filler events;
- values: unique or collision-rich;
- surface form: compact ledger, prose A, prose B;
- operation: add, subtract, multiply, and copy, with bounded integer results.

The development split uses at most four variables, two writes, short delays,
two surface forms, and add/subtract. The locked test split contains longer
histories, eight variables, four writes, long delays, the third surface form,
collisions, multiply/copy, and unseen combinations. Split by program graph and
random seed, not by rendered prompt.

### Stages

1. **Behavioral pilot.** Generate 200 development cases per model under free
   generation. Require valid parsed steps, at least 80% easy accuracy, and
   35--75% hard accuracy. Adjust difficulty only before locking the data.
   Run Test 0 on at least 150 later reads. A CoT-write claim survives only if
   its normalized patch effect exceeds the problem-source effect by 0.15 with a
   95% interval above zero; otherwise reframe the mechanism as read-time
   aggregation.
2. **One-pass screen.** On an independent 256-case screen, collect residual, Q/K/V,
   per-head output, and attention at marked reads. Screen all heads by ablating
   one head only at read tokens. Rank by held-out change in the correct-vs-best-
   wrong logit margin. Freeze the smallest head set within 90% of the full
   selected effect; never select on hard-test outcomes.
3. **Prospective test.** Without refitting, predict each hard-test read error
   in the model's own parsed free-generation trace from a value-norm-weighted
   routing margin, competitor count, and step index. The competitor set includes
   original problem mentions, prior computed results, wrong-variable sources,
   and the rest of the attention support through its retained mass. Teacher-
   forced gold traces provide only secondary measurements.
   Compare with token distance, raw attention entropy, residual norm, question
   length, and short-history accuracy. Report held-out AUC, Brier score, and
   calibration, plus first-error and whole-trajectory prediction.
4. **Failure split.** Probe value, variable, and operation on held-out graph and
   surface splits. Count a stored-but-misrouted failure only when value decoding
   passes its locked test and the valid-write margin fails. The main result is
   the share of long-run errors assigned to each failure type with bootstrap
   intervals over program graphs.
5. **Matched intervention.** For stored-but-misrouted cases, patch Q only, K
   only, Q+K, V only, head output, or residual at the frozen heads/layers. The
   donor matches the graph but has different names and values. Include donors
   with the valid write shifted to a different position and donors that fail on
   another read. A repair must improve both the next dependent step and final answer,
   while not changing matched unrelated reads.

Primary gates on at least 250 held-out reads: routing AUC at least 0.75, Brier
score at most 0.18, and AUC at least 0.07 above the best simple baseline; the
out-of-range AUC drop is at most 0.08. For repair, Q/K must beat each donor
control by 0.10 absolute and preserve unrelated reads with a 95% lower bound of
0.95 over at least 200 reads. Mistral must retain at least 60% of each Qwen
effect. Missed gates narrow or stop the claim; they are not tuned after lock.

## Experiment 2: natural-trace transfer

Freeze the metric, layer ranges, head-selection rule, probes, and all cutoffs
from Experiment 1. Apply them first to existing mixed-success GSM-Symbolic
generations using the repository's arithmetic-update parser. Use only parsed
steps whose operands and result map to exact token spans; audit a random sample
by hand. Then test MATH problems with short executable arithmetic chains.

First audit at least 300 hand-labeled update events. This arm runs only if the
write-graph parser reaches precision 0.95 and recall 0.90. For each correct
prefix followed by a wrong numeric update, ask whether the
frozen routing score predicted the first error before its answer token. Compare
against step confidence, answer-token entropy, length, operation type, and a
probe trained directly on the natural development split. Intervene only on
cases with a matched counterfactual trace and a different donor answer. Natural
data confirm scope; they do not select the circuit or rescue a failed controlled
test.

## Theory target

If the valid source's attention logit exceeds every other token in the full
softmax support by `delta`, with context length `T`, its attention mass is at
least

`1 / (1 + (T - 1) exp(-delta))`.

This exact bound may be loose. Treat any fitted mass-to-correctness link and
trajectory extrapolation as a calibrated predictor, not a theorem. Do not call
it a capacity law unless one fixed predictor works across held-out lengths,
loads, surfaces, operations, and a second model.

## Prior work and novelty boundary

Retrieval heads already causally support long-context recall; Thought Anchors
and Sequential Activation Patching already trace and ablate paths across
reasoning. Most directly, Tang et al. show that models aggregate entity updates
at the last token rather than maintain state incrementally; they recover PUT
binding circuits and REMOVE tags, use the mechanism to predict failures, and
partly repair them. We will run their public MIT code as a named baseline, not
reimplement that claim. Our target is the quantitative step they do not test:
one frozen, value-aware routing score that predicts the first error in a model's
own arithmetic reasoning beyond fitted lengths, plus routing-only repair that
survives position-shifted and failing-donor controls.

Closest work: [Retrieval Heads](https://arxiv.org/abs/2404.15574), [Thought
Anchors](https://arxiv.org/abs/2506.19143), [Sequential Activation
Patching](https://arxiv.org/abs/2608.22332), [(How) Do Language Models Track
State?](https://proceedings.mlr.press/v267/li25r.html), [Entity State
Changes](https://arxiv.org/abs/2605.30233), and [Retrieval-Conditioned
Rebinding](https://arxiv.org/abs/2606.08644).

## Run plan

Use pinned Qwen2.5-7B-Instruct and Mistral-7B-Instruct first so the run stack and
prior artifacts stay shared. A Test 0 pilot uses one H100 and must finish within
one hour. Store only marked-query attention rows and top sources, never dense
maps. Subsequent jobs depend on the pilot gates and use independent two-hour-or-
less shards for the screen, prospective test, and repair. Each stage writes one
row per case and resumes by stable case/stage key. No later stage may edit the
locked dataset, score, or head set.
