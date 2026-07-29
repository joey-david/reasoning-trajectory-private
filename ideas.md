## 1. Are equivalent reasoning states causally interchangeable?

This is the strongest fit for your work.

Take two reasoning trajectories that have reached the same formally defined solution object through different wording, ordering, notation, or intermediate calculations. At the corresponding boundary, transplant the hidden state—or selected layers/KV states—from trajectory A into trajectory B. Then continue generating.

The elementary question is:

> **Once two traces “know the same thing,” can their internal states be substituted?**

Recent work finds that representations of equivalent operations become more similar across different wordings during extended reasoning, but explicitly notes that representational alignment does not establish functional interchangeability. ([arXiv][2])

A minimal experiment:

- Generate symbolic arithmetic or state-tracking problems with exact solution-object annotations.
- Produce several valid derivations reaching identical states.
- Patch one trace into another at each layer.
- Measure continuation correctness, answer-logit recovery and whether subsequent trajectories converge.

All three plausible results are publishable:

- **Interchange works:** reasoning contains modular, reusable semantic states.
- **Similarity is high but interchange fails:** representation is decodable but not causally modular.
- **It works only at particular layers or boundaries:** those locations are genuine state-handoff interfaces.

This is much sharper than “how are solution objects represented?” It asks one binary causal question.

**Novelty confidence: high.**

**Pilot correction.** Patch-switch utility may be governed more by trajectory
alignment than by formal state equality: a state can be abstractly correct yet
occupy an incompatible trace coordinate. Test this directly with a factorial
crossing same/different formal values and aligned/misaligned trace forms. The
key positive control is whether an aligned donor with a different value
transfers its continuation, while a formally equivalent but misaligned donor
does not. Run both a direct read continuation, which should expose basic
transfer if it exists, and an add-modulo continuation, which tests whether the
transplanted value remains usable by a later operation.

## 2. Does the model know which intermediate results will matter later?

Construct paired problems containing the same facts and calculations, but alter which derived quantity is eventually needed. When the model computes a value, compare its internal representation depending on whether that value lies on the future answer path.

The question:

> **Is future utility encoded when a result is first produced?**

For example, suppose a trace computes both:

[
a=17,\qquad b=23.
]

In one problem, only (a) will be reused; in the matched problem, only (b) will. At the token completing each calculation, test whether a probe can decode:

- whether the result will be reused;
- how many future steps depend on it;
- when it will next become relevant.

Then perform a causal test: transplant the “future-relevant” representation onto an otherwise irrelevant result and see whether the model attends to or recalls it later.

AttriCoT estimates how earlier reasoning units causally influence later units, but that is an ex-post dependency analysis rather than asking whether a result is prospectively marked for future use when created. ([arXiv][3])

This could reveal a latent analogue of register allocation, salience tagging or working-memory prioritization.

**Novelty confidence: high.**

## 3. Does correcting an intermediate result erase the original plan?

Let the model begin a solution and commit a controlled error. Then replace the erroneous intermediate result with the correct one while preserving as much of the surrounding trajectory as possible.

Ask:

> **Does the model recompute from the correction, or continue executing the plan induced by the original mistake?**

Call the phenomenon **reasoning hysteresis**.

You could compare four interventions:

1. Text correction only.
2. Hidden-state correction only.
3. Text and hidden-state correction.
4. Restarting from the corrected prefix.

Then measure whether downstream quantities follow the corrected value, the old value, or an incoherent mixture. Existing work injects controlled intermediate errors to generate process-supervision data, and other work studies commitment points, but I did not find this specific correction-versus-plan-inertia decomposition. ([arXiv][4])

The interesting result would be that text says one thing while the latent trajectory continues the previous computational plan.

**Novelty confidence: medium-high.**

## 4. Can a reasoning model carry an unresolved dependency?

Delete a required intermediate calculation and replace it with something like:

> “Let (x) denote the required quantity; I will compute it later.”

Then allow the model to continue.

The question:

> **Can the model maintain “information debt,” and when does it repay it?**

Vary:

- number of unresolved variables;
- distance before they are needed;
- whether they can be reconstructed from the original prompt;
- whether the missing operation is arithmetic, logical or retrieval-like.

Probe whether the hidden state distinguishes:

- known values;
- unknown-but-recoverable values;
- forgotten values;
- values that are no longer relevant.

Work on sparse and shuffled chains shows that models can extract answers from surprisingly damaged traces, but it does not directly characterize deferred unresolved computation. ([arXiv][1])

This gives you a very clean bridge between chain-of-thought, memory and solution-object state.

**Novelty confidence: medium-high.**

## 5. Is a step boundary actually a sufficient state handoff?

After each reasoning step, preserve only the activation at its final boundary token and remove, mask or destroy access to the preceding step’s token-level KV states.

Ask:

> **Can one boundary state summarize everything the next step requires?**

This is subtly different from asking whether punctuation is important. Punctuation and boundary tokens have already been investigated for necessity and sufficiency in several models. ([arXiv][5]) Your version would test a stronger computational claim:

[
\text{full previous step}
\quad\longrightarrow\quad
\text{single state sufficient for continuation}.
]

The key curve is accuracy versus handoff bandwidth:

- one token from one layer;
- one token from several layers;
- several boundary tokens;
- compressed KV states;
- full previous step.

That naturally produces the information-budget Pareto curve you have already been considering.

**Novelty confidence: medium**, because several compression and pause-token papers are adjacent; the formal solution-object framing would be the contribution.

## 6. Can one reasoning state answer a different query?

Give the model a world description and query A. Let it reason until it has built a substantial intermediate state, then replace query A with query B concerning the same world.

Ask:

> **Has the model constructed a reusable world state, or merely an answer-specific trajectory?**

Compare its performance against:

- solving B from scratch;
- continuing from A’s textual trace;
- continuing from A’s hidden state;
- patching a state from a trace originally solving B.

This is an operational test of whether reasoning representations are **query-conditioned plans** or **query-independent solution objects**. It also avoids needing to interpret what every hidden dimension means.

**Novelty confidence: high.**
