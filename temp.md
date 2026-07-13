# Verdict

The closest unoccupied hole is:

## **Causal depth relief: does writing an intermediate reasoning state reduce the layer depth required for the next operation?**

The corresponding thesis is:

> **Chain-of-thought is a depth-to-memory conversion protocol.** A difficult forward pass computes a state in later layers; emitting a checkpoint makes that state persist in the context; the following operation can then use it with less internal depth.

This is more central than “reasoning trajectories have curvature,” “thinking tokens are information peaks,” or “written states matter.” Those claims are now occupied. What remains unmeasured is the **resource exchange itself**:

$\text{causally usable scratchpad information}\quad\longleftrightarrow\quad\text{downstream layer depth}.$

I found no primary paper that measures this exchange in standard autoregressive reasoning models, with self-generated intermediate states and causal controls, as of July 13, 2026. Several papers establish every adjacent edge of the argument, but not this edge.

---

### 5. “CoT serves as persistent memory”

**Theoretically occupied.** Opaque Serial Depth formalizes the architectural argument that visible autoregressive outputs allow information produced late in one forward pass to become available to early computation on later passes. A memory-budget separation proves that CoT has a persistent mutable scratchpad unavailable to compressed recurrent states. A specialized Universal Transformer experiment even shows that learned memory tokens and recurrent ponder depth substitute for one another.

But that last result uses a single-block adaptive recurrent architecture and learned memory tokens. It does **not** establish the tradeoff in an ordinary decoder-only LLM using self-generated natural-language or symbolic reasoning states.

That is the remaining hole.

---

# The actual hypothesis

Let $S_k$ be the latent state after reasoning step $k$, and let $z_k$ be the emitted token span describing it.

A checkpoint cycle should have four phases:

1. **Compute:** $S_k$ becomes causally available only in middle or late layers.
2. **Write:** the model emits $z_k$, placing a representation of $S_k$ into the persistent token/KV context.
3. **Read:** the next position retrieves $S_k$ from that checkpoint.
4. **Relief:** the next update $S_{k+1}=f_k(S_k)$ becomes causally determined at a shallower depth than if $S_k$ had to be recomputed.

Define the downstream causal depth under a condition $c$ as

$D_{k+1}^{(c)} = \min\left\{\ell:\text{the correct next-state computation is causally fixed by layer }\ell\right\}.$

Then define **depth relief**:

$R_k = D_{k+1}^{(\text{no checkpoint})} - D_{k+1}^{(\text{checkpoint})}.$

The central prediction is

$R_k>0$

for genuinely useful reasoning checkpoints.

Crucially, measure the depth of the **next state update**, not the checkpoint token itself. The checkpoint’s identity is present in its embedding, so showing that it is shallowly decodable would be trivial. The question is whether it makes the _following computation_ shallower.

---

# The decisive experiment

## Controlled task family

Start with tasks whose true state and transition rule are exactly known:

- pointer chasing through random maps;
- modular function composition;
- small program/register execution;
- variable binding;
- finite-state systems like the causal-register paper;
- symbolic compositions adapted from the abstract-reasoning mechanisms literature.

Construct each problem as

$S_0 \xrightarrow{f_0} S_1 \xrightarrow{f_1} S_2 \cdots \xrightarrow{f_{K-1}} S_K.$

Use states with adjustable information content—one bit, two bits, four bits, and so on—so you can estimate an actual information–depth curve.

For every transition, evaluate four matched conditions:

| Condition               | Prefix before predicting (S\_{k+1})              |
| ----------------------- | ------------------------------------------------ |
| No register             | Original problem and operations; (S_k) omitted   |
| Self-written register   | The model generated (S_k) itself                 |
| Gold register           | The correct (S_k) is inserted                    |
| Counterfactual register | A minimally altered but valid (S_k') is inserted |

Add length-matched random and semantically empty spans. The essential signature is:

$D_{\text{self}}\approx D_{\text{gold}}<D_{\text{none}}.$

while the counterfactual register produces the **rule-consistent wrong branch at shallow depth**:

$S_{k+1}'=f_k(S_k').$

That last condition rules out generic confidence, copying and next-token steering.

## Measure depth three ways

**Distributional settling depth.** Use the DTR construction—distance between intermediate and final output distributions—as an inexpensive screen. DTR already formalizes per-token settling depth, but it does not test whether a prior state caused the settling to occur earlier.

**Read/integration depth.** At the prediction position for $S_{k+1}$, prevent attention to the checkpoint span from layer $\ell$ onward. Once this intervention stops affecting the next-state prediction, the checkpoint has already been integrated:

D_k^{\text{read}} = \min\left\{\ell:\Delta_{\text{mask checkpoint after }\ell}\approx 0\right\}.$

**Causal state editing.** Hold the written text fixed while editing the state-specific residual/KV representation, following the causal-register design. The continuation must change according to (f_k), not merely toward an arbitrary target token. This separates state presence, state verbalization and actual state use.

The first method is correlational and scalable. The latter two make the result causal.

## State information rather than generic HSIC

Because the latent state is known, estimate information through held-out predictive log loss:

$\widehat I(S_k;H_{t,\ell}) = H(S_k) - \widehat H_{\mathrm{CV}}(S_k\mid H_{t,\ell}).$

This gives a cross-validated lower bound tied to a meaningful variable. MI Peaks instead applies HSIC to final-layer representations and the gold answer; its authors explicitly leave the origin of the peaks unresolved.

With variable state entropy, test a **depth–state rate curve**:

$D(r)\quad\text{versus}\quad I(S_k;Z_k)=r.$

where (r) is the number of correct state bits exposed in the checkpoint. The strongest simple result would be a monotonic law:

> At matched accuracy, additional causally usable checkpoint information systematically reduces the layer depth needed for the next update.

That is a genuine information-computation result, not a physical metaphor imposed on cosine curves.

---

# The paper-quality result ladder

## Minimum publishable result

Across at least two model families and three controlled task types:

- self-written and gold checkpoints reduce downstream settling/integration depth;
- random or irrelevant text does not;
- counterfactual checkpoints redirect computation according to the task rule;
- the effect survives token-length, position and token-frequency controls.

This extends causal-register work from “the state is used” to “the state substitutes for internal depth.”

## Strong result

Reasoning post-training changes the exchange rate.

Compare:

- Qwen3 thinking versus non-thinking mode;
- a reasoning-distilled checkpoint versus its corresponding base/instruct model;
- optionally a model fine-tuned with explicit running-state supervision.

The prediction is not merely that reasoning models use more deep-thinking tokens. It is that they are better at **turning expensive internal computation into reusable external state**:

R_k^{\text{reasoning model}} > R_k^{\text{base model}}.$

This would give a concrete mechanistic account of what reasoning training buys.

## Very strong result

Decompose errors into three causal classes:

1. **Compute failure:** the correct (S_k) never becomes available internally.
2. **Write failure:** the correct (S_k) appears in late-layer or workspace representations, but the emitted checkpoint preserves the wrong state.
3. **Read failure:** the correct state is written, but downstream computation bypasses or misuses it.

This is especially plausible because activation patching has already found correct-answer-recoverable information inside some incorrect CoTs, while newer work distinguishes latent knowledge from its verbalization in other mathematical settings. Neither gives a repeated, state-by-state compute/write/read decomposition of reasoning failure.

A robust taxonomy would be more valuable than another correct-versus-incorrect classifier because every category implies a different intervention.

---

# The next experiments, ordered by value

### 1. State-information dose response

Expose (0,1,\ldots,b) correct bits of (S_k), holding span length fixed. Test whether downstream depth decreases monotonically with usable information rather than with token count.

This is the cleanest information-theoretic experiment.

### 2. Self-written versus externally supplied state

Compare the same checkpoint when generated by the model and when supplied as gold text. A self-written state matching gold in both causal use and depth relief would show that the model has genuinely established a register protocol.

### 3. Visible-token versus hidden-KV contribution

A scratchpad position is hybrid: later tokens can use its visible token identity and its layerwise cached K/V representations. Decompose depth relief into:

$R^{\text{text}},\qquad R^{\text{hidden-KV}}.$

Hold text fixed and edit hidden state; then hold the state representation fixed while changing or neutralizing the visible symbol. The causal-register paper proves that hidden state can matter even with unchanged text, but does not quantify how much downstream computation that hidden channel saves.

This also determines whether visible CoT is a faithful computational record or merely the readable surface of a richer hidden register.

### 4. Native reasoning-event alignment

On GSM8K or programmatically generated arithmetic, align token×layer fields around machine-verifiable intermediate results. Look for:

- late-layer state formation before emission;
- a checkpoint write;
- lower downstream settling depth;
- the next state formation event.

The predicted visualization is a repeated sawtooth or late-to-shallow handoff, not a smooth global annealing curve.

### 5. Depth-relief-guided CoT compression

Delete spans with low (R_k), retaining spans that actually reduce future computational depth. Compare against:

- MI peaks;
- DTR;
- attention;
- token likelihood;
- semantic-step pruning;
- random matched deletion.

The method wins only if it preserves accuracy better at matched token budgets. This would turn the scientific result into a practical improvement.

### 6. J-space handoff

Measure whether (S_k) first appears in J-space, becomes verbalized, and is subsequently restored into the workspace from the checkpoint. Anthropic shows that J-space supports causal intermediate reasoning and future verbalization, but does not study long CoT as a repeated workspace-to-context-to-workspace protocol.

This is the highest-upside follow-up, but it is not the first experiment: the Jacobian-lens infrastructure and sparse decomposition add substantial complexity.

---

# What your current bundle says about this

Your present data remain useful, but they cannot establish this result.

The bundle contains group-averaged trajectories for 74 correct and 26 incorrect generations. Over the shared prefix in which no rollouts have terminated, directional persistence rises strongly in roughly layers 4–17 for both outcome groups, while normalized update magnitude falls. The layerwise patterns are almost identical between correct and incorrect traces. That makes them more consistent with a **generic reasoning protocol** than with a correctness discriminator.

The problem is that averaging by absolute token index destroys precisely the event structure now of interest. If different rollouts write useful states at different times, a repeated compute–write–read cycle becomes a smooth apparent annealing trend after averaging.

For the next runs, retain:

$\text{problem},\text{rollout},\text{token},\text{layer},\text{feature}$

along with token text, exact state labels, step boundaries and correctness. Do not aggregate before event alignment.

Your current metrics should become secondary diagnostics:

- directional persistence: does computation straighten after a checkpoint?
- normalized update magnitude: does downstream action fall after the write?
- cosine and normalized update magnitude: treat as mostly redundant;
- raw Euclidean distance: normalize for residual-stream norm.

The primary object should instead be a **state-conditioned causal information field**

$\mathcal C_k(t,\ell),$

showing where a specific intermediate state is formed, written and subsequently used.

---

# Final ranking of the surviving holes

**1. Causal depth relief / depth–state substitution.**
Closest to the center, conceptually simple, and experimentally decisive.

**2. Compute–write–read failure decomposition.**
Likely the strongest correctness result, using the same infrastructure.

**3. Visible-text versus hidden-KV depth relief.**
More technically difficult, but important for CoT faithfulness and monitorability.

**4. J-space checkpoint handoff.**
Highest mechanistic upside, but substantially more expensive.

**5. Depth-relief-defined reasoning steps.**
Define a reasoning step as a minimal span whose emission measurably lowers the causal depth of the remaining computation. This could finally give your “solution-object update” or segmentation work a non-heuristic operational definition.

I would now demote the kinetic-energy framing. It can supply descriptive measurements, but it is not the hole. The hole is the missing empirical law connecting **information written into persistent reasoning state** to **computation no longer required in depth**.

A clean paper title would be:

## **From Depth to Memory: Chain-of-Thought as Causal Checkpointing**

Its central result should be:

> Reasoning-trained language models convert late-layer computation into persistent intermediate state; causally usable checkpoints reduce the depth required for subsequent operations, and reasoning failures can be localized to computation, writing, or reading.

A weekly arXiv watch for causal scratchpads, depth–state tradeoffs and token×layer reasoning papers would help protect this novelty claim; tell me to schedule it.
