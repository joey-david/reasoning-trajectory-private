# State handoff paper workspace

This folder is the paper-facing map of the state-handoff experiments. It does
not replace the run folders. Every number in the draft and figures comes from a
saved artifact named in [`evidence.json`](evidence.json).

## Reading order

```text
state_handoff_paper/
├── README.md                 this guide and claim boundary
├── evidence.json             machine-readable number → source mapping
├── main.tex                  open paper draft and section skeleton
├── references.bib            sources used by the draft
└── figures/
    ├── 01_native_factorization.png
    ├── 02_explicit_handoff.png
    ├── 03_rate_capacity.png
    ├── 04_closure_training.png
    ├── 05_distribution_shift.png
    ├── 06_local_global_reliability.png
    ├── 07_proof_active_depth.png
    └── 08_register_redundancy_negative.png
```

Regenerate all figures and the evidence map with:

```bash
.venv/bin/python scripts/experiments/build_state_handoff_paper.py
```

## Part I — Native latent trajectories do not guarantee a reusable state

**Question.** Does a model that can read and update state also consolidate a
history into a state that another operation can use?

**Test.** Qwen2.5-32B receives balanced three-bit addition programs. Separate
single-forward prompts ask it to copy a supplied state (Read), apply one update
to a supplied state (Update), execute a history and emit its endpoint
(Synthesize), or execute the history and a pointer-table final rule (Compose).
An exact Python interpreter scores the one-token output. The 1,920 cases span
30 held-out program contexts.

**Result.** At h2, Read and Update are 100%, Synthesize is 76.98%, and Compose
is 13.44%. The model can perform the parts without routing the computed
endpoint into the final operation. Hidden-state probes also retain path detail
and do not expose one transferable implicit code under the tested readout.

**Primary artifacts.**

- `runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history/depth_relief/factorization_summary.json`
- `runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history/depth_relief/state_abstraction/information_summary.json`
- Figure 1.

## Part II — A state handoff is a causal intervention, not a probe

**Question.** Does deleting the history and passing only the current state
repair the routing failure?

**Test.** Call 1 emits the current state. Call 2 receives only that state and
the final rule. Gold handoff replaces Call 1 with the simulator state.
Stepwise execution performs one update per call and discards every used
operation.

**Result.** H2 self-handoff reaches 76.98% versus 13.44% one-pass. Gold handoff
and stepwise decimal execution are 100%. This shows that the supplied state is
a sufficient external interface. It does not show that the unmodified model
forms the same interface internally.

**Primary artifact.**

- `runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history/depth_relief/explicit_handoff/summary.json`
- Figure 2.

## Part III — Information rate and causal meaning

**Question.** How many bits must the interface carry, and do tokens have
state-dependent causal meaning?

**Test.** Qwen2.5-7B LoRA adapters emit one opaque token. Codebooks contain 4,
8, or 16 symbols for an eight-state task, giving declared rates of 2, 3, and 4
bits. Producer and consumer program contexts are disjoint. Test contexts are
unseen. A context-bound control changes the code dictionary across contexts.
In donor interchange, a saved code is substituted into a recipient consumer
that receives no history.

**Result.** The 2-bit code closes perfectly but reaches exactly 50% answer
accuracy, its information ceiling. The 3-bit code reaches 74.27% at h2 and
42.19% at h16. The redundant 4-bit code reaches 97.71% and 59.48%. Gold-code
interchange follows the donor state perfectly for the canonical and redundant
codes. Context-bound codes stay near eight-way chance.

**Primary artifacts.**

- `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls/evaluation/interfaces/comparison_summary.json`
- `runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls/evaluation/interfaces/*/predicted_equivalence_summary.json`
- Figure 3.

## Part IV — Closure training produces long-horizon reuse

**Question.** Is the gain due to a state interface, or merely to more supervised
examples?

**Test.** Both conditions use the same addition programs, 20,000 forwards,
5.12M fixed-padding tokens, and 20,000 supervised one-token targets. The
endpoint control trains history-to-code and code-to-answer. Closure training
instead trains code-plus-one-operation to next-code and code-to-answer.
Evaluation recursively feeds the model its own predicted code. A second test
uses IID, shuffled, cancellation, repeated-operation, and structured histories.

Example semantic program:

```text
start = 2
op1 = add 0 modulo 8
op2 = add 6 modulo 8
FINAL = [3, 5, 7, 6, 2, 0, 4, 1]
true endpoint = 0
answer = 3
```

The producer sees either a history or one code plus one operation and targets
one state-code token. The consumer sees only one code and `FINAL` and targets
one answer token. Prompt tokens carry no loss.

**Result.** Redundant closure reaches 97.81% at h8 and 97.08% at h16, while
matched endpoint training reaches 53.96% and 52.50%. The paired gains are
+43.85 points (95% CI +36.46 to +51.04) and +44.58 (+37.60 to +51.46). Across
the five shifted families, closure averages 93.25% at h16 versus 48.50%.

**Primary artifacts.**

- `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_finetune/evaluation/closure_comparison.json`
- `runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_stress/evaluation/stress/probe/comparison_summary.json`
- Figures 4–5.

## Part V — From addition to state-based reasoning

### Register programs: a replicated limit

Two two-bit registers form a 16-state machine. Instructions add to either
register, XOR registers, swap them, or apply a conditional add. A held-out
16-way dispatch table maps the final register pair to the answer. Three Qwen
seeds and one Mistral run compare an explicit four-bit interface with matched
one-pass training.

At held-out h32, Qwen applies each transition correctly to its own supplied
state 89.31% of the time, but only 7.92% of complete programs finish correctly;
one-pass reaches 5.83%. About 31 self-fed calls amplify modest local error into
near-chance global behavior. An extra redundant bit does not repair this:
completed five-bit seeds reach 12.81% and 12.19%. The third five-bit seed
finished training but evaluation stopped at 510/640, so no aggregate includes
it. This is not an obviously unfinished optimization run: at step 313, the two
complete seeds have 100% batch state and answer accuracy, total losses 0.0031
and 0.0044, 100% validation answer accuracy, and 86.72%/90.63% validation state
accuracy. More training could improve entry accuracy, but it would need a large
change in self-conditioned closure to make h32 reliable.

### Proof states: a positive but small depth probe

Four fact bits represent a proof frontier. Sixty fixed-h64 programs vary the
number of rules that actually add a fact from zero to four. The final question
is binary, and surface length stays fixed. The redundant five-bit interface
scores 81.67% overall, equal to one-pass; at four active deductions it scores
100% versus 75% one-pass. The canonical four-bit interface scores 68.33%
overall and 33.33% at depth four. Each depth has only 12 cases, so this result
supports a redundancy hypothesis but does not confirm it.

**Primary artifacts.**

- Qwen register seed summaries under
  `runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed*/evaluation/`
- `runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_depth_fullrate/evaluation/challenges/*/summary.json`
- Figures 6–8.

## Part VI — Claim boundary

The current evidence supports this claim:

> A language model can learn a small, causally sufficient state channel.
> Long-horizon reuse requires both enough channel rate and closure under
> self-produced states; endpoint supervision alone does not provide closure.

It does **not** yet support these stronger claims:

- that ordinary chain-of-thought models spontaneously form the same discrete
  state interface;
- that redundant codes always act like error-correcting codes;
- that the method broadly improves natural-language reasoning;
- that 90% one-step accuracy is enough for reliable long programs;
- that the proof-depth pilot is a confirmed multi-seed result.

The strongest paper framing joins latent trajectories and state-based
reasoning: hidden trajectories may contain state information, but information
being present is weaker than the model exposing a sufficient, shared, closed
computational interface. The experiments turn that distinction into measurable
rate, closure, interchange, and error-accumulation tests.
