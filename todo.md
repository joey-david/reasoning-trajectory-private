## TODO

### Causal reasoning questions

Branch: `causal-reasoning-questions`

Remaining:

- [x] Keep the completed first-wave artifacts unchanged and add a separate
  second-wave suite.
- [x] Cross same/different formal values with aligned/misaligned trace forms.
- [x] Add direct-read and add-modulo strata so state transfer has both a clean
  positive control and a compositional continuation.
- [x] Patch the actual result or correction token for utility and hysteresis.
- [x] Add behavioral debt strata over known, recoverable, deferred, irrelevant,
  and unpaid dependencies.
- [x] Sweep boundary bandwidth from one residual through the full prior trace.
- [x] Compare answer-focused and query-independent trace transfer to a new
  query.
- [x] Add per-factor results and representation-similarity/effect correlations.
- [x] Add one restart-safe second-wave runner for `ourasi:0` and
  `coktailjet:0,1`.
- [x] Validate both tokenizers and run local text, token-patch, and full-span
  patch smokes.
- [ ] Run the second-wave Qwen and Mistral suites remotely.
- [ ] Inspect raw transfer cells before choosing any result for the paper.

First-wave work:

- [x] Reuse the exact checkpoint residual capture and patching owner.
- [x] Give all six questions deterministic paired data and separate run folders.
- [x] Pin Qwen2.5-7B and Mistral-7B revisions and validate both tokenizers.
- [x] Add validation-only layer choice, fixed test reports, and grouped bootstrap
  intervals.
- [x] Add a text-only control for future-use and information-debt probes.
- [x] Add a same-answer, wrong-state control to the equivalence experiment.
- [x] Add one resumable runner for `ourasi:0` and `coktailjet:0,1`.
- [x] Run dataset tests, a local MLX behavior smoke, and a local 1.5B
  end-to-end residual-patching smoke.
- [ ] Run the two 7B suites remotely.
- [ ] Inspect raw failure cells before promoting any positive summary.

Issues and corrections:

- The first wave gave all six ideas run names, but ideas 2--6 mostly replaced
  the same constant boundary marker. The second wave targets the produced
  value, correction, full history span, or fixed-slot world trace instead.
- A local Qwen2.5-1.5B direct-read smoke changed the answer from recipient to
  donor at every tested non-final layer for both aligned and differently
  worded donors. This proves that the value-token patch path works, but also
  warns that token identity may dominate surface trace alignment. The 7B
  factorial must decide whether similarity adds predictive value.
- A local quantized Llama-3.1-8B text screen scored 6/6 direct reads and 3/6
  add-modulo continuations. The second wave reports those strata separately.
- Full-history, all-layer replacement completed on a local 1.5B case with 55
  aligned Qwen tokens. This checks execution only; the 7B behavior remains
  unmeasured.
- Qwen does not encode `A` as one new token after the chat-formatted `Answer=`
  boundary. Future utility now uses the shared one-token `0/1` alphabet while
  retaining A/B as the semantic register names.
- A same-state swap can look positive when it has no effect. Equivalent-state
  cases now include both a donor that flips the queried fact and a donor that
  changes an unqueried fact while keeping the answer fixed.
- Future utility is stated in the prompt, so a hidden-state probe alone cannot
  establish a special latent tag. The reducer reports a character n-gram
  text-only control and treats causal answer transfer as the stronger test.
- The boundary marker spans 11 Qwen tokens and 12 Mistral tokens. Bandwidth
  conditions therefore patch the final one or three marked tokens, not the
  whole marker; the all-layer condition patches one token at every decoder
  layer.
- The local SmolLM3 text smoke answered four of six first examples with the
  expected leading symbol. It did not follow the one-symbol contract on
  hysteresis or debt and missed the boundary case. This is a useful difficulty
  screen, not evidence about the pinned 7B models.

### 0. Causal state handoff

Branch: `causal-state-handoff`

Remaining:

- [x] Add the five-bit padded canonical control that uses 16 of 32 symbols, so
  raw alphabet size can be separated from path-conditioned code aliases.
- [x] Add a full-support Horn bank over all 16 proof states and all four
  separating fact queries; report exact recovery separately from quotient
  compatibility.
- [x] Add the locked three-graph reducer and decision rules for rate
  sufficiency, gauge fixing, and active depth versus surface horizon.
- [x] Add the restart-safe two-GPU runner and validate its pending task and
  challenge counts locally.
- [x] Audit the causal-state suite against the completed proof failures:
  - match padded state `s` to aliased token `2s` exactly;
  - cross four queries with both alias variants and eight distinct histories;
  - use validation-selected checkpoints for every interface and control;
  - replicate the padded control over all three seeds;
  - gate exact h256 state, worst-state full support, and gold-code continuation.
- [x] Implement tonight's closed proof-state transition suite:
  - balance blocked, idempotent, unconditional, unary, and conjunction updates;
  - train every recursive transition on one rule only;
  - retain a matched encoder fraction and matched outcome-only compute.
- [x] Replace the confounded four-fact depth-four comparison with:
  - endpoint-balanced four-fact depths 1--3;
  - endpoint-balanced five-fact depths 1--4 with an exact 32-code interface.
- [x] Add fixed-active-depth h16/h64/h128/h256 proof streams, per-transition
  error classes, false-positive fact rates, and all-facts collapse checks.
- [x] Add a resumable three-host runner for `coktailjet:1`, `seacove:3`, and
  `upnquick:0`; give the A100 two serial training slots.
- [x] Run the narrow local generator, prompt, orchestration, Ruff, and compile
  checks; commit each tested owner before remote handoff.
- [x] Implement the under-three-hour five-GPU decisive screen:
  - exact rate sweep at 2/3/4/5 bits on the eight-state addition machine;
  - one-step algebra acquisition before held-out operation composition;
  - full-ledger Horn proof actions so every state error changes the target;
  - a two-register program machine with held-out instruction compositions;
  - independent-adapter producer/consumer substitution on a small fixed bank.
- [x] Add one fault-tolerant runner for `ourasi:0,1`, `seacove:3`, and
  `coktailjet:0,1`, with a target runtime below three hours and separate
  logs/status for each wave.
- [x] Keep all new screens to one epoch, 4,000 training forwards per condition,
  at most 1,000 validation forwards, and small context-clustered test banks.
- [ ] Gate three-seed, second-model, longer-horizon, and full Chen et al.
  consecutive-addition confirmation on these screens; do not put confirmation
  work inside the three-hour budget.
- [x] Run `interface-stress-7b` on the saved decimal and opaque adapters.
- [x] Run the matched `interface-closure-7b` transition versus endpoint test.
- [x] Run `interface-closure-stress-7b` after transition closure wins.
- [x] Prepare deterministic three-seed aggregation and two extra Qwen seeds.
- [x] Run and reduce the three-seed Qwen register confirmation.
- [x] Prepare held-out transition compositions and a larger state space.
- [x] Prepare a proof/program execution task with latent state
  that is necessary for the final answer.
- [x] Prepare and token-validate the Mistral-7B proof replication.
- [x] Run the Mistral register comparison; retain it as a failed mixed-h32
  confirmation rather than a positive cross-family result.
- [x] Split tensor/checkpoint helpers and saved-report rendering from the
  training and evaluation owners without changing their artifact contracts.
- [x] Write, compile, and visually inspect the one-page state-channel paper.
- [x] Run the existing explicit adapter through recursive h2 blocks at
  h2/h4/h8/h16 before changing its weights.
- [x] Prepare terminal, recursive decimal, recursive opaque, context-bound,
  and rate-limited interfaces with matched budgets.
- [x] Implement sufficiency, closure, path leakage, actual code entropy, and
  causal same-state/different-state interchange.
- [x] Gate the full 30-context confirmation on a small continuation probe.
- [x] Add a local visual guide to the matched state-handoff training pilot.
- [x] Add artifact-only oracle handoff from saved factorization rows.
- [x] Add resumable LM self, gold, and stepwise handoff inference.
- [x] Analyze the saved Qwen2.5-32B artifacts and enforce a pending Phase 1 gate.
- [x] Run the 32B handoff inference and apply the Phase 1 gate.
- [x] Prepare the frozen Qwen2.5-7B capability screen.
- [x] Generate group-disjoint, hashed pilot train, validation, and OOD test data.
- [x] Add one shared LoRA trainer for `outcome_only` and `explicit_handoff`.
- [x] Add saved-artifact evaluation, plots, and comparison summary.
- [x] Add local tests and a tiny-model save, reload, and resume smoke test.
- [x] Prepare the `scripts/remote` handoff without running it or using SSH.
- [x] Update `experiments/results.md` and `experiments/README.md`.
- [x] Pull and reduce the redundant-register and full-rate proof continuation.
- [x] Build a paper-facing evidence map, eight source-generated figures, and a
  compiled seven-page draft under `experiments/state_handoff_paper/`.
- [x] Pull all three causal-state seeds and the 30-profile challenge matrix,
  rerun the artifact-only reducer, inspect raw recursive traces, and update the
  result ledger.

Issues and corrections:

- The earlier five-fact "rate" plot sampled only five endpoint states, so it
  could not identify a four-bit rate threshold. The replacement rate test
  covers all 16 states, uses four queries that separate every state pair, and
  scores exact state recovery rather than set-valued quotient compatibility.
- A 32-symbol redundant code changes both alphabet size and alias entropy. The
  corrected padded control has the same 32-symbol output softmax and the exact
  same even code `2s` used by redundant variant zero. Only the redundant arm
  adds the odd path-conditioned code `2s+1`, isolating one bit of gauge entropy.
- The first full-support draft repeated one producer rollout once per query and
  tied alias parity to query parity. The corrected bank runs eight distinct
  histories per state and crosses every query with both alias variants. Each
  state, query, variant, and binary answer cell is exactly balanced.
- Prior binary proof answers often stayed correct after the state had already
  collapsed. The new primary gates therefore use exact semantic state, include
  a worst-state threshold over all 16 states, and treat final accuracy only as
  the downstream behavioral consequence.
- All confirmation profiles now load the checkpoint selected by held-out
  one-step state accuracy. This selection never reads a long-horizon challenge
  and applies equally to interface and one-pass arms.
- The causal-state campaign completed all 30 profiles. Canonical and padded
  interfaces recover exact state at 100% through h256 in all three seeds.
  Canonical h256 final accuracy is 97.22% versus 72.22% one-pass, paired
  difference +25 points with context-bootstrap 95% CI +12.5 to +38.89.
- The planned gauge plot used the full-support coverage bank, where both padded
  and aliased codes reach 100% semantic state, so its title overstated that
  contrast. The corrected plot shows both coverage and fixed-h32 depth-three
  deduction. The latter has a replicated padded-minus-aliased exact-state gap
  of 60, 70, and 65 points.
- The locked overall suite remains failed. Its prespecified alias gate used the
  easy full-support bank and observed a zero gap. Its universal gold-consumer
  gate also fails because padded seed 3 reaches 81.25% and aliased seeds reach
  77.34--92.97%. Canonical gold consumers reach at least 95.31% in every seed.
- H256 varies padded surface length while holding active deduction depth to
  one through three. It proves length stability for recursive state updates,
  not 256-step deductive depth. The scaffold also uses 257 calls at h256 versus
  one call for the outcome control, so no inference-compute-efficiency claim is
  valid.
- Local preparation finds exactly five unfinished adapters, 30 interface
  profiles, nine shared one-pass owners, 2,840 interface cases, and 171,520
  recursive transition calls. A synthetic complete-artifact pass exercised
  every reducer branch, verified the exact 0/1-bit information contracts, and
  wrote all four PNGs without model inference.
- The first closed-proof launch stopped before model loading. Every new config
  omitted the inherited `phase1_source_run`, so worker startup raised
  `KeyError: phase1_source_run`; no optimizer metrics or adapters were written.
  All eight configs now declare both gate sources, and the runner checks each
  gate locally before starting remote workers.
- `seacove:3` cannot use the shared PyTorch build: its NVIDIA driver reports
  CUDA 12.2 while the environment needs a newer driver. The retry schedule
  excludes seacove and uses only `coktailjet:1` plus `upnquick:0`.
- Tonight's closure data uses exactly one Horn rule per recursive training
  target. Its six transition classes differ by at most one case in both the
  10,000-program train split and 1,000-program validation split. The matched
  outcome folders have byte-identical train/validation/test hashes.
- The clean four-fact challenge holds endpoints to the four three-fact states
  at active depths 1--3. The five-fact challenge holds endpoints to the five
  four-fact states at active depths 1--4, so depth four no longer identifies
  the all-facts state.
- The same code consumer now handles all/any/parity entailment and a
  `proof_next_rule` operation that returns the first candidate Horn rule which
  adds a new fact. The saved evaluator reports this separately from state
  recovery and gold-code continuation.
- Real Qwen token validation passed every 4-bit and 5-bit run. The longest
  width-five prompt has 439 active tokens against a 512-token training and
  1,024-token evaluation contract. The queue exposes 11 tasks; the challenge
  matrix has 23 profiles and 2,248 interface cases.
- The five-bit register continuation does not repair mixed h32 execution.
  Completed seeds reach 12.81% and 12.19% semantic/final accuracy. Seed 3
  finished training but its evaluation stopped at 510/640, so it has no
  summary and is excluded from aggregates.
- The corrected full-rate proof probe is valid behaviorally. Redundant five-bit
  reaches 81.67% overall and 100% on the 12 depth-four cases, versus 81.67%
  overall and 75% at depth four for one-pass. Canonical four-bit reaches
  68.33% overall and 33.33% at depth four. Small, non-monotone strata make this
  a pilot, not a confirmed rate-reliability result.
- `experiments/state_handoff_paper/evidence.json` is now the paper-number source
  map. Its builder rejects partial runs by requiring their completed summaries;
  the README separates diagnosis, intervention, rate, closure, transfer, and
  claim limits.
- The decisive suite has 12 one-epoch training/evaluation tasks: four exact
  rate arms, one matched outcome arm, one independent rate-16 donor, and one
  interface/control pair for algebra, proof, and the register machine. Four
  workers execute three waves. Horizon-128 addition and horizon-64 proof
  challenges reuse the adapters and add no training.
- Rate arms share the same completed redundant-addition warm start, while each
  one-pass control starts from its nearest completed outcome adapter. This
  avoids treating unequal prior task exposure as a rate effect. Current
  training forwards and fixed-padding tokens remain exactly matched.
- The small screen uses two context clusters and multiple paths per state so
  every redundant code variant appears and quotient agreement is defined.
  Treat its intervals as go/no-go evidence; three-seed and 30-context estimates
  remain gated confirmation work.
- Local Qwen token validation passed all nine configs. Standard h32 one-pass
  prompts peak at 905 tokens and recursive calls at 223; all training calls fit
  the fixed 256-token contract. The separate h128 addition and h64 proof
  controls use 1,676 and at most 1,210 prompt tokens, respectively.
- The next-run suite now has four separate questions: joint encoder/transition
  closure, unseen ordered algebra composition, a 16-state rate sweep, and
  forward-chaining proof state with a live held-out conjunction. Each owns a
  matched one-pass outcome control where needed.
- The first proof generator marked rules as active when their premises held,
  even if the conclusion was already true. Ignoring those rules left the same
  state, so they could not test composition. The corrected four-fact bank
  reserves a state-changing two-premise rule for every endpoint with at least
  three facts. It yields 1,500 causal-conjunction cases: 300 at each of
  h2/h4/h8/h16/h32, balanced across five endpoint fact sets.
- The causal proof subset has `H(S)=log2(5)=2.322` bits. Information retention
  now divides by the empirical entropy of the selected stratum, not the global
  four-bit state entropy. The proof task does not use the register task's exact
  50% answer ceiling because FINAL is binary.
- Qwen and Mistral need a shared one-token 16-state surface. Decimal 10--15
  split in both tokenizers, while Qwen's full-width hexadecimal symbols split
  in Mistral. The proof task uses 16 Cyrillic fact-set labels, 32 separate
  Greek interface codes, and decimal 0/1 answers; all pass both real chat
  boundaries.
- Qwen does not encode decimal 10--15 or ASCII hexadecimal A--F as one token
  after `Answer=`. Four-bit states now use the full-width hexadecimal surface
  alphabet `０１２３４５６７８９ＡＢＣＤＥＦ`; all 16 symbols pass the real
  one-token boundary check while the semantic arithmetic stays in 0..15.
- New chat-formatted one-pass h32 controls need up to 619 algebra tokens. The
  corrected proof controls need up to 687 Qwen tokens and 740 Mistral tokens.
  Training remains fixed at 256 tokens; evaluation has a separate validated
  1,024-token limit. Recursive interface calls stay at at most 110 algebra
  tokens; corrected proof calls need at most 183 Qwen and 203 Mistral tokens.
- Cross-run controls now match two forward passes, 5,120,000 attended padded
  tokens, and 20,000 supervised targets exactly. The saved-artifact comparison
  rejects a mismatch in active tokens as well as forwards and target count.
- Prompt inspection found that interface arms used chat templates while the
  outcome controls rendered plain text: one caller passed the prompt
  sub-config where the shared formatter expected the full config. The
  formatter now accepts both documented shapes, a regression test covers the
  direct sub-config path, and all new outcome-control length manifests were
  regenerated with chat prompts.
- Final local validation passed: 68 focused depth-relief, handoff, LoRA smoke,
  proof-generator, generalization, and replication tests; Ruff; compileall;
  remote-runner shell syntax; Qwen and Mistral token/length checks; and a
  warning-free one-page LaTeX build. No remote job or SSH call was made.
- Interface/control pairs share one dynamic orchestration queue. With two GPUs,
  three interface conditions and the one outcome control form two balanced
  waves instead of leaving one GPU unused for the control.
- Candidate alphabets are validated once at the stable answer boundary, while
  each target extension is still checked. This cut the 66,000-sequence algebra
  preflight from about 90 seconds to 25 seconds without changing token IDs.
- The one-page paper treats the learned object as a closed semantic quotient,
  keeps the 2-bit result as an exact rate control, and separates the current
  finite-state claim from the planned operation-transfer and reasoning-domain
  tests.
- The cleanup keeps the trainer and evaluator below 500 lines. The new runtime
  and reporting modules own tensor/checkpoint work and plot assembly; existing
  private imports remain available from their old modules for test and caller
  compatibility.
- The matched closure gate passed. Redundant transition supervision beats the
  endpoint control by 43.85 points at h8 and 44.58 points at h16 with
  context-paired lower bounds above zero. Canonical transition calls are
  perfect after the first call, but transition-only fine-tuning overwrote its
  decimal-to-code encoder and caps end-to-end accuracy near 55%.
- The closure stress probe confirms the redundant result across structured,
  IID, shuffled, cancellation, and repeated histories: aggregate answer
  accuracy is 97.75/95.25/95.00/93.25% at h2/h4/h8/h16 versus the endpoint
  control's 100/81.25/63.50/48.50%. Canonical closure does not transfer because
  its first encoder call falls to roughly 24--46%, depending on family.
- At redundant h16 in the matched closure run, 42.40% of predictions have the
  wrong nuisance bit but the correct semantic state; only 2.92% change state.
  Treat the learned object as an eight-class quotient of a sixteen-symbol
  channel. Exact-code accuracy is not the primary semantic metric.
- Resume from the step-156 closure checkpoints initially failed because
  `torch.load(..., map_location=cuda)` moved saved CUDA RNG byte tensors onto
  the GPU, while `torch.cuda.set_rng_state_all` requires CPU byte tensors.
  Resume now moves and validates each RNG state on CPU before restoring it.
- The next suite is implemented as three separate owners: artifact-only
  predicted-code equivalence, inference-only five-family stress, and matched
  transition-closure versus endpoint-only fine-tuning. This keeps causal
  analysis, distribution shift, and new training from being read as one result.
- The closure and endpoint controls use byte-identical train, validation, and
  test programs. Their train hash is
  `953b4f63e3fc334d6ef46205cc7cf7c327760e934ee64d16932529b03b5c14e5`;
  both conditions use 20,000 forwards, 20,000 target tokens, and 5,120,000
  padded tokens for the one-epoch comparison.
- The stress probe has 1,600 cases across five contexts, four horizons, eight
  states, two paths, and five history families. Its saved program hash is
  `33e72edb66c5d49ab210794d68241ff3c07010c2cace6604aafea46388f83d7a`.
- The recursive confirmation passed all preset gates: 9,600 cases over 30
  unseen contexts achieved 100% state, answer, local-closure, and same-state
  agreement through h32 with one- and two-operation blocks.
- The first interface evaluation selected step 250 for all four conditions
  because `best_checkpoint` uses answer accuracy alone. This invalidates the
  producer-side canonical and redundant comparisons: their h2 validation state
  accuracy rose from 25.81% to 84.27% and from 31.05% to 96.37% by step 625,
  while answer accuracy was already 100% at step 250. Preserve the early
  evaluation, fix checkpoint selection, and evaluate the final adapters into
  separate artifacts before applying the interface gate.
- Checkpoint selection now uses validation state accuracy for every condition
  with a state/code target and answer accuracy only for `outcome_only`.
  `interface-final-eval-7b` archives the flawed artifacts as
  `evaluation/interfaces_step250/` and evaluates the saved final adapters
  without retraining.
- Final-adapter interface evaluation completed on 3,840 cases per condition.
  The strict interface gate failed: canonical opaque scored 74.27%/58.65%/
  42.19%/42.19% at h2/h4/h8/h16; redundant 4-bit scored 97.71%/80.63%/
  63.85%/59.48%. Context-bound stayed near chance. The compressed 2-bit
  condition achieved exactly 50% answer accuracy with 100% code accuracy and
  closure, matching its information ceiling.
- Redundant exact-code accuracy understates semantic state accuracy because
  either of its two code variants decodes to the same state. At h16 its exact
  code accuracy is 35.73%, but semantic/final-answer accuracy is 59.48%;
  per-step semantic closure after the encoder remains 83.33%--90.83%.
- Reframe redundant codes as a two-representative equivalence class, not only
  as failed canonicalization. The extra bit raises short-horizon reliability
  while lowering exact invariance, suggesting a rate--reliability--invariance
  tradeoff. Measure invariance after quotienting codes by downstream behavior.
- The current interchange artifact reuses `gold_final` calls. It proves that
  the consumer follows supplied gold codes, but canonical same-state swaps are
  identical-token substitutions and it does not test interchange of
  independently predicted codes. Add predicted-donor and alternate-representative
  interventions before claiming a learned causal state code.
- Test IID and adversarial addition streams. Current matched histories set the
  first operation from `path_code`, use a context-periodic middle sequence, and
  choose the last operation to force the endpoint. Equal h8/h16 canonical
  accuracy may reflect this construction rather than a true error plateau.
- The continuation probe has 640 balanced cases over h2/h4/h8/h16 and reuses
  the existing adapter without training. A passed probe unlocks the larger
  h2/h4/h8/h16/h32 confirmation; a failed probe stays saved and does not block
  the separately trained interface controls.
- The rate-control run keeps producer and consumer supervision in disjoint
  training contexts. Its four conditions each validate at 20,000 forwards,
  20,000 target tokens, and 5,120,000 padded tokens per epoch. Pinned hashes:
  train `4c7c62c04ffcec8ceaa62b5347086f70b0bb372fe847ce65177a26bafccb98ec`,
  validation `2cc06eaf70955ec76941de40e439631045e5865f8d33f7a00071c153cdb8dd2d`,
  test `de35ab315505f6b6e524b1d4b4950e43309b21a23271cd1f304c71038589e781`.
- Deterministic information analysis of the failed terminal adapter retains
  3.00 state bits at h2, 0.276 at h4, and 0.040 at h8. Within-state code entropy
  rises from 0 to 2.646 and 2.636 bits, respectively.
- The first LoRA pilot failed its OOD gate. Explicit handoff reached 100% at
  unseen-context h2, then 12.08% at h4 and 11.77% at h8. Gold continuation was
  100% at every horizon, so the failure is history-to-state synthesis, not
  state use or formatting.
- Training was already saturated: h1/h2 validation was 100% at steps 250, 500,
  and 625. More epochs on the terminal-state objective cannot test closure.
- The continuation suite must first reuse the learned h2 mapping recursively.
  This is a feasibility result, not the final paper claim; the stronger claim
  needs equal-rate canonical/context-bound controls, independent producer and
  consumer data, and causal swaps.
- The training explainer lives in the existing static `web/` surface at
  `web/state-handoff-training.html`. It shows the exact two-forward-pass loss
  contract, matched compute, split sizes, training loop, and pilot gate.
- The current branch started from a dirty `layer-paper-replications` tree. The
  depth-relief owners and Qwen run folders are untracked against Git HEAD, so
  commits must stage only files owned by this task plus their required existing
  dependencies.
- The saved 32B run has Compose and Synthesize results, but no LM self, gold, or
  stepwise handoff rows. The artifact-only analysis can test the oracle path;
  the full Phase 1 gate must wait for the prepared remote inference job.
- The 32B artifact analysis completed for all 1,920 cases. At horizon 2, oracle
  execution scores 76.98% versus 13.44% Compose, a paired +63.54 points with a
  program-context bootstrap interval of +58.33 to +68.75 points. At horizon 4,
  oracle execution remains at 12.71% because Synthesize has already failed.
- Phase 1 remains pending: 0/1,920 LM self, gold, and stepwise inference rows
  exist. Do not claim that the gate passed from the oracle result alone.
- The frozen 7B screen now has 3,840 balanced h1/h2/h4/h8 cases across 60
  program contexts. Its dataset hash is
  `2d551239e72c9bd160a05813d6895da11db86bdd57b6f2135c38947f91cc10d0`.
  Local token-contract validation passed for all 29,760 prompts; no inference
  was run.
- Pilot semantic hashes: train
  `b27623034cabce19fe9dcea3dd047728f28a1423e0cda408a18e814de614612a`,
  validation
  `63fca3627a32042ca82b8a93b92968e6632bbffaad6280416bf73e8299f7cd7b`,
  and test
  `8c4215684ad19ea63a6d9998bbd3fe29a4f1e6cfbf9625a1613aae0b321cb50c`.
- The repo virtual environment has no `pip` module. The failed pip attempt made
  no change; `uv pip install --python .venv/bin/python 'peft>=0.19.1'` installed
  PEFT 0.19.1 for local validation.
- The tiny CPU LoRA smoke passed two resumed optimizer steps, finite state and
  answer losses, adapter save/reload equality, and evaluation after removal of
  the training dataset. It did not load Qwen or use a GPU.
- The trainer blocks until the 32B Phase 1 gate passes and the frozen 7B screen
  finishes. Since Phase 1 inference is still absent, no pilot training ran.
- Training now fixes the random seed before it creates LoRA weights, so fresh
  adapters start from the stated seed as well as resumed data order and updates.
- Per-step metrics now stay in the checkpoint until the checkpoint manifest is
  durable. Resume can finish a partial JSONL flush without duplicate rows, and
  it rolls back cleanly if a crash happens before the next checkpoint.
- The pilot comparison summary reuses the saved frozen factorization and handoff
  summaries instead of rerunning either screen.
- The remote runner already owns the terminal `tqdm` bars. State handoff now
  reports its current case and subcall, plus training steps, validation,
  checkpoints, and evaluation batches through that display.
- Opaque codes, causal code interchange, full-scale three-seed training, and
  `interchange_matrix.png` remain gated on a successful pilot and were not
  implemented early.
- `todo.md`, run artifacts, and `AGENTS.md` are ignored. Keep `todo.md` current
  locally; force-add only if the user asks to track it in Git.

### 1. Paul's layer variations experiments

The idea is to track different statistical metrics throughout a forward pass at different steps throughout generation of the CoT, to try and see different things:

**Experiments**:

- [X] Does the cosine similarity from layer output to layer output vary throughout a forward pass? How about the variance of activations ? How does it vary? We're looking for 2d plots, where x is the layer index, and y is the metric tracked (cosine sim, variance, maybe more? mutual information?)
- [X] Throughout generation of the CoT, e.g. when problem-solving, how does the evolution of the prev tracked metrics evolve? Could yield 3d plots - y metrics, x layer, z nth sampled token from the generation.
- [ ] Compare 3d plots of many succesful traces to many unsuccesful traces + compare averages of groups of successful vs unsuccesful traces.

**Depending on the experimental results**:

- If the cosine sim cleanly jumps up (resp falls down for mutual information) at certain layers, earlier or later for certain tokens, with a general trend for jumping (resp dropping) earlier and earlier throughout generation or as the CoT converges on an answer, strong signal that our simple hypothesis of the layers doing a certain amount of work to update the repr of the current token is almost completely verified. Ez paper then.
- If the cosine sim (resp MI) doesn't cleanly jump but is consistently irregular, try to identify the layers for which it is irregular and see if big trends emerge depending on the token type. If so, could yield results on what layers do given types of updates.
- If still no clear trend, go further down to the level of individual heads in layers. Reproduce the experiment, and identify trends in heads doing certain work. Plot one MI/cosine sim line per head, and try to identify head-wise trends.

**Intuition**: [this paper](https://arxiv.org/abs/2502.20332) claims that given heads perform given operations in given layers. We suspect that things aren't so simple, and that transformers aren't this cleanly modular.
Trying to separate CoT into steps (something that a lot of papers do, but don't justify cleanly: [example 1](https://arxiv.org/abs/2604.05655), [example 2](https://arxiv.org/abs/2605.14619)), especially on a latent basis, leads to wondering what the best separation is, and what differentiates token updates from other token updates.
This (potentially information-theoretic) approach may show that transformers need more "depth" for certain token updates, e.g. that certain operations need to make use of all of the layers of a transfomrer while other dumb updates may converge quickly. By converge/use all layers, we mostly mean how the latent representation of the pass evolves layer by layer - if it changes a lot, the model is probably doing work. If not, it's probably not.

**Ideas for future exps**:

- [ ] Compute the mean/median latent displacement of information, or at least the mean magnitude on a LOT of tokens. Then, identify outlier forward passes - find passes that differ a lot from the mean magnitude/direction, on which layers they do, and try to understand why they do. See if certain token types diverge consistently on certain layers.
[ ]

### State handoff overnight sweep

- [x] Add one ordered two-GPU driver for the prepared closure, algebra,
  proof, rate-shift, seed-confirmation, and second-model actions.
- [x] Continue after action failure or timeout and keep per-action logs plus an
  append-only status ledger.
- [x] Stop the active process group on `Ctrl-C`, `TERM`, or `HUP`; preserve
  completed checkpoints and cases for resume.
- [x] Keep every result in its existing run folder and list all owned folders
  in the session's `run_paths.txt`.
- [ ] Run the sweep on an A100 host and pull the listed light artifacts.
- Local note: macOS has no GNU `timeout`, so the local control-flow smoke ran
  without deadlines. The target Linux hosts provide the intended per-action
  timeout path; no GPU or remote process was started locally.
- [x] Fix bulk `scripts/remote.sh pull`: inner SSH no longer consumes the run
  inventory, stalled folders time out without ending the sweep, and signals
  still stop the driver.
- [x] Pull and reduce all nine three-hour run folders.
- [x] Add artifact-only conditional semantic transition accuracy, which scores
  each recursive output against the opaque state actually supplied to it.
- [ ] Continue rate-8/16/32, algebra, and Horn training from 63 to at least 313
  optimizer steps with matched outcome-control budgets. Their last-window
  losses were still falling when the one-epoch scheduler reached zero.
- [ ] Replace the invalid register screen with mixed decimal-entry and opaque
  transition producer examples plus enough consumer training to pass the
  gold-code continuation check.
- [ ] Build proof banks that cross surface horizon with active deduction depth.
  The current h64 bank averages only 2.0 state changes, so it tests ledger
  retention under distractors more than deep proof composition.
- [x] Prepare the focused paper-confirmation suite: three disjoint-data Qwen
  register seeds, one Mistral register pair, five-epoch best-checkpoint
  training, and h64 proof banks with exactly 0--4 active deductions.
- [x] Run `scripts/remote/state_handoff_paper_confirmation.sh`, pull its eight
  run folders, and reduce the Qwen seed, second-model, and active-depth gates.
- Local confirmation validation: all eight interface/control folders pass the
  real tokenizer and length checks. Each pair has byte-identical program banks
  and exactly matched forwards, padded tokens, active tokens, and target
  tokens. The four independent test banks have zero shared case IDs. Each has
  640 H2/H32 cases; held-out H32 paths average 23.99--24.20 real state changes.
- The focused suite passed 71 depth-relief tests, Ruff, compileall, shell
  syntax, and its no-GPU dry run. The remote package exposes exactly eight
  pending tasks across five workers.
- All eight 313-step tasks completed without a GPU failure in 2 h 39 min; the
  proof-depth probe took another 4 min 46 s.
- The dense register H32 mixed-composition gate failed in three Qwen seeds and
  Mistral. Qwen averages 7.92% interface versus 5.83% one-pass at held-out h32;
  Mistral scores 6.88% versus 6.25%. Seen repeated-family h32 remains 31.88%
  versus 9.79% across Qwen seeds.
- The useful mechanism is the local/global gap. Qwen conditional semantic
  transitions remain 88.55--90.69% correct on held-out h32, but about 31
  recursive calls compound to near-chance endpoints. Conditional add is the
  weakest instruction at roughly 74--81%; swap is perfect.
- The first local reducer failed because one path per state makes the
  within-context agreement set empty. Analysis now falls back to same-state
  comparisons across disjoint program contexts and records which scope it
  used. No model rerun was needed.
- The h64 proof probe used a compressed three-bit code with an arbitrary
  16-way action table. That code aliases two ledgers and cannot determine an
  arbitrary action. State accuracy by active depth remains a useful diagnostic
  (100/100/75/75/50% at depths 0--4), but 0% strict final accuracy is not a
  valid causal-handoff result. Repeat with a full-rate proof code.
- [x] Prepare the rate-reliability continuation with one redundant five-bit
  register seed per free GPU, up to three, using the exact saved canonical
  program banks and outcome controls.
- [x] Replace the invalid compressed-action proof probe with byte-identical
  canonical four-bit and redundant five-bit h64 banks whose binary proof query
  is determined by either full-rate code.
- [ ] Run `scripts/remote/state_handoff_closure_polish.sh`, pull the selected
  register seeds and `state_interface_proof_depth_fullrate`, then compare local
  transition reliability, quotient agreement, and active-depth curves.
- Runtime contract: each register seed is one 313-step task. The runner assigns
  at most one seed to each listed worker, so two or three free GPUs should
  finish the training wave in about 60--80 minutes; the proof profiles add
  roughly five minutes and require no training.
- Result note: Horn h64 scores 40.63% versus 15.63% one-pass, paired difference
  +25 points with 95% CI +6.25 to +43.75. Thirteen of 32 predictions are strict
  supersets of the true fact ledger, showing monotone false-fact accumulation.
- Result note: h1-trained algebra retains 81.12% conditional semantic
  transition accuracy over 1,488 held-out h32 recursive calls even though
  end-to-end held-out accuracy falls to 20.83%. Seen h32 is 47.92% versus
  18.75% one-pass.

### 2. J-space work

**Context**: (see [Anthropic's Paper](https://transformer-circuits.pub/2026/workspace/index.html#methods), [Neuronpedia's Qwen3.6-27B implementation](https://www.neuronpedia.org/qwen3.6-27b/jlens))

z_t sortie de couche
MI between $z_l_t$ and final answer. Cheat with MLP.
full greedy, decode different formulations of the same problem. change text, keep calculations constant - change numbers, keep ops, change ops, keep text. Change computations, keep text (a*b + c) instead of (a + b)*c.

Force induce false steps, when crossing a strong computational step with strong inertia, can you recover? what changes how much you recover?
If we have a latent dynamic that follows this false information, then text matters A LOT and CoT constantly rebases itself on it.

Calculation only, how much does that help the model?

How independent, how reliant is the computational graph.

Compare distances between trajs of same language, different ops and different language, same ops.
[x] Make `scripts/remote.sh pull` exclude PyTorch `.pt` and `.safetensors` files by default; add `--pt` opt-in.
[x] Show resumed optimizer-step bars and epoch-aware ETAs for each state-handoff training worker.

### Balanced proof-state weekend confirmation

- [x] Remove the pilot's depth/answer imbalance by using positive-depth states
  and a 50/50 proof query within every depth, horizon, and topology stratum.
- [x] Prepare matched compressed 3-bit, canonical 4-bit, and redundant 5-bit
  conditions for three Qwen2.5-7B seeds.
- [x] Add Qwen2.5-3B and 14B scale checks plus a Mistral-7B family check.
- [x] Add fixed-h64 depth 1--4, h16/64/128 length, and
  independent/chain/conjunction proof banks.
- [x] Add one restart-safe two-GPU runner and one aggregate reducer with
  prespecified depth, length, topology, rate, and seed gates.
- [ ] Run `scripts/remote/state_handoff_proof_weekend.sh` on both ourasi GPUs.
- [ ] Pull the anchor and source runs, inspect raw errors and loss curves, then
  update `experiments/results.md` from saved artifacts.
- [x] Stop the closed proof runner from launching challenge evaluation when any
  linked adapter or its base evaluation is incomplete. The first retry exposed
  this after training stopped before `outcome_seed1` wrote its checkpoint
  manifest; the challenge traceback was a downstream error, not the training
  cause.
- [ ] Inspect the failed closed proof session's `training.log`, fix the first
  training error, then restart. Completed adapter/evaluation pairs remain
  resumable inputs and must not run again.
- [x] Fix the closed-proof base evaluation contract: its h1 cases cannot use
  the old two-rule default block. All interface runs now evaluate the
  trained one-rule transition with `block_size: 1`, and preflight rejects any
  future test horizon that does not split into complete blocks.
- [x] Move the closed-proof runner default from occupied `upnquick` to all four
  free GPUs on `coktailjet` and `kaisertrot`.
- [x] Fix the closed-proof reducer to read the shared bootstrap schema's
  `ci95[0]` lower bound. All remote training and challenge tasks completed;
  only this artifact-only reduction had failed.
