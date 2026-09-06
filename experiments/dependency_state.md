# Usable reasoning state: pending dependencies and correction validity

## Objective

Explain when an available intermediate result remains usable in later reasoning.
Seek held-out prediction, a causal explanation, and better complete solutions;
neither probe accuracy nor an oracle-guided digit flip meets that standard.

The existing routing interventions establish access to earlier values on controlled
tasks. They do not establish a mechanism for natural long-run failures. The natural
digit assay resets the prefix at each digit; its `exact` metric is not a generated
number or continuation. The prior `confidence` baseline uses the gold-answer margin,
not deployable uncertainty. Preserve these artifacts, but do not extend their claims.

## A. Pending dependencies

Target claim: interference between pending result–use bindings predicts a substantial
part of reasoning failure beyond length, arithmetic difficulty, and retrieval distance.
A lower-load solution order improves held-out accuracy at matched compute.

1. Execute a verified arithmetic dependency graph in different valid orders. Keep
   operations, literals, names, and final answer fixed within each pair. Compare
   branch-local and breadth-first orders. Cross branch count, depth, and equal-value
   collisions; compute live-variable counts and definition–use distances from the AST.
2. Score complete generated answers and inspect the first wrong step. Never assemble
   an answer from independent teacher-forced digit tests. All paired variants share
   a family ID; numeric variants are not independent graph families.
3. Fit a small predictor on development graphs; compare live load with length,
   operation count, source distance, and answer-independent uncertainty. Test once on
   new graphs, longer dependencies, and another model. A schedule effect alone is
   insufficient: the pilot changes distance too, and does not force the model's
   internal or written solution order to follow the code order.
4. If the behavioral effect survives controls, test binding versus value loss using
   direct retrieval probes, same-value/different-entity donors, and source-selection
   interventions. Compare with generic retrieval heads, not only random heads.
5. Test a low-load schedule on verified natural math/code solutions at matched token
   budgets, including the cost of selecting that schedule. No state-monitoring prompt.

Falsifiers: distance alone explains the paired effect; held-out prediction fails;
or scheduling helps only the supplied synthetic code. Do not call live width a hard
transformer capacity bound: the model can reread and recompute.

## B. Correction validity

Target claim: a model can use a corrected premise while still using conclusions
derived from its old value. Selectively invalidating those conclusions restores
downstream reasoning without damaging independent branches.

1. Supply verified partial calculations for two branches, then revise one initial
   input. The ordinary program output includes that input, a joint downstream result,
   and an independent branch result. Score all three separately and jointly.
2. Pair real and no-op revisions; vary dependency depth and equal-valued branches.
   Compare intact history, removal of affected descendants, removal of the same
   number of independent lines, removal of the old root line only, and a fresh solve
   of the revised program. Record actual lengths: line matching is not token matching.
3. Primary diagnostic: corrected root with a stale downstream answer and a preserved
   independent result. Report unconditional rates as well as conditional rates; do
   not filter to successes when estimating overall repair. Use paired differences
   within families, and separate no-op damage from revision benefit.
4. After the pilot, repeat on the model's own checked calculations, with sequential
   revisions and held-out natural tasks. Supplied work is not evidence of spontaneous
   self-correction. A restart is a reference, not a compute-matched intervention yet.
5. Locate causal selection of invalid descendants; distinguish repairing source
   selection from injecting the gold value. Freeze the discovery set and intervention
   before confirmation. Recomputing identical causal text is not a cache repair.

Falsifiers: the model never accepts the changed input; general arithmetic errors
explain failures; unrelated deletion works equally well; or selective deletion breaks
independent conclusions. Then the specific validity mechanism lacks support.

## First implementation and run contract

The first runs are **behavioral diagnostic pilots**, not confirmation of either claim.
They reuse the repository generation pipeline, Hugging Face Transformers, Python AST
and graphlib, and the run-folder artifact schema. Python executes only validated
straight-line integer programs made by the builder, never model output. The scorer
uses `ast.literal_eval` on a final `Answer:` line; missing/invalid answers count as
failures, with no last-number fallback. Generation uses EOS or a fixed cap, no numeric
regex stop. Save full continuations, token IDs, capped-output counts, and paired scores.

Run two models independently on the same pinned cases, one H100 per run, at most two
hours each. First smoke both experiments; inspect actual text, token counts and scores
before larger submissions. Do not launch mechanistic sweeps on a failed task contract.
Use separate pilot seeds; these templates never become a held-out confirmation set.
The code task does not establish a natural-reasoning claim, even if the pilot succeeds.

Smoke inspection found duplicate BOS tokens in the shared chat-generation path for
Mistral (`[1, 1, 3, ...]`). The caller now disables extra special tokens for rendered
chat prompts, following [Transformers guidance](https://huggingface.co/docs/transformers/v4.48.0/chat_templating).
Plain prompts still receive their normal tokens. Keep the original smoke artifacts;
rerun Mistral in fresh `dependency_*_bos_smoke` folders before any broader Mistral run.
Earlier Mistral generations through this shared path need a token audit before reuse.
The Qwen smoke's changed-input errors include incorrect arithmetic after correct
revision; they are not evidence of failed invalidation. A real-output regression test
keeps that distinction. Broader order pilots allow 2,048 tokens because their programs
have up to 49 lines (the nine-line smoke already uses 226 tokens); revision stays at
1,024. These limits apply equally to all paired arms and are fixed before launch.

Llama's initial smoke emitted the correct order answer, then `<|eot_id|>` (128009),
then repeated assistant turns until the cap. The cached checkpoint's generation
config only lists 128001; the runtime also replaced model EOS lists with the
tokenizer's scalar EOS. The runtime now preserves model EOS lists and accepts a
standard per-run `generation.eos_token_id` override. Llama runs pin `[128001, 128009]`,
as in the [model's usage example](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
Real tiny-model tests check that a stop token other than the tokenizer EOS ends
generation immediately, through both model defaults and a run override. Fresh
`dependency_*_eot_smoke` runs must stop correctly before a broader Llama pilot.

## Prior work and novelty boundary

- [Retrieval heads](https://arxiv.org/abs/2404.15574) already connect retrieval to CoT.
- [Entity state tracking](https://arxiv.org/abs/2605.30233) studies read-time aggregation,
  binding and state changes; sparse state-related heads alone are not novel.
- [Reasoning DAG probing](https://arxiv.org/abs/2601.17593) studies decodable graph
  structure, not the causal effect of equivalent execution orders on complete answers.
- [Working-memory limits](https://aclanthology.org/2024.emnlp-main.938/) studies n-back
  behavior. Live-variable analysis is established compiler theory, not our invention.
- [Belief revision](https://aclanthology.org/2024.emnlp-main.586/) already tests adapting
  reasoning after changed premises. Dependency-based invalidation is established
  truth maintenance. The proposed contribution needs causal downstream selectivity
  during reasoning, not simply another correction benchmark.

This pass does not establish exhaustive novelty. Before any confirmatory campaign,
check newer dependency-aware agent repair work and replicate existing code where its
experiment already answers our question.
