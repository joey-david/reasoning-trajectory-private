# Computation interference

Branch: `research/computation-interference`.

## Question and target claim

Does a model reuse an earlier calculation after matching its operands but missing
a changed operator? The candidate mechanism is a head-to-result attention path
that supports valid reuse but also carries results into the wrong computation.
The [old pilots](experiments/dependency_state.md) do not establish this explanation:
they found neither consistent restart gains nor selective-deletion repair.

## First experiment

Hold the current expression and final task fixed. For example, request
`2 * (620 + 130) + 7` after true working calculations containing one of:

| Prior calculation | Purpose |
|---|---|
| `620 + 130 = 750` | Valid reuse |
| `620 - 130 = 490` | Same operands, changed operator |
| `700 - 210 = 490` | Same prior result and operator, different operands |
| `310 + 180 = 490` | Same prior result, current operator, different operands |

Match token counts and source positions across histories within each model.
Screen every attention head by blocking only its access to the earlier result
at the current query. Use all eight discovery cases; freeze four heads before
16 new test cases with longer histories. Compare unmodified generation, source
blocking, layer-matched control heads, and an adjacent control-number source.
The latter matches token count, not exact distance or attention mass.

Require selective reduction in copying the wrong prior result and better complete
answers beyond both controls. Test valid reuse with those same heads. Record
first-token probabilities separately from full generated-number accuracy, and
count format failures and damage to correct answers. No gold-value injection,
independent digit reconstruction, state-monitoring task, or probe-based claim.

## Next stage, conditional on that evidence

Separate source selection from value content using key-versus-value interventions
between equal-valued source calculations. Freeze the resulting error predictor
and intervention, then test on unseen natural, model-generated reasoning prefixes
and complete continuations. The first supplied-arithmetic pilot cannot establish
natural reasoning, a hidden-state representation, or long-run reliability.

## Reuse, prior work, and status

Reuse Hugging Face masks and generation, the repo loader, token alignment,
arithmetic verifier, and head selector. Preserve older experiments and raw data.
[Function induction](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8d9bbba8cac9cabb54e85ee7f21441c8-Abstract-Conference.html)
and [RaSteer](https://arxiv.org/abs/2507.00322) already cover arithmetic functions
carried by heads and improvement by component steering. The candidate distinction
is source-specific misuse of an entirely correct prior calculation, with its
numeric result held fixed in the controls. Novelty remains provisional.

29 focused local tests pass, including real Llama/Qwen modules, GQA, mask scope,
hook cleanup, and a saved Llama failure regression. Smokes **1817030/1817031**
completed in under 90 seconds: 32/32 correct continuations per model, no caps,
correct EOS, matched token contracts, and zero attention on blocked edges.
This validates plumbing, not the hypothesis. Initial jobs 1816914/1816915 failed
before inference because the submission shell lacked the site's module path.

Seed-26090611 pilots: **1817248** [Qwen](runs/Qwen2.5-7B-Instruct/interventions/computation_interference_pilot/)
and **1817249** [Llama](runs/Meta-Llama-3-8B-Instruct/interventions/computation_interference_pilot/).
Each uses one H100, a one-hour limit, eight screen families, and 16 test families
(256 full continuations). Source and dataset hashes are fixed in each manifest.
Initialize the login shell with `source /etc/profile.d/z_modules.sh` before
`sbatch`; set the reasoning repo's `repo_root` before sourcing the shared GPU env.

Entry point: `scripts/experiments/computation_interference.py prepare|run|reduce RUN`.
