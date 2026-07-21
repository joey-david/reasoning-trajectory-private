# Layer-paper replication checklist

Generated from pulled artifacts. A completed job passes only when its
central empirical result also passes the paper-grounded check.

## [ ] The Remarkable Robustness of LLMs: Stages of Inference?

Fidelity: paper protocol on a paper-listed Qwen2.5-1.5B model.

- [ ] complete intervention matrix: expected all drop/swap cells; observed null
- [ ] middle layers are deletion-robust: expected middle KL < edge KL; observed null
- [ ] middle layers are swap-robust: expected middle KL < edge KL; observed null
- [ ] middle swaps are less harmful than drops: expected swap KL < drop KL in middle half; observed null
- [ ] middle deletion preserves predictions: expected middle top-1 consistency > edge consistency; observed null

## [ ] Emergent Symbolic Mechanisms Support Abstract Reasoning in LLMs

Fidelity: paper-sized Qwen2.5-7B identity-rule CMA.

- [ ] complete causal-mediation matrix: expected all selected pairs; observed null
- [ ] significant heads for all three mechanisms: expected FWER p<0.05 count > 0 for each mechanism; observed null
- [ ] three-stage depth hierarchy: expected abstraction center < induction center < retrieval center; observed null

## [ ] Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training

Fidelity: published Qwen3-1.7B anchor layers; full 28-layer scan optional.

- [ ] core base/full/layer evaluations: expected base, full, and layers 1/7/10/12/24; observed {"missing": ["base", "full", "layer-01", "layer-07", "layer-10", "layer-12", "layer-24"]}
- [ ] full GRPO improves the base model: expected full math average > base; observed null
- [ ] best published middle layer matches full GRPO: expected max contribution at layers 10/12 >= 0.9; observed null
- [ ] middle-layer contribution concentration: expected mean layers 7/10/12 > mean layers 1/24; observed null
- [ ] late control underperforms best middle layer: expected layer 24 contribution < max layer 10/12 contribution; observed null
