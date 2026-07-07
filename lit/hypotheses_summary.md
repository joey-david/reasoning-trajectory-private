# Summary of Hypotheses and Results

## H1: Text Boundaries Versus Latent Boundaries

**Idea and purpose.** Test whether naturally occurring reasoning boundaries
coincide with sharp latent transitions, and whether forcing numbered, sentence,
or paragraph boundaries creates measurable architectural friction.

**Result.** Magnitude spikes recovered only 3–5% of symbolic updates, while
surprisal transitions recovered 52–64% but with low precision; segment
Silhouette scores were near zero or negative. Prompting increased net
displacement rather than path length or interval width, favoring compression or
regularization over a boundary-friction account.

[Implementation](../src/experiments/boundary_comparison.py) ·
[Results](../runs/SmolLM3-3B/pilots/h1_freeform_replay/analysis/experiments/h1_boundaries/report.json)

## H2: Localized Solution-Object Updates

**Idea and purpose.** Use symbolically verified arithmetic updates to test
whether intermediate solution objects are written by brief, localized latent
events or by extended changes across the update interval.

**Result.** Across 17,602 updates, interval path length ranked at the 73rd
matched-window percentile, but mean peak share was only 0.19, effective width
covered 0.98 of the interval, and net/path ratio was 0.17. Updates therefore
resemble distributed, direction-changing activity rather than discrete latent
spikes.

[Implementation](../src/experiments/localized_updates.py) ·
[Results](../runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h2_localized_updates/report.json)

## H3: Causal Process-Isomer Patching

**Idea and purpose.** Patch full or H4-restricted activations between different
derivation paths that reach the same symbolic state, testing whether the
completed solution object is portable and causally sufficient.

**Result.** In 960 attention-layer-18 continuations, equivalent patches improved
accuracy over position-random controls by 6.9 points for full vectors (95% CI:
−3.6, 16.9) and 5.8 points for subspace patches (−6.7, 18.9); effects versus
mismatched states were only 3.3 and 2.8 points. The intervals include zero and
subspace restriction did not rescue the effect, giving no reliable evidence for
a portable state-specific object at this site.

[Patching implementation](../src/experiments/causal_patching.py) ·
[Analysis implementation](../src/experiments/patching_analysis.py) ·
[Results](../runs/SmolLM3-3B/failed/h3_process_isomer_patching/analysis/report.json) ·
[Raw continuations](../runs/SmolLM3-3B/failed/h3_process_isomer_patching/patching/continuations.jsonl)

## H4: Structural Operation Discovery

**Idea and purpose.** Ask whether latent interval transitions naturally group
by arithmetic operation across lexically unrelated problems, then test whether
symbolic supervision can expose operation-relevant directions.

**Result.** On 10,421 updates from 278 questions, raw cosine AUC was 0.394, but a
supervised 128-dimensional linear projection reached 0.957 on held-out
questions. Operation identity is strongly linearly decodable but not salient in
raw geometry; this supports supervised recoverability, not natural clustering,
causal use, or a non-linguistic schema.

[Implementation](../src/experiments/structural_contrast.py) ·
[Results](../runs/SmolLM3-3B/replay/h4_structural_replay/analysis/experiments/h4_structural_contrast/report.json)

## H5: Correctness Prediction by Latent Segmentation

**Idea and purpose.** Test whether representations aggregated over symbolic
updates or sustained-change segments predict final correctness better than
ordinary sentence-level summaries.

**Result.** At 25/50/75% checkpoints, sentence features achieved ROC-AUC
0.757/0.740/0.762, versus 0.631/0.652/0.700 for symbolic intervals and
0.639/0.616/0.680 for sustained-change bands. The proposed latent segmentations
were consistently worse, so they do not provide a superior predictive unit on
this corpus.

[Implementation](../src/experiments/correctness_prediction.py) ·
[Results](../runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/h5_correctness_prediction/report.json)

## Overall

The experiments consistently argue against sharp, context-independent latent
solution objects. The strongest positive result is that operation information
can be extracted with symbolic supervision; the broader evidence instead
favors distributed, path-dependent reasoning dynamics.
