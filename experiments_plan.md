# Experiment Plan

The hypotheses are implemented under `src/experiments/`, with thin commands in
`scripts/experiments/`. GPU inference is complete except for the interrupted
paragraph pilot and the prespecified H3 MLP fallback.

## Local Results

Source corpus:

```text
runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k
```

All local metrics use exactly 10 seed-sorted trajectories per question: 580
traces across 58 questions. Raw generation data remains unchanged.

- **H1:** freeform, numbered, and sentence-separated conditions have 60 matched
  traces each; paragraph-separated has 21/60. Literal format compliance is
  `13%` for numbered and `38%` for sentence-separated. Sentence prompting cuts
  matched trace length to `44%` of freeform without a detectable accuracy
  change. Magnitude spikes recover only `3-5%` of symbolic updates; surprisal
  transitions recover `52-64%` with low precision. Segment Silhouette scores
  remain near zero or negative. Prompting does not increase interval path
  length or width; it increases net displacement, favoring compression or
  regularization over the proposed architectural-friction effect.
- **H2:** 17,602 verified symbolic updates reject the point-spike model. By
  question, interval path length ranks at the `73rd` matched-window percentile
  (`95% CI: 71st-75th`), mean peak share is `0.19`, and effective width covers
  `0.98` of the interval. Net displacement is near the `52nd` percentile and
  net/path ratio is `0.17`: distributed, meandering activity rather than a
  coherent impulse. `VERIFY` has the strongest path elevation (`81st`
  percentile); `EXTRACT` is weakest (`26th`), consistent with terminal answer
  readout after earlier construction. The component replay localizes the
  strongest distributed signal to attention output at layer 18 (score `0.739`),
  narrowly ahead of MLP output at layer 18 (`0.732`).
- **H4:** a linear 128-dimensional contrastive projection trained on strict
  lexical controls raises question-disjoint pair AUC from `0.394` to `0.957`
  on 10,421 updates from 278 questions using full-interval net displacement.
  This supports symbolically supervised operator decodability, not natural
  unsupervised clustering.
- **H5:** sentence-segment mean plus variance reaches ROC-AUC `0.757`, `0.740`,
  and `0.762` at 25%, 50%, and 75%. Equal-dimensional symbolic interval
  features reach `0.631`, `0.652`, and `0.700`; sustained-change bands reach
  `0.639`, `0.616`, and `0.680`. Both are significantly worse at every
  checkpoint. H5 is rejected on the current corpus.
- **H3 attention-18:** all 960 cells completed without collapse, but equivalent
  patches did not separate reliably from controls. Full-vector
  equivalent-minus-random accuracy was `+6.9` points (95% CI `-3.6` to
  `+16.9`); subspace was `+5.8` (`-6.7` to `+18.9`). Effects versus mismatched
  patches were only `+3.3` and `+2.8` points. Subspace did not rescue the
  full-vector result, so the null primary result triggers MLP-18.

Reports live under:

```text
runs/SmolLM3-3B/frontier_identification/gsm_symb_pure_mixed_latents_10k/analysis/experiments/
```

Additional reports live under:

```text
runs/SmolLM3-3B/h1_freeform_replay/analysis/experiments/
runs/SmolLM3-3B/h2_component_replay/analysis/experiments/
runs/SmolLM3-3B/h4_structural_replay/analysis/experiments/
```

## Next Run

Run the triggered MLP-18 fallback on GPUs of one model type. The replay already
contains the required MLP activations:

```bash
H3_DEVICES=0,1 ./scripts/experiments/run_h3_protocol.sh fallback
./scripts/remote.sh pull runs/SmolLM3-3B/failed_hypotheses/h3_process_isomer_patching_mlp18
```
