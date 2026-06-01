# Shared Trajectory Schema

The central schema lives in `reasoning_trajectory.core.schema`.

`Trajectory` stores run metadata, prompt/output text, final correctness, optional solution object metadata, and ordered `Step` records. `Step` stores token spans, text, per-layer hidden states, optional logits, verifier state, labels, and metadata. `VerifierState` labels process feedback such as valid transition, branch point, recoverable failure, goal-reducing, or dead end. `SolutionObject` represents proof, program, symbolic, or verifier-state objects.

Storage helpers in `reasoning_trajectory.core.storage` support JSON, JSONL, NPZ tensors, and JSONL/Parquet tables. Small examples can keep hidden states inline in JSONL; larger runs should place heavy arrays in NPZ and keep references in metadata.

Runnable commands:

```bash
rt run --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30 --layer 32
rt analyze --input experiments/runs/r1_distill_sheep30 --layer 32
```
