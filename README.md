# Reasoning Trajectory

Research sandbox for LLM reasoning analysis.

## Workflow

Each experiment is a run folder:

```text
runs/<model>/<experiment>/
  config.yaml
  dataset.jsonl        # optional prepared snapshot
  generation/          # generated text, per-run JSON, optional activations
  analysis/            # later analysis outputs
```

### Dataset preparation

1. Select the dataset in `config.yaml`.
   Use `dataset.source: hf` for Hugging Face datasets or `jsonl` for a local file.
   Set `dataset.adapter`, `split`, `sample_offset`, `sample_limit`, and optional
   `shuffle_seed` there.

2. Optionally prepare a pinned dataset snapshot:

```bash
python scripts/prepare_dataset.py runs/<model_name>/<run_name>
```

If `dataset.jsonl` exists, generation uses it. If not, generation loads and
normalizes the dataset directly from the `dataset:` config block.

### Main Generation

1. Parse prompts through `prompt:` config.
   `prompt.mode: plain` joins system, instruction, and question text. `chat`
   uses the tokenizer chat template when the model provides one.

2. Run generation and activation capture:

```bash
python scripts/generate.py runs/<model_name>/<run_name>
```

Outputs use a compact normalized schema: `metadata.json` stores run-level
metadata, `samples/<sample>.json` stores prompt/input/gold fields once, and
`generations.jsonl` stores per-generation fields and drives resume checks.
Reconstruct full tokens with `input_ids + generated_token_ids`. Captured activations
live under `generation/hidden_states/` and are referenced by `hidden_states_file`.

### Activations capture

Set `capture.layers: [lt1, lt2, ... ltn]` to capture activations of target layers `lt1`, `lt2`, etc. Set `capture.layers: [-1]` for the final layer only. Set `capture.layers: []`
to skip activation capture; generation still runs and `hidden_states_file` stays
`null`.

1. For remote runs, push the run folder, generate remotely, then pull results:

```bash
bash scripts/remote.sh push runs/<model_name>/<run_name>
bash scripts/remote.sh pull runs/<model_name>/<run_name>
```

Post-generation analysis is separate from this generation/capture workflow.
Read [ARCHITECTURE.md](ARCHITECTURE.md) before changing shared schemas.
