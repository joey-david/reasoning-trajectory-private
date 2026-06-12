# Minimal Research Architecture

## Mental Model

```text
dataset jsonl -> config.yaml -> generate.py -> generations.jsonl + activations
```

V1 needs only four jobs:

1. Read a run config from `runs/<model>/<experiment>/config.yaml`.
2. Load a small dataset from JSONL according to the parsed config.
3. Generate text for each selected prompt (several times, or in a specific
   manner according to the config).
4. Optionally save hidden states for selected layers (again, config).

5. On top of this, it should contain a simple script to push configs and their
   empty run folder to the inference server, and pull generations and activations
   when the run is done.

6. Finally, implement at least one tool (PCA visualization of activations is good) to affirm the modular design and pattern-making of tools.

## Repo Layout

```text
scripts/
  generate.py       # CLI: run one experiment folder.
  dataset.py        # CLI: make small JSONL subsets by hand.
  remote.sh         # CLI: push or pull one run folder.

src/
  config.py         # Load YAML and resolve paths.
  data.py           # Read/write JSONL and select examples.
  generation.py     # Hugging Face generation sketch.
  activations.py    # Hidden-state saving sketch.
  records.py        # Output record shapes.

web/
  index.html        # Static notes page, no server.
```

## Run Config Shape

Keep configs explicit and readable:

```yaml
model_name: Qwen/Qwen3-14B
dataset_path: datasets/gpqa/gpqa_diamond.jsonl
sample_limit: 3
seeds: [0]
temperatures: [0.7]
max_new_tokens: 256

activations:
  layers: [-1]
  precision: float16
  save_every_token: true
```

Older configs may still use `layers:` and `activation_storage_dtype:` at the top
level. When you implement this by hand, either support both forms or migrate old
configs one by one.

## Output Shape

Prefer plain files:

```text
generation/generations.jsonl
generation/activations/<record_id>.npz
generation/logs/
```

Each JSONL row should be easy to inspect:

```json
{
  "sample_id": "gpqa_diamond_0",
  "seed": 0,
  "temperature": 0.7,
  "prompt": "...",
  "text": "...",
  "token_ids": [1, 2, 3],
  "activation_file": "generation/activations/gpqa_diamond_0_seed0_temp0.7.npz"
}
```

## Remote Flow

Remote execution should not be an orchestration framework. Keep it as file sync:

```bash
bash scripts/remote.sh push runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
ssh lamgate
cd /home/lamsade/jdavid/research/reasoning
source .venv/bin/activate
python scripts/generate.py runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
exit
bash scripts/remote.sh pull runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
```

That is slower than automation, but it teaches you where the files are and what
failed.

## What To Implement First

1. `src/config.py`: load YAML with `yaml.safe_load`.
2. `src/data.py`: read JSONL with `json.loads` line by line.
3. `src/generation.py`: load `AutoTokenizer` and `AutoModelForCausalLM`.
4. `scripts/generate.py`: loop over samples, seeds, and temperatures.
5. `src/activations.py`: call the model with `output_hidden_states=True`.

Do not build analysis tools until generation outputs are boring and reliable.
