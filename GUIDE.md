# First Implementation Guide

This guide is for learning the code by writing the important parts yourself.

## 1. Read A Config

Open [src/config.py](src/config.py). The key function is `load_config`.

Things to learn:

- `Path("some/file")`
- `with open(...) as handle:`
- `yaml.safe_load(handle)`

Check it with:

```bash
python - <<'PY'
from src.config import load_config
print(load_config("runs/Qwen3-14B/gpqa_diamond_x6_last48_int8")["model_name"])
PY
```

## 2. Read A Dataset

Open [src/data.py](src/data.py). The key functions are
`load_samples`, `select_samples`, and `prompt_from_sample`.

Things to learn:

- JSONL means one JSON object per line.
- `json.loads(line)` turns text into a Python dict.
- list slicing like `rows[10:20]`.

Make a tiny dataset:

```bash
python scripts/dataset.py subset datasets/gpqa/gpqa_diamond.jsonl datasets/gpqa/tiny.jsonl --limit 3
```

## 3. Generate Text

Open [src/generation.py](src/generation.py). Implement the
TODOs in the most direct possible way.

The Hugging Face pattern is:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=128)
text = tokenizer.decode(output[0], skip_special_tokens=True)
```

Do this before worrying about activations.

## 4. Save Activations

Open [src/activations.py](src/activations.py). Once generation works, call the
model directly with:

```python
outputs = model(**inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states
```

Then save the layers you care about with `numpy.savez_compressed`.

## 5. Remote Use

Remote work is just sync now:

```bash
bash scripts/remote.sh push runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
ssh lamgate
cd /home/lamsade/jdavid/research/reasoning
source .venv/bin/activate
python scripts/generate.py runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
exit
bash scripts/remote.sh pull runs/Qwen3-14B/gpqa_diamond_x6_last48_int8
```

This is intentionally manual so failures are visible.
