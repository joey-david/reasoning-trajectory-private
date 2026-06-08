# Extract

Purpose: generate completions and save token/step hidden states in the shared schema.
Inputs: YAML config with model, prompts, seeds, temperatures, layers, and pooling.
Outputs: `trajectories.jsonl`.
CLI: `rt extract --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_distill_sheep30`
Python API: `reasoning_trajectory.extract.generations.extract_from_config`.
Example: `python -c "from reasoning_trajectory.extract.generations import extract_from_config; extract_from_config('experiments/configs/r1_distill_sheep30.yaml','experiments/runs/r1_distill_sheep30')"`
Notes: `model_name: mock` is deterministic; any HuggingFace causal LM id uses real generation and hidden-state extraction.
Per-prompt `expected_answer` values are used to infer `final_answer` and `final_correct` from the generated final numeric answer.
Failure modes: missing `torch`/`transformers` for HuggingFace models; unavailable model weights.
