# HF Check

Purpose: run a minimal HuggingFace generation and hidden-state extraction check.
Inputs: config with `model_name`, optional `hf_token_env`, cache/device settings, and prompt.
Outputs: JSON summary with generated preview, token count, hidden-layer count, shape, device, and elapsed time.
CLI: `rt extract hf-check --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_hf_check.json`
Python API: `reasoning_trajectory.extract.hf_check.hf_inference_check`.
Example: `HF_TOKEN=... rt extract hf-check --config experiments/configs/r1_distill_sheep30.yaml --out experiments/runs/r1_hf_check.json`
Notes: use `use_safetensors: false` if a remote HF/Xet path stalls on tiny models.
Failure modes: missing `torch`/`transformers`, invalid token, unavailable weights, or insufficient device memory.
