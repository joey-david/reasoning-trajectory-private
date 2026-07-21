#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${DEPTH_RELIEF_NODES:-upnquick}"
DEVICES="${DEPTH_RELIEF_DEVICES:-0,1}"

RUNS=(
  "runs/Qwen3-8B/interventions/depth_relief_main"
  "runs/Qwen2.5-7B-Instruct/interventions/depth_relief_base"
  "runs/DeepSeek-R1-Distill-Qwen-7B/interventions/depth_relief_reasoning"
  "runs/SmolLM3-3B/interventions/depth_relief_main"
)

for run in "${RUNS[@]}"; do
  echo "=== Depth relief: $run ==="
  "$PYTHON" scripts/orchestrate.py \
    --job depth_relief \
    --nodes "$NODES" \
    --devices "$DEVICES" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py analyze "$run"
done

"$PYTHON" scripts/experiments/depth_relief.py compare \
  "runs/Qwen2.5-7B-Instruct/interventions/depth_relief_base" \
  "runs/DeepSeek-R1-Distill-Qwen-7B/interventions/depth_relief_reasoning" \
  --output \
  "runs/DeepSeek-R1-Distill-Qwen-7B/interventions/depth_relief_reasoning/depth_relief/base_comparison.json"

echo "All depth-relief runs and analyses completed."
