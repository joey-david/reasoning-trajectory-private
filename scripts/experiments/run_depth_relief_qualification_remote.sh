#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${DEPTH_RELIEF_NODES:-upnquick}"
SMALL_DEVICES="${DEPTH_RELIEF_SMALL_DEVICES:-0,1}"
LARGE_DEVICES="${DEPTH_RELIEF_LARGE_DEVICES:-0+1}"

RUNS=(
  "runs/Qwen3-8B/interventions/depth_relief_qualification"
  "runs/Qwen3-32B/interventions/depth_relief_qualification"
  "runs/Qwen2.5-32B-Instruct/interventions/depth_relief_qualification"
)
DEVICES=("$SMALL_DEVICES" "$LARGE_DEVICES" "$LARGE_DEVICES")

for index in "${!RUNS[@]}"; do
  run="${RUNS[$index]}"
  echo "=== Depth-relief qualification: $run ==="
  "$PYTHON" scripts/orchestrate.py \
    --job depth_relief_qualification \
    --nodes "$NODES" \
    --devices "${DEVICES[$index]}" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-qualification "$run"
done

echo "All depth-relief qualification runs completed."
