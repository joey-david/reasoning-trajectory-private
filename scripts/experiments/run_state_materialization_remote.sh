#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${STATE_MATERIALIZATION_NODES:-upnquick}"
SMALL_DEVICES="${STATE_MATERIALIZATION_SMALL_DEVICES:-0,1}"
LARGE_DEVICES="${STATE_MATERIALIZATION_LARGE_DEVICES:-0+1}"

RUNS=(
  "runs/Qwen2.5-32B-Instruct/interventions/state_materialization_factorization"
  "runs/Qwen3-8B/interventions/state_materialization_factorization"
  "runs/Qwen3-32B/interventions/state_materialization_factorization"
)
DEVICES=("$LARGE_DEVICES" "$SMALL_DEVICES" "$LARGE_DEVICES")

for index in "${!RUNS[@]}"; do
  run="${RUNS[$index]}"
  echo "=== State-materialization factorization: $run ==="
  "$PYTHON" scripts/orchestrate.py \
    --job state_materialization \
    --nodes "$NODES" \
    --devices "${DEVICES[$index]}" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-factorization "$run"
done

"$PYTHON" scripts/experiments/depth_relief.py compare-factorization \
  "${RUNS[1]}" "${RUNS[2]}" "${RUNS[0]}" \
  --output "${RUNS[0]}/depth_relief/factorization_model_comparison.json"

echo "All state-materialization factorization runs completed."
