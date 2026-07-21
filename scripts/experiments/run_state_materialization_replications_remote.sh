#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${STATE_REPLICATION_NODES:-upnquick}"
DEVICES="${STATE_REPLICATION_DEVICES:-0+1}"

FACTORIZATION_RUNS=(
  "runs/Qwen2.5-32B-Instruct/interventions/state_materialization_symbolic"
  "runs/Mistral-Small-24B-Instruct-2501/interventions/state_materialization_factorization"
)
ROUTING_RUNS=(
  "runs/Qwen2.5-32B-Instruct/interventions/state_routing_symbolic_confirmation"
  "runs/Mistral-Small-24B-Instruct-2501/interventions/state_routing_confirmation"
)

for run in "${FACTORIZATION_RUNS[@]}"; do
  echo "=== State-materialization replication: $run ==="
  "$PYTHON" scripts/experiments/depth_relief.py prepare-factorization "$run"
  "$PYTHON" scripts/experiments/depth_relief.py validate-factorization "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_materialization \
    --nodes "$NODES" \
    --devices "$DEVICES" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py analyze-factorization "$run"
done

REFERENCE="runs/Qwen2.5-32B-Instruct/interventions/state_materialization_factorization"
MISTRAL="${FACTORIZATION_RUNS[1]}"
if [[ -f "$REFERENCE/depth_relief/factorization_cases.jsonl" ]]; then
  "$PYTHON" scripts/experiments/depth_relief.py compare-factorization \
    "$REFERENCE" "$MISTRAL" \
    --output "$MISTRAL/depth_relief/cross_family_comparison.json"
fi

for index in "${!FACTORIZATION_RUNS[@]}"; do
  source_run="${FACTORIZATION_RUNS[$index]}"
  routing_run="${ROUTING_RUNS[$index]}"
  if "$PYTHON" scripts/experiments/depth_relief.py \
    routing-eligibility "$source_run"; then
    echo "=== Matched state routing replication: $routing_run ==="
    "$PYTHON" scripts/experiments/depth_relief.py prepare-routing "$routing_run"
    "$PYTHON" scripts/experiments/depth_relief.py validate-routing "$routing_run"
    "$PYTHON" scripts/orchestrate.py \
      --job state_routing \
      --nodes "$NODES" \
      --devices "$DEVICES" \
      --run "$routing_run"
    "$PYTHON" scripts/experiments/depth_relief.py analyze-routing "$routing_run"
  else
    status=$?
    if [[ $status -ne 3 ]]; then
      exit "$status"
    fi
    echo "Routing confirmation skipped: the factorization gate did not pass."
  fi
done

echo "All admissible state-materialization replications completed."
