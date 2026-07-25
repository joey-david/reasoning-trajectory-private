#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODE_LIST="${STATE_HANDOFF_PROOF_NODES:-ourasi}"
DEVICE_LIST="${STATE_HANDOFF_PROOF_DEVICES:-0,1}"
IFS=', ' read -r -a NODES <<< "$NODE_LIST"
IFS=', ' read -r -a DEVICES <<< "$DEVICE_LIST"
ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_weekend_confirmation"
SESSION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_ROOT="runs/_confirmation/state_handoff_proof_weekend/$SESSION"

TRAINING_RUNS=(
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed2
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_proof
  runs/Qwen2.5-3B-Instruct/interventions/state_interface_horn_proof_weekend
  runs/Qwen2.5-14B-Instruct/interventions/state_interface_horn_proof_weekend
)
PREPARE_RUNS=(
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed3
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed3
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_proof
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_outcome
  runs/Qwen2.5-3B-Instruct/interventions/state_interface_horn_proof_weekend
  runs/Qwen2.5-3B-Instruct/interventions/state_interface_horn_outcome_weekend
  runs/Qwen2.5-14B-Instruct/interventions/state_interface_horn_proof_weekend
  runs/Qwen2.5-14B-Instruct/interventions/state_interface_horn_outcome_weekend
)

mkdir -p "$LOG_ROOT"
STATUS="$LOG_ROOT/status.jsonl"

record() {
  local stage=$1
  local attempt=$2
  local status=$3
  local exit_code=$4
  printf \
    '{"stage":"%s","attempt":%d,"status":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$stage" "$attempt" "$status" "$exit_code" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$STATUS"
}

run_logged() {
  local stage=$1
  shift
  local attempt exit_code
  for attempt in 1 2; do
    echo
    echo "$stage (attempt $attempt/2)"
    "$@" 2>&1 | tee -a "$LOG_ROOT/${stage}.log"
    exit_code=${PIPESTATUS[0]}
    if ((exit_code == 0)); then
      record "$stage" "$attempt" complete 0
      return 0
    fi
    record "$stage" "$attempt" failed "$exit_code"
  done
  echo "$stage failed twice; continuing"
  return 1
}

prepare_run() {
  local run=$1
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run" &&
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
      validate-data "$run"
}

train_run() {
  local run=$1
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$run"
}

echo "proof-state weekend confirmation"
echo "nodes: ${NODES[*]}"
echo "devices: ${DEVICES[*]}"
echo "logs: $LOG_ROOT"
echo "order: Qwen 7B seeds, Mistral 7B, Qwen 3B, Qwen 14B"

for run in "${PREPARE_RUNS[@]}"; do
  name="$(basename "$run")-$(basename "$(dirname "$(dirname "$run")")")"
  run_logged "prepare-$name" prepare_run "$run" || continue
done

for run in "${TRAINING_RUNS[@]}"; do
  name="$(basename "$run")-$(basename "$(dirname "$(dirname "$run")")")"
  run_logged "train-$name" train_run "$run" || true
done

run_logged \
  prepare-balanced-challenges \
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  prepare-challenges "$ANCHOR" || true

run_logged \
  evaluate-balanced-challenges \
  "$PYTHON" scripts/orchestrate.py \
  --job state_interface_challenge \
  --nodes "${NODES[@]}" \
  --devices "${DEVICES[@]}" \
  --run "$ANCHOR" || true

run_logged \
  reduce-proof-confirmation \
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  compare-proof-confirmation "$ANCHOR" || true

echo
echo "weekend suite finished"
echo "status: $STATUS"
echo "summary: $ANCHOR/evaluation/proof_confirmation_summary.json"
