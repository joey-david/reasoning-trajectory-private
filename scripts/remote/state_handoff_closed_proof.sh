#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODE_LIST="${STATE_HANDOFF_CLOSED_NODES:-coktailjet,upnquick}"
DEVICE_LIST="${STATE_HANDOFF_CLOSED_DEVICES:-1;0}"
IFS=', ' read -r -a NODES <<< "$NODE_LIST"
IFS=';' read -r -a DEVICES <<< "$DEVICE_LIST"

TRAINING_ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed1"
CHALLENGE_ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_confirmation"
SESSION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_ROOT="runs/_confirmation/state_handoff_closed_proof/$SESSION"

RUNS=(
  "$TRAINING_ANCHOR"
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_outcome_seed1
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_outcome_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed3
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_outcome_seed3
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_width5
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_width5_outcome
)

if ((${#NODES[@]} != ${#DEVICES[@]})); then
  echo "STATE_HANDOFF_CLOSED_NODES and STATE_HANDOFF_CLOSED_DEVICES must align" >&2
  exit 2
fi

if [[ "${STATE_HANDOFF_CLOSED_DRY_RUN:-false}" == true ]]; then
  echo "closed proof-state confirmation dry run"
  echo "workers: ${NODES[*]} / ${DEVICES[*]}"
  echo "training tasks: 11"
  echo "challenge profiles: 23"
  echo "challenge cases across interface profiles: 2248"
  echo "the faster upnquick:0 worker takes another task whenever it finishes"
  printf 'run: %s\n' "${RUNS[@]}"
  exit 0
fi

mkdir -p "$LOG_ROOT"
STATUS="$LOG_ROOT/status.jsonl"

record() {
  local phase=$1
  local attempt=$2
  local status=$3
  local exit_code=$4
  printf \
    '{"phase":"%s","attempt":%d,"status":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$phase" "$attempt" "$status" "$exit_code" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$STATUS"
}

run_phase() {
  local phase=$1
  shift
  local attempt exit_code
  for attempt in 1 2; do
    echo
    echo "[$phase] attempt $attempt/2"
    "$@" 2>&1 | tee -a "$LOG_ROOT/$phase.log"
    exit_code=${PIPESTATUS[0]}
    if ((exit_code == 0)); then
      record "$phase" "$attempt" complete 0
      return 0
    fi
    record "$phase" "$attempt" failed "$exit_code"
  done
  echo "$phase failed twice; continuing with every independent phase" >&2
  return 1
}

prepare_run() {
  local run=$1
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    check-gate "$run" &&
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run" &&
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
      validate-data "$run"
}

echo "closed proof-state confirmation"
echo "workers: ${NODES[*]} / ${DEVICES[*]}"
echo "logs: $LOG_ROOT"

for run in "${RUNS[@]}"; do
  run_phase "prepare-$(basename "$run")" prepare_run "$run" || true
done

run_phase \
  prepare-challenges \
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  prepare-challenges "$CHALLENGE_ANCHOR" || true

run_phase \
  training \
  "$PYTHON" scripts/orchestrate.py \
  --job state_handoff_training \
  --nodes "${NODES[@]}" \
  --devices "${DEVICES[@]}" \
  --run "$TRAINING_ANCHOR" || true

run_phase \
  challenges \
  "$PYTHON" scripts/orchestrate.py \
  --job state_interface_challenge \
  --nodes "${NODES[@]}" \
  --devices "${DEVICES[@]}" \
  --run "$CHALLENGE_ANCHOR" || true

for run in \
  "$TRAINING_ANCHOR" \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed2 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed3 \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_width5
do
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$run" \
    >"$LOG_ROOT/$(basename "$run")-interfaces.json" 2>&1 || true
done

run_phase \
  reduce \
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  compare-closed-proof "$CHALLENGE_ANCHOR" || true

echo
echo "closed proof-state suite finished"
echo "status: $STATUS"
echo "summary: $CHALLENGE_ANCHOR/evaluation/closed_proof_confirmation_summary.json"
