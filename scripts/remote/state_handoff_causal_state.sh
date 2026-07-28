#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODE_LIST="${STATE_CAUSAL_NODES:-local}"
DEVICE_LIST="${STATE_CAUSAL_DEVICES:-0,1}"
IFS=', ' read -r -a NODES <<< "$NODE_LIST"
IFS=';' read -r -a DEVICES <<< "$DEVICE_LIST"

TRAINING_ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed1"
CHALLENGE_ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_causal_state_phase"
SESSION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_ROOT="runs/_confirmation/state_handoff_causal_state/$SESSION"
TRAINING_RUNS=(
  "$TRAINING_ANCHOR"
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_closure_seed3
)

if ((${#NODES[@]} != ${#DEVICES[@]})); then
  echo "STATE_CAUSAL_NODES and STATE_CAUSAL_DEVICES must align" >&2
  exit 2
fi

if [[ "${STATE_CAUSAL_DRY_RUN:-false}" == true ]]; then
  echo "causal-state confirmation dry run"
  echo "workers: ${NODES[*]} / ${DEVICES[*]}"
  echo "new training tasks: 5"
  echo "challenge profiles: 30"
  echo "interface challenge cases: 2840"
  echo "interface transition calls: 171520"
  echo "target runtime on two 48GB or larger GPUs: 9-12 hours"
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
  echo "$phase failed twice; later independent phases will still run" >&2
  return 1
}

prepare_training_run() {
  local run=$1
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    check-gate "$run" &&
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run" &&
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    validate-data "$run"
}

echo "causal-state confirmation"
echo "workers: ${NODES[*]} / ${DEVICES[*]}"
echo "logs: $LOG_ROOT"

for run in "${TRAINING_RUNS[@]}"; do
  run_phase "prepare-$(basename "$run")" prepare_training_run "$run" || true
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

training_ready=0
"$PYTHON" scripts/experiments/run_state_handoff_training.py \
  check-linked-training "$TRAINING_ANCHOR" \
  2>&1 | tee -a "$LOG_ROOT/training-completeness.log"
training_check_exit=${PIPESTATUS[0]}
if ((training_check_exit == 0)); then
  training_ready=1
  record training-completeness 1 complete 0
else
  record training-completeness 1 incomplete "$training_check_exit"
  echo "challenge inference skipped until all linked adapters finish" >&2
fi

if ((training_ready)); then
  run_phase \
    challenges \
    "$PYTHON" scripts/orchestrate.py \
    --job state_interface_challenge \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$CHALLENGE_ANCHOR" || true
fi

run_phase \
  reduce \
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  compare-causal-state "$CHALLENGE_ANCHOR" || true

echo
echo "causal-state suite finished"
echo "status: $STATUS"
echo "summary: $CHALLENGE_ANCHOR/evaluation/causal_state_phase_summary.json"
