#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
QWEN_NODE="${CAUSAL_QWEN_NODE:-ourasi}"
QWEN_DEVICES="${CAUSAL_QWEN_DEVICES:-0,1}"
MISTRAL_NODE="${CAUSAL_MISTRAL_NODE:-coktailjet}"
MISTRAL_DEVICES="${CAUSAL_MISTRAL_DEVICES:-0,1}"
QWEN_SUITE="runs/Qwen2.5-7B-Instruct/interventions/causal_reasoning_suite"
MISTRAL_SUITE="runs/Mistral-7B-Instruct-v0.3/interventions/causal_reasoning_suite"
SESSION="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_ROOT="runs/_causal_reasoning/$SESSION"

prepare_suite() {
  local label=$1
  local suite=$2
  "$PYTHON" scripts/experiments/causal_reasoning.py prepare "$suite" \
    >"$LOG_ROOT/$label-prepare.json"
  "$PYTHON" scripts/experiments/causal_reasoning.py validate-tokens "$suite" \
    >"$LOG_ROOT/$label-tokens.json"
  echo "$label data and token contracts passed"
}

run_suite() {
  local label=$1
  local suite=$2
  local node=$3
  local devices=$4
  local attempt exit_code
  for attempt in 1 2; do
    echo "[$label] inference attempt $attempt/2"
    "$PYTHON" scripts/orchestrate.py \
      --job causal_reasoning \
      --nodes "$node" \
      --devices "$devices" \
      --run "$suite" \
      2>&1 | tee -a "$LOG_ROOT/$label.log"
    exit_code=${PIPESTATUS[0]}
    if ((exit_code == 0)); then
      break
    fi
  done
  "$PYTHON" scripts/experiments/causal_reasoning.py reduce "$suite" \
    >"$LOG_ROOT/$label-summary.json" 2>&1 || true
  "$PYTHON" scripts/experiments/causal_reasoning.py status "$suite" \
    >"$LOG_ROOT/$label-status.json" 2>&1 || true
  return "$exit_code"
}

mkdir -p "$LOG_ROOT"
echo "causal reasoning suite"
echo "Qwen: $QWEN_NODE GPUs $QWEN_DEVICES"
echo "Mistral: $MISTRAL_NODE GPUs $MISTRAL_DEVICES"
echo "logs: $LOG_ROOT"

prepare_suite qwen "$QWEN_SUITE"
prepare_suite mistral "$MISTRAL_SUITE"

if [[ "${CAUSAL_REASONING_DRY_RUN:-false}" == true ]]; then
  "$PYTHON" scripts/experiments/causal_reasoning.py status "$QWEN_SUITE"
  "$PYTHON" scripts/experiments/causal_reasoning.py status "$MISTRAL_SUITE"
  exit 0
fi

set +e
run_suite qwen "$QWEN_SUITE" "$QWEN_NODE" "$QWEN_DEVICES" &
qwen_pid=$!
run_suite mistral "$MISTRAL_SUITE" "$MISTRAL_NODE" "$MISTRAL_DEVICES" &
mistral_pid=$!

qwen_exit=0
mistral_exit=0
wait "$qwen_pid" || qwen_exit=$?
wait "$mistral_pid" || mistral_exit=$?
set -e

echo "Qwen exit: $qwen_exit"
echo "Mistral exit: $mistral_exit"
echo "Pull the six child run folders for each completed model suite."
((qwen_exit == 0 && mistral_exit == 0))
