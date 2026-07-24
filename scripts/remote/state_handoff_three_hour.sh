#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_sweep_3h"
NODES=(${STATE_HANDOFF_3H_NODES:-ourasi seacove coktailjet})
DEVICES=(${STATE_HANDOFF_3H_DEVICES:-0,1 3 0,1})
LOG_ROOT="${STATE_HANDOFF_3H_LOG_DIR:-runs/_three_hour/state_handoff/$(date -u +%Y%m%dT%H%M%SZ)-$$}"

RUNS=(
  "$ANCHOR"
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_sweep_outcome_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_sweep_donor_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_primitives_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_primitives_outcome_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_actions_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_actions_outcome_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_machine_3h
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_machine_outcome_3h
)

if ((${#NODES[@]} != ${#DEVICES[@]})); then
  echo "STATE_HANDOFF_3H_NODES and STATE_HANDOFF_3H_DEVICES must have one entry per host" >&2
  exit 2
fi

if [[ "${STATE_HANDOFF_3H_DRY_RUN:-false}" == true ]]; then
  echo "state-interface decisive suite dry run"
  echo "workers: ${NODES[*]} / ${DEVICES[*]}"
  echo "training tasks: 12"
  echo "length-extrapolation tasks: 4"
  echo "cross-adapter substitution tasks: 1"
  printf 'run: %s\n' "${RUNS[@]}"
  exit 0
fi

mkdir -p "$LOG_ROOT"
STATUS="$LOG_ROOT/status.jsonl"

record() {
  local phase=$1
  local status=$2
  local exit_code=$3
  printf '{"phase":"%s","status":"%s","exit_code":%d,"finished_at":"%s"}\n' \
    "$phase" "$status" "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$STATUS"
}

run_phase() {
  local phase=$1
  shift
  echo
  echo "[$phase]"
  if "$@" 2>&1 | tee "$LOG_ROOT/$phase.log"; then
    record "$phase" complete 0
    return 0
  fi
  local code=${PIPESTATUS[0]}
  record "$phase" failed "$code"
  echo "$phase failed with exit $code; continuing with every runnable later phase" >&2
  return 0
}

echo "state-interface decisive suite"
echo "workers: ${NODES[*]} / ${DEVICES[*]}"
echo "logs: $LOG_ROOT"

prepare_failures=0
for run in "${RUNS[@]}"; do
  name="$(basename "$run")"
  if ! "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run" >"$LOG_ROOT/prepare-$name.json" 2>&1
  then
    echo "data preparation failed for $run" >&2
    ((prepare_failures += 1))
    continue
  fi
  if ! "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    validate-data "$run" >"$LOG_ROOT/validate-$name.json" 2>&1
  then
    echo "data validation failed for $run" >&2
    ((prepare_failures += 1))
  fi
done
if ! "$PYTHON" scripts/experiments/run_state_handoff_training.py \
  prepare-challenges "$ANCHOR" >"$LOG_ROOT/challenge-manifest.json" 2>&1
then
  ((prepare_failures += 1))
fi
if ((prepare_failures)); then
  record prepare partial "$prepare_failures"
else
  record prepare complete 0
fi

run_phase training \
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$ANCHOR"

run_phase long-horizon \
  "$PYTHON" scripts/orchestrate.py \
    --job state_interface_challenge \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$ANCHOR"

run_phase substitution \
  "$PYTHON" scripts/orchestrate.py \
    --job state_interface_substitution \
    --nodes "${NODES[0]}" \
    --devices 0 \
    --run "$ANCHOR"

for run in \
  "$ANCHOR" \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_primitives_3h \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_actions_3h \
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_machine_3h
do
  name="$(basename "$run")"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$run" >"$LOG_ROOT/$name-interfaces.json" 2>&1 || true
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-generalization "$run" >"$LOG_ROOT/$name-generalization.json" 2>&1 || true
done

echo
echo "suite finished; inspect $STATUS and pull the nine run paths listed in this script"
