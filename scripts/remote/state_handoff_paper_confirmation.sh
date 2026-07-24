#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
ANCHOR="runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed1"
NODES=(${STATE_HANDOFF_CONFIRM_NODES:-ourasi kaisertrot coktailjet})
DEVICES=(${STATE_HANDOFF_CONFIRM_DEVICES:-0,1 1 0,1})
LOG_ROOT="${STATE_HANDOFF_CONFIRM_LOG_DIR:-runs/_confirmation/state_handoff/$(date -u +%Y%m%dT%H%M%SZ)-$$}"

RUNS=(
  "$ANCHOR"
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_outcome_seed1
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_outcome_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed3
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_outcome_seed3
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_register_confirm
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_register_confirm_outcome
)

INTERFACE_RUNS=(
  "$ANCHOR"
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_confirm_seed3
  runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_register_confirm
)

if ((${#NODES[@]} != ${#DEVICES[@]})); then
  echo "STATE_HANDOFF_CONFIRM_NODES and STATE_HANDOFF_CONFIRM_DEVICES must align" >&2
  exit 2
fi

if [[ "${STATE_HANDOFF_CONFIRM_DRY_RUN:-false}" == true ]]; then
  echo "state-handoff paper confirmation dry run"
  echo "workers: ${NODES[*]} / ${DEVICES[*]}"
  echo "training tasks: 8"
  echo "Qwen seeds: 3 with disjoint 10-context test banks"
  echo "second model: Mistral-7B-Instruct-v0.3"
  echo "proof-depth challenge: h64 with 0,1,2,3,4 active deductions"
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
  echo "$phase failed with exit $code; later phases will still run" >&2
  return "$code"
}

echo "state-handoff paper confirmation"
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
"$PYTHON" scripts/experiments/run_state_handoff_training.py \
  prepare-challenges "$ANCHOR" >"$LOG_ROOT/proof-depth-manifest.json" 2>&1 \
  || ((prepare_failures += 1))
if ((prepare_failures)); then
  record prepare partial "$prepare_failures"
else
  record prepare complete 0
fi

if ! run_phase training \
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$ANCHOR"
then
  echo "retrying only unfinished training/evaluation tasks" >&2
  run_phase training-retry \
    "$PYTHON" scripts/orchestrate.py \
      --job state_handoff_training \
      --nodes "${NODES[@]}" \
      --devices "${DEVICES[@]}" \
      --run "$ANCHOR" || true
fi

for run in "${INTERFACE_RUNS[@]}"; do
  name="$(basename "$run")"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$run" >"$LOG_ROOT/$name-interfaces.json" 2>&1 || true
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-generalization "$run" >"$LOG_ROOT/$name-generalization.json" 2>&1 || true
done
"$PYTHON" scripts/experiments/run_state_handoff_training.py \
  compare-replication "$ANCHOR" >"$LOG_ROOT/register-replication.json" 2>&1 || true

run_phase proof-depth \
  "$PYTHON" scripts/orchestrate.py \
    --job state_interface_challenge \
    --nodes "${NODES[@]}" \
    --devices "${DEVICES[@]}" \
    --run "$ANCHOR" || true

echo
echo "confirmation finished"
echo "status: $STATUS"
echo "pull the eight run paths printed by the dry run"
