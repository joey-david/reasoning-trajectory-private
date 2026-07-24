#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
WORKERS=(${STATE_HANDOFF_POLISH_WORKERS:-local:0 local:1})
LOG_ROOT="${STATE_HANDOFF_POLISH_LOG_DIR:-runs/_confirmation/state_handoff_polish/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
PROOF_RUN="runs/Qwen2.5-7B-Instruct/interventions/state_interface_proof_depth_fullrate"
REGISTER_RUNS=(
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_redundant5_seed1
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_redundant5_seed2
  runs/Qwen2.5-7B-Instruct/interventions/state_interface_register_redundant5_seed3
)

if ((${#WORKERS[@]} == 0)); then
  echo "STATE_HANDOFF_POLISH_WORKERS must name at least one host:gpu worker" >&2
  exit 2
fi
for worker in "${WORKERS[@]}"; do
  if [[ ! $worker =~ ^[^:]+:[0-9]+$ ]]; then
    echo "invalid worker '$worker'; expected host:gpu" >&2
    exit 2
  fi
done

run_count=${#WORKERS[@]}
if ((run_count > ${#REGISTER_RUNS[@]})); then
  run_count=${#REGISTER_RUNS[@]}
fi
SELECTED_RUNS=("${REGISTER_RUNS[@]:0:run_count}")

if [[ "${STATE_HANDOFF_POLISH_DRY_RUN:-false}" == true ]]; then
  echo "state-handoff closure polish dry run"
  echo "workers: ${WORKERS[*]}"
  echo "parallel register seeds: $run_count"
  printf 'register run: %s\n' "${SELECTED_RUNS[@]}"
  echo "proof run: $PROOF_RUN"
  if ((${#WORKERS[@]} > run_count)); then
    echo "concurrent proof workers: ${WORKERS[*]:run_count}"
  else
    echo "proof workers after training: ${WORKERS[*]}"
  fi
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

echo "state-handoff closure polish"
echo "workers: ${WORKERS[*]}"
echo "parallel register seeds: $run_count"
echo "logs: $LOG_ROOT"

prepare_failures=0
for run in "${SELECTED_RUNS[@]}"; do
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
  prepare-challenges "$PROOF_RUN" >"$LOG_ROOT/proof-manifest.json" 2>&1
then
  ((prepare_failures += 1))
fi
if ((prepare_failures)); then
  record prepare failed "$prepare_failures"
  exit 1
fi
record prepare complete 0

pids=()
for index in "${!SELECTED_RUNS[@]}"; do
  run="${SELECTED_RUNS[$index]}"
  worker="${WORKERS[$index]}"
  node="${worker%:*}"
  gpu="${worker##*:}"
  name="$(basename "$run")"
  (
    set -o pipefail
    "$PYTHON" scripts/orchestrate.py \
      --job state_handoff_training \
      --nodes "$node" \
      --devices "$gpu" \
      --run "$run" 2>&1 \
      | sed -u "s/^/[$name] /" \
      | tee "$LOG_ROOT/train-$name.log"
  ) &
  pids+=("$!")
  echo "started $name on $worker"
done

proof_pid=""
if ((${#WORKERS[@]} > run_count)); then
  proof_nodes=()
  proof_devices=()
  for worker in "${WORKERS[@]:run_count}"; do
    proof_nodes+=("${worker%:*}")
    proof_devices+=("${worker##*:}")
  done
  (
    set -o pipefail
    "$PYTHON" scripts/orchestrate.py \
      --job state_interface_challenge \
      --nodes "${proof_nodes[@]}" \
      --devices "${proof_devices[@]}" \
      --run "$PROOF_RUN" 2>&1 \
      | tee "$LOG_ROOT/proof-depth.log"
  ) &
  proof_pid=$!
  echo "started proof depth on ${WORKERS[*]:run_count}"
fi

failed_indices=()
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    record "train-$(basename "${SELECTED_RUNS[$index]}")" complete 0
  else
    code=$?
    record "train-$(basename "${SELECTED_RUNS[$index]}")" failed "$code"
    failed_indices+=("$index")
  fi
done

for index in "${failed_indices[@]}"; do
  run="${SELECTED_RUNS[$index]}"
  worker="${WORKERS[$index]}"
  node="${worker%:*}"
  gpu="${worker##*:}"
  name="$(basename "$run")"
  echo "retrying unfinished $name on $worker"
  if "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$node" \
    --devices "$gpu" \
    --run "$run" >>"$LOG_ROOT/train-$name.log" 2>&1
  then
    record "retry-$name" complete 0
  else
    code=$?
    record "retry-$name" failed "$code"
  fi
done

for run in "${SELECTED_RUNS[@]}"; do
  name="$(basename "$run")"
  if "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-generalization "$run" >"$LOG_ROOT/generalization-$name.json" 2>&1
  then
    record "analyze-$name" complete 0
  else
    code=$?
    record "analyze-$name" failed "$code"
  fi
done

if [[ -n $proof_pid ]]; then
  if wait "$proof_pid"; then
    record proof-depth complete 0
  else
    code=$?
    record proof-depth failed "$code"
  fi
else
  nodes=()
  devices=()
  for worker in "${WORKERS[@]}"; do
    nodes+=("${worker%:*}")
    devices+=("${worker##*:}")
  done
  if "$PYTHON" scripts/orchestrate.py \
    --job state_interface_challenge \
    --nodes "${nodes[@]}" \
    --devices "${devices[@]}" \
    --run "$PROOF_RUN" 2>&1 | tee "$LOG_ROOT/proof-depth.log"
  then
    record proof-depth complete 0
  else
    code=${PIPESTATUS[0]}
    record proof-depth failed "$code"
  fi
fi

echo
echo "closure polish finished"
echo "status: $STATUS"
printf 'pull: %s\n' "${SELECTED_RUNS[@]}" "$PROOF_RUN"
