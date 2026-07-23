#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUNNER="${STATE_HANDOFF_OVERNIGHT_RUNNER:-$REPO_ROOT/scripts/remote/state_handoff.sh}"
ACTION_TIMEOUT="${STATE_HANDOFF_ACTION_TIMEOUT:-7h}"
DRY_RUN="${STATE_HANDOFF_OVERNIGHT_DRY_RUN:-false}"
SESSION_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_ROOT="${STATE_HANDOFF_OVERNIGHT_LOG_DIR:-runs/_overnight/state_handoff/$SESSION_ID}"

DEFAULT_ACTIONS=(
  interface-joint-closure-7b
  interface-algebra-transfer-7b
  interface-proof-transfer-7b
  interface-width4-transfer-7b
  interface-algebra-confirm-7b
  interface-proof-confirm-7b
  interface-proof-second-model-7b
)

action_run_paths() {
  case "$1" in
  interface-joint-closure-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_joint_closure"
    ;;
  interface-algebra-transfer-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome"
    ;;
  interface-proof-transfer-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome"
    ;;
  interface-width4-transfer-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_algebra"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_outcome"
    ;;
  interface-algebra-confirm-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed2"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed2"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed3"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed3"
    ;;
  interface-proof-confirm-7b)
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed2"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed2"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed3"
    echo "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed3"
    ;;
  interface-proof-second-model-7b)
    echo "runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_proof"
    echo "runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_outcome"
    ;;
  *)
    echo "unknown overnight action: $1" >&2
    return 2
    ;;
  esac
}

if (($#)); then
  ACTIONS=("$@")
else
  ACTIONS=("${DEFAULT_ACTIONS[@]}")
fi

for action in "${ACTIONS[@]}"; do
  action_run_paths "$action" >/dev/null || exit 2
done

if [[ "$DRY_RUN" == true ]]; then
  echo "state handoff overnight dry run"
  echo "per-action timeout: $ACTION_TIMEOUT"
  printf 'actions:\n'
  printf '  %s\n' "${ACTIONS[@]}"
  printf 'run paths:\n'
  for action in "${ACTIONS[@]}"; do
    action_run_paths "$action"
  done | awk '!seen[$0]++ {print "  " $0}'
  exit 0
fi

mkdir -p "$LOG_ROOT"
STATUS_PATH="$LOG_ROOT/status.jsonl"
RUN_PATHS_PATH="$LOG_ROOT/run_paths.txt"
CURRENT_ACTION=""
CURRENT_STARTED_AT=""
CURRENT_STARTED_EPOCH=0
CURRENT_LOG_NAME=""
CHILD_PID=""
CHILD_HAS_SESSION=false

for action in "${ACTIONS[@]}"; do
  action_run_paths "$action"
done | awk '!seen[$0]++' >"$RUN_PATHS_PATH"

if ! git_commit="$(git rev-parse HEAD 2>/dev/null)"; then
  git_commit=unavailable
fi

{
  echo "schema_version=1"
  echo "session_id=$SESSION_ID"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit=$git_commit"
  echo "host=$(hostname)"
  echo "nodes=${STATE_HANDOFF_NODES:-upnquick}"
  echo "devices=${STATE_HANDOFF_7B_DEVICES:-0,1}"
  echo "per_action_timeout=$ACTION_TIMEOUT"
} >"$LOG_ROOT/session.txt"

record_status() {
  local action=$1
  local status=$2
  local exit_code=$3
  local started_at=$4
  local finished_at=$5
  local duration_seconds=$6
  local log_name=$7
  printf \
    '{"schema_version":1,"action":"%s","status":"%s","exit_code":%d,"started_at":"%s","finished_at":"%s","duration_seconds":%d,"log":"%s"}\n' \
    "$action" "$status" "$exit_code" "$started_at" "$finished_at" \
    "$duration_seconds" "$log_name" >>"$STATUS_PATH"
}

stop_current_action() {
  local signal=$1
  trap - INT TERM HUP
  echo
  echo "overnight sweep interrupted by $signal during ${CURRENT_ACTION:-startup}"
  if [[ -n "$CHILD_PID" ]]; then
    if [[ "$CHILD_HAS_SESSION" == true ]]; then
      kill -TERM -- "-$CHILD_PID" 2>/dev/null || true
    else
      kill -TERM "$CHILD_PID" 2>/dev/null || true
    fi
    wait "$CHILD_PID" 2>/dev/null || true
  fi
  if [[ -n "$CURRENT_ACTION" && -n "$CURRENT_STARTED_AT" ]]; then
    local interrupted_at
    local duration_seconds
    interrupted_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    duration_seconds=$(( $(date +%s) - CURRENT_STARTED_EPOCH ))
    record_status \
      "$CURRENT_ACTION" interrupted 130 "$CURRENT_STARTED_AT" "$interrupted_at" \
      "$duration_seconds" "$CURRENT_LOG_NAME"
  fi
  printf '%s\n' "${CURRENT_ACTION:-startup}" >"$LOG_ROOT/interrupted_at.txt"
  exit 130
}

trap 'stop_current_action INT' INT
trap 'stop_current_action TERM' TERM
trap 'stop_current_action HUP' HUP

echo "state handoff overnight session: $SESSION_ID"
echo "logs: $LOG_ROOT"
echo "per-action timeout: $ACTION_TIMEOUT"

success_count=0
failure_count=0
for index in "${!ACTIONS[@]}"; do
  CURRENT_ACTION="${ACTIONS[$index]}"
  ordinal=$((index + 1))
  CURRENT_LOG_NAME="$(printf '%02d_%s.log' "$ordinal" "$CURRENT_ACTION")"
  log_path="$LOG_ROOT/$CURRENT_LOG_NAME"
  CURRENT_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  CURRENT_STARTED_EPOCH="$(date +%s)"

  {
    echo
    echo "[$ordinal/${#ACTIONS[@]}] $CURRENT_ACTION"
    echo "started: $CURRENT_STARTED_AT"
    echo "run paths:"
    action_run_paths "$CURRENT_ACTION" | sed 's/^/  /'
  } | tee -a "$log_path"

  command=(timeout --foreground "$ACTION_TIMEOUT" bash "$RUNNER" "$CURRENT_ACTION")
  if ! command -v timeout >/dev/null 2>&1; then
    echo "warning: timeout is unavailable; running without an action deadline" \
      | tee -a "$log_path"
    command=(bash "$RUNNER" "$CURRENT_ACTION")
  fi
  if command -v setsid >/dev/null 2>&1; then
    command=(setsid "${command[@]}")
    CHILD_HAS_SESSION=true
  else
    CHILD_HAS_SESSION=false
  fi

  "${command[@]}" > >(tee -a "$log_path") 2>&1 &
  CHILD_PID=$!
  wait "$CHILD_PID"
  exit_code=$?
  CHILD_PID=""
  CHILD_HAS_SESSION=false

  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  duration_seconds=$(( $(date +%s) - CURRENT_STARTED_EPOCH ))
  if ((exit_code == 0)); then
    status=success
    ((success_count += 1))
  elif ((exit_code == 124)); then
    status=timeout
    ((failure_count += 1))
  else
    status=failed
    ((failure_count += 1))
  fi
  record_status \
    "$CURRENT_ACTION" "$status" "$exit_code" "$CURRENT_STARTED_AT" "$finished_at" \
    "$duration_seconds" "$CURRENT_LOG_NAME"
  echo "finished: $finished_at status=$status exit=$exit_code" | tee -a "$log_path"
  CURRENT_ACTION=""
  CURRENT_STARTED_AT=""
  CURRENT_STARTED_EPOCH=0
  CURRENT_LOG_NAME=""
done

{
  echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "successful_actions=$success_count"
  echo "failed_or_timed_out_actions=$failure_count"
} >>"$LOG_ROOT/session.txt"

echo
echo "overnight sweep finished"
echo "successful actions: $success_count"
echo "failed or timed out actions: $failure_count"
echo "status: $STATUS_PATH"
echo "run paths: $RUN_PATHS_PATH"
