#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
QUEUE_SCRIPT="${STATE_HANDOFF_QUEUE_SCRIPT:-$REPO_ROOT/../queue_job.sh}"
SEVEN_B_GPU="${STATE_HANDOFF_7B_GPU:-0}"

if [[ ! -f "$QUEUE_SCRIPT" ]]; then
  echo "missing queue script: $QUEUE_SCRIPT" >&2
  exit 2
fi
if [[ ! "$SEVEN_B_GPU" =~ ^[01]$ ]]; then
  echo "STATE_HANDOFF_7B_GPU must be 0 or 1" >&2
  exit 2
fi

queue_stage() {
  local gpu_list="$1"
  local action="$2"
  bash "$QUEUE_SCRIPT" localhost "$gpu_list" env \
    STATE_HANDOFF_NODES=local \
    STATE_HANDOFF_32B_DEVICES=0+1 \
    STATE_HANDOFF_7B_DEVICES="$SEVEN_B_GPU" \
    bash scripts/remote/state_handoff.sh "$action"
}

summary_status() {
  local path="$1"
  local key="$2"
  "$PYTHON" - "$path" "$key" <<'PY'
import json
from pathlib import Path
import sys

summary = json.loads(Path(sys.argv[1]).read_text())
for part in sys.argv[2].split("."):
    summary = summary[part]
print(summary)
PY
}

queue_stage "0,1" phase1-32b

PHASE1_SUMMARY="runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history/depth_relief/explicit_handoff/summary.json"
phase1_status="$(summary_status "$PHASE1_SUMMARY" "phase1_gate.status")"
if [[ "$phase1_status" != "passed" ]]; then
  echo "Phase 1 ended with status: $phase1_status. Stopping before the 7B jobs." >&2
  exit 0
fi

queue_stage "$SEVEN_B_GPU" screen-7b
queue_stage "$SEVEN_B_GPU" pilot-7b

PILOT_SUMMARY="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest/evaluation/comparison_summary.json"
pilot_status="$(summary_status "$PILOT_SUMMARY" "pilot_gate.status")"
echo "State-handoff pilot gate: $pilot_status"
echo "Stopping here; opaque codes and full confirmation require review of the pilot."
