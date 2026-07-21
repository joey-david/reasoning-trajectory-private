#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
action="${1:-}"

case "$action" in
phase1-32b)
  run="runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_32B_DEVICES:-0+1}"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-explicit-handoff "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_explicit_handoff \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-explicit-handoff "$run"
  ;;
screen-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0}"
  "$PYTHON" scripts/experiments/depth_relief.py prepare-abstraction "$run"
  "$PYTHON" scripts/experiments/depth_relief.py validate-factorization "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_materialization \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py analyze-factorization "$run"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-explicit-handoff "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_explicit_handoff \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-explicit-handoff "$run"
  ;;
prepare-pilot-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    validate-data "$run"
  ;;
pilot-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0}"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-data "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    validate-data "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare "$run"
  ;;
*)
  echo "usage: scripts/remote/state_handoff.sh phase1-32b | screen-7b | prepare-pilot-7b | pilot-7b" >&2
  exit 2
  ;;
esac
