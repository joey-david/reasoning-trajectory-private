#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
action="${1:-}"

prepare_training_run() {
  local run=$1
  "$PYTHON" scripts/experiments/run_state_handoff_training.py prepare-data "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py validate-data "$run"
}

run_generalization_pair() {
  local interface=$1
  local control=$2
  local nodes="${STATE_HANDOFF_NODES:-upnquick}"
  local devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  prepare_training_run "$interface"
  prepare_training_run "$control"
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$interface"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$interface"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-generalization "$interface"
}

run_replication_pair() {
  local seed2=$1
  local control2=$2
  local seed3=$3
  local control3=$4
  local nodes="${STATE_HANDOFF_NODES:-upnquick}"
  local devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  for run in "$seed2" "$control2" "$seed3" "$control3"; do
    prepare_training_run "$run"
  done
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$seed2"
  for run in "$seed2" "$seed3"; do
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
      compare-interfaces "$run"
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
      compare-generalization "$run"
  done
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-replication "$seed2"
}

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
continuation-probe-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0}"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-continuation "$run" --profile probe
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_continuation \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    analyze-information "$run" --profile probe
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    gate-continuation "$run"
  ;;
continuation-confirm-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_handoff_killtest"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0}"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-continuation "$run" --profile confirmation
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_continuation \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    analyze-information "$run" --profile confirmation
  ;;
interface-pilot-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
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
    compare-interfaces "$run"
  ;;
interface-final-eval-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_interface_rate_controls"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  current="$run/evaluation/interfaces"
  archive="$run/evaluation/interfaces_step250"
  if [[ -d "$current" && ! -e "$archive" ]]; then
    mv "$current" "$archive"
  fi
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$run"
  ;;
interface-stress-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_interface_stress"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-stress "$run" --profile probe
  "$PYTHON" scripts/orchestrate.py \
    --job state_interface_stress \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-stress "$run" --profile probe
  ;;
interface-closure-7b)
  closure="runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_finetune"
  control="runs/Qwen2.5-7B-Instruct/interventions/state_interface_endpoint_control"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  for run in "$closure" "$control"; do
    "$PYTHON" scripts/experiments/run_state_handoff_training.py prepare-data "$run"
    "$PYTHON" scripts/experiments/run_state_handoff_training.py validate-data "$run"
    "$PYTHON" scripts/orchestrate.py \
      --job state_handoff_training \
      --nodes "$nodes" \
      --devices "$devices" \
      --run "$run"
    "$PYTHON" scripts/experiments/run_state_handoff_training.py \
      compare-interfaces "$run"
  done
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-closure "$closure"
  ;;
interface-closure-stress-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_interface_closure_stress"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    prepare-stress "$run" --profile probe
  "$PYTHON" scripts/orchestrate.py \
    --job state_interface_stress \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-stress "$run" --profile probe
  ;;
interface-joint-closure-7b)
  run="runs/Qwen2.5-7B-Instruct/interventions/state_interface_joint_closure"
  nodes="${STATE_HANDOFF_NODES:-upnquick}"
  devices="${STATE_HANDOFF_7B_DEVICES:-0,1}"
  prepare_training_run "$run"
  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_training \
    --nodes "$nodes" \
    --devices "$devices" \
    --run "$run"
  "$PYTHON" scripts/experiments/run_state_handoff_training.py \
    compare-interfaces "$run"
  ;;
interface-algebra-transfer-7b)
  run_generalization_pair \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome"
  ;;
interface-width4-transfer-7b)
  run_generalization_pair \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_algebra" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_width4_outcome"
  ;;
interface-proof-transfer-7b)
  run_generalization_pair \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome"
  ;;
interface-algebra-confirm-7b)
  run_replication_pair \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed2" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed2" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_transfer_seed3" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_algebra_outcome_seed3"
  ;;
interface-proof-confirm-7b)
  run_replication_pair \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed2" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed2" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_proof_seed3" \
    "runs/Qwen2.5-7B-Instruct/interventions/state_interface_horn_outcome_seed3"
  ;;
interface-proof-second-model-7b)
  run_generalization_pair \
    "runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_proof" \
    "runs/Mistral-7B-Instruct-v0.3/interventions/state_interface_horn_outcome"
  ;;
*)
  echo "usage: scripts/remote/state_handoff.sh phase1-32b | screen-7b | prepare-pilot-7b | pilot-7b | continuation-probe-7b | continuation-confirm-7b | interface-pilot-7b | interface-final-eval-7b | interface-stress-7b | interface-closure-7b | interface-closure-stress-7b | interface-joint-closure-7b | interface-algebra-transfer-7b | interface-width4-transfer-7b | interface-proof-transfer-7b | interface-algebra-confirm-7b | interface-proof-confirm-7b | interface-proof-second-model-7b" >&2
  exit 2
  ;;
esac
