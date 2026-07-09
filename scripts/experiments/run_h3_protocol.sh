#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/experiments/run_h3_protocol.sh primary | fallback"
stage="${1:?$usage}"

replay_run="runs/SmolLM3-3B/failed/h3_process_isomer_replay"
case "$stage" in
primary)
  patch_run="runs/SmolLM3-3B/failed/h3_process_isomer_patching"
  ;;
fallback)
  patch_run="runs/SmolLM3-3B/failed/h3_process_isomer_patching_mlp18"
  ;;
*)
  echo "$usage" >&2
  exit 2
  ;;
esac

.venv/bin/python scripts/experiments/trajectory_dynamics/replay_capture.py "$replay_run"
.venv/bin/python scripts/experiments/process_isomers/causal_patching.py \
  "$patch_run" --validate-only
.venv/bin/python scripts/experiments/process_isomers/causal_patching.py \
  "$patch_run" --max-pairs 2 --continuations-per-condition 1
.venv/bin/python scripts/experiments/process_isomers/analyze_causal_patching.py \
  "$patch_run" --smoke-gate --smoke-pairs 2 --smoke-continuations 1
if [[ -n "${H3_DEVICES:-}" ]]; then
  .venv/bin/python scripts/orchestrate.py \
    --job causal_patching \
    --nodes local \
    --devices "$H3_DEVICES" \
    --run "$patch_run"
else
  .venv/bin/python scripts/experiments/process_isomers/causal_patching.py "$patch_run"
fi
.venv/bin/python scripts/experiments/process_isomers/analyze_causal_patching.py "$patch_run"
