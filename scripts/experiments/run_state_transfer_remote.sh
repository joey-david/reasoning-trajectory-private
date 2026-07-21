#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${STATE_TRANSFER_NODES:-upnquick}"
DEVICES="${STATE_TRANSFER_DEVICES:-0+1}"
RUN="runs/Qwen2.5-32B-Instruct/interventions/state_transfer_causal_anchor"

"$PYTHON" scripts/experiments/depth_relief.py prepare-transfer "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py validate-transfer "$RUN"

"$PYTHON" scripts/orchestrate.py \
  --job state_transfer_capture \
  --nodes "$NODES" \
  --devices "$DEVICES" \
  --run "$RUN"

"$PYTHON" scripts/experiments/depth_relief.py fit-transfer "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py analyze-localization "$RUN"

"$PYTHON" scripts/orchestrate.py \
  --job state_transfer_patch \
  --nodes "$NODES" \
  --devices "$DEVICES" \
  --run "$RUN"

"$PYTHON" scripts/experiments/depth_relief.py analyze-transfer "$RUN"

if "$PYTHON" scripts/experiments/depth_relief.py handoff-eligibility "$RUN"; then
  "$PYTHON" scripts/experiments/depth_relief.py prepare-handoff "$RUN"

  "$PYTHON" scripts/orchestrate.py \
    --job state_handoff_patch \
    --nodes "$NODES" \
    --devices "$DEVICES" \
    --run "$RUN"

  "$PYTHON" scripts/experiments/depth_relief.py analyze-handoff "$RUN"
else
  status=$?
  if [[ $status -ne 3 ]]; then
    exit "$status"
  fi
  echo "Self-handoff skipped: localization and transfer gates did not both pass."
fi
