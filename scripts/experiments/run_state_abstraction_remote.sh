#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${STATE_ABSTRACTION_NODES:-upnquick}"
DEVICES="${STATE_ABSTRACTION_DEVICES:-0+1}"
RUN="runs/Qwen2.5-32B-Instruct/interventions/state_abstraction_matched_history"

echo "=== Matched-history state abstraction capture: $RUN ==="
"$PYTHON" scripts/experiments/depth_relief.py prepare-abstraction "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py validate-abstraction "$RUN"
"$PYTHON" scripts/orchestrate.py \
  --job state_abstraction_capture \
  --nodes "$NODES" \
  --devices "$DEVICES" \
  --run "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py analyze-factorization "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py \
  analyze-abstraction-information "$RUN"

if "$PYTHON" scripts/experiments/depth_relief.py \
  abstraction-interchange-eligibility "$RUN"; then
  echo "=== Matched implicit-state interchange: $RUN ==="
  "$PYTHON" scripts/experiments/depth_relief.py \
    prepare-abstraction-interchange "$RUN"
  "$PYTHON" scripts/orchestrate.py \
    --job state_abstraction_interchange \
    --nodes "$NODES" \
    --devices "$DEVICES" \
    --run "$RUN"
  "$PYTHON" scripts/experiments/depth_relief.py \
    analyze-abstraction-interchange "$RUN"
else
  code=$?
  if [[ $code -ne 3 ]]; then
    exit "$code"
  fi
  echo "Causal interchange skipped: matched behavioral pair gate did not pass."
fi

echo "State-abstraction experiment completed."
