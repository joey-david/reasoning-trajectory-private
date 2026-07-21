#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${STATE_ROUTING_NODES:-upnquick}"
DEVICES="${STATE_ROUTING_DEVICES:-0+1}"
RUN="runs/Qwen2.5-32B-Instruct/interventions/state_routing_confirmation"

"$PYTHON" scripts/orchestrate.py \
  --job state_routing \
  --nodes "$NODES" \
  --devices "$DEVICES" \
  --run "$RUN"
"$PYTHON" scripts/experiments/depth_relief.py analyze-routing "$RUN"
