#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
NODES="${DEPTH_RELIEF_NODES:-upnquick}"
DEVICES="${DEPTH_RELIEF_LARGE_DEVICES:-0+1}"
RUN="runs/Qwen2.5-32B-Instruct/interventions/depth_relief_frontier_calibration"

"$PYTHON" scripts/orchestrate.py \
  --job depth_relief_calibration \
  --nodes "$NODES" \
  --devices "$DEVICES" \
  --run "$RUN"

"$PYTHON" scripts/experiments/depth_relief.py analyze-calibration "$RUN"
