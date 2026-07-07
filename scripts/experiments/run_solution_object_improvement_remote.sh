#!/usr/bin/env bash
set -euo pipefail
trap 'kill $(jobs -pr) 2>/dev/null || true' EXIT

if [[ -z "${SOLUTION_OBJECT_12H_GUARD:-}" ]]; then
  export SOLUTION_OBJECT_12H_GUARD=1
  exec timeout --foreground 12h "$0" "$@"
fi

RUN="${1:-runs/SmolLM3-3B/interventions/solution_object_extraction_medium}"
PYTHON="${PYTHON:-.venv/bin/python}"
ENTRY="scripts/experiments/solution_object_extraction.py"
LOG_DIR="$RUN/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/improvement_remote.log") 2>&1

echo "Preparing and capturing five-layer feature views on GPU 0"
"$PYTHON" "$ENTRY" prepare "$RUN"
capture_report="$RUN/analysis/experiments/solution_object_extraction/capture_report.json"
capture_features="$RUN/analysis/experiments/solution_object_extraction/captured_features.npz"
if [[ -f "$capture_report" && -f "$capture_features" ]]; then
  echo "Reusing completed capture: $capture_features"
else
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 45m \
    "$PYTHON" "$ENTRY" capture "$RUN"
fi

echo "Running retrieval sweep"
retrieval_report="$RUN/analysis/experiments/solution_object_extraction/improvement/retrieval_sweep.json"
if [[ -f "$retrieval_report" ]]; then
  echo "Reusing completed retrieval sweep: $retrieval_report"
else
  CUDA_VISIBLE_DEVICES="" timeout --foreground 30m \
    "$PYTHON" "$ENTRY" retrieval-sweep "$RUN"
fi

echo "Running causal and nonlinear sweeps on GPU 0"
causal_report="$RUN/analysis/experiments/solution_object_extraction/improvement/causal_sweep.json"
nonlinear_report="$RUN/analysis/experiments/solution_object_extraction/improvement/nonlinear_sweep.json"
if [[ -f "$causal_report" ]]; then
  echo "Reusing completed causal sweep: $causal_report"
else
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 4h \
    "$PYTHON" "$ENTRY" causal-sweep "$RUN"
fi
if [[ -f "$nonlinear_report" ]]; then
  echo "Reusing completed nonlinear sweep: $nonlinear_report"
else
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 4h \
    "$PYTHON" "$ENTRY" nonlinear "$RUN"
fi

echo "Running fixed writer and targeted matched ablations on GPU 0"
CUDA_VISIBLE_DEVICES=0 timeout --foreground 5h \
  "$PYTHON" "$ENTRY" writer "$RUN"
ablation_report="$RUN/analysis/experiments/solution_object_extraction/improvement/ablation_sweep.json"
if [[ -f "$ablation_report" ]]; then
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 2h \
    "$PYTHON" "$ENTRY" ablation-grid "$RUN"
else
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 2h \
    "$PYTHON" "$ENTRY" ablation-sweep "$RUN"
  CUDA_VISIBLE_DEVICES=0 timeout --foreground 2h \
    "$PYTHON" "$ENTRY" ablation-grid "$RUN"
fi

echo "Validating completed medium artifacts"
"$PYTHON" "$ENTRY" validate-improvement "$RUN"
