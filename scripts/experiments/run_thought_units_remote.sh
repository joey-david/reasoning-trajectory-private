#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

all_nodes=(ourasi kaisertrot upnquick coktailjet)
all_devices=(1 0,1 0 0,1)

run_distributed_job() {
  local job="$1"
  local run_path="$2"

  .venv/bin/python scripts/orchestrate.py \
    --job "$job" \
    --nodes "${all_nodes[@]}" \
    --devices "${all_devices[@]}" \
    --run "$run_path"
}

run_single_job() {
  local job="$1"
  local run_path="$2"

  .venv/bin/python scripts/orchestrate.py \
    --job "$job" \
    --nodes upnquick \
    --devices 0 \
    --run "$run_path"
}

run_single_job gold_answer_capture \
  runs/SmolLM3-3B/thought_units_gold_answers

run_distributed_job generation \
  runs/Qwen3-14B/thought_units_gsm_symbolic

run_single_job gold_answer_capture \
  runs/Qwen3-14B/thought_units_gold_answers

run_distributed_job boundary_intervention \
  runs/SmolLM3-3B/thought_units_boundary_interventions
