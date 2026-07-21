#!/usr/bin/env bash
set -euo pipefail

cache_root="${HF_LOCAL_CACHE:-${HOME}/.cache/huggingface}"
mkdir -p "$cache_root"

export HF_HOME="$cache_root"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
unset HF_HUB_ENABLE_HF_TRANSFER

if (($#)); then
  command_dir="$(cd "$(dirname "$1")" && pwd)"
  export PATH="$command_dir:$PATH"
  exec "$@"
fi

exec python scripts/generation/generate.py \
  runs/SmolLM3-3B/screening/frontier_identification/gsm_symbolic_p1_screen \
  runs/SmolLM3-3B/screening/frontier_identification/mbppplus_codegen_screen \
  runs/Qwen3-4B/screening/polymath_medium_numeric_screen \
  runs/Qwen3-4B/screening/mbppplus_codegen_screen \
  runs/DeepSeek-R1-Distill-Qwen-7B/screening/polymath_medium_numeric_screen \
  runs/DeepSeek-R1-Distill-Qwen-7B/screening/bigcodebench_hard_codegen_screen
