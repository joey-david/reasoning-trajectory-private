#!/usr/bin/env bash
set -euo pipefail

cache_root="${HF_LOCAL_CACHE:-/tmp/${USER}/huggingface}"
mkdir -p "$cache_root"

export HF_HOME="$cache_root"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
unset HF_HUB_ENABLE_HF_TRANSFER

probe_url="https://huggingface.co/HuggingFaceTB/SmolLM3-3B/resolve/a07cc9a04f16550a088caea529712d1d335b0ac1/tokenizer.json"
curl -4 -L --fail --silent --show-error \
  --retry 5 --retry-all-errors --connect-timeout 20 --max-time 120 \
  --range 0-1023 --output /dev/null "$probe_url"

if (($#)); then
  command_dir="$(cd "$(dirname "$1")" && pwd)"
  export PATH="$command_dir:$PATH"
  exec "$@"
fi

exec python scripts/generation/generate.py \
  runs/SmolLM3-3B/frontier_identification/gsm_symbolic_p1_screen \
  runs/SmolLM3-3B/frontier_identification/mbppplus_codegen_screen \
  runs/Qwen3-4B/polymath_medium_numeric_screen \
  runs/Qwen3-4B/mbppplus_codegen_screen \
  runs/DeepSeek-R1-Distill-Qwen-7B/polymath_medium_numeric_screen \
  runs/DeepSeek-R1-Distill-Qwen-7B/bigcodebench_hard_codegen_screen
