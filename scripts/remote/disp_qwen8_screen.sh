#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
  echo "usage: scripts/remote/disp_qwen8_screen.sh runs/<model>/<purpose>/<run>" >&2
  exit 2
fi

run_path="$1"
cd /home/joey.david/reasoning-trajectory-private

# Source .env so that HF_TOKEN etc. are exported and reach Docker
if [ -f .env ]; then
  set -a; source .env; set +a
fi

docker run --rm \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e HF_HOME=/root/.cache/huggingface \
  -e HF_TOKEN \
  -e HUGGING_FACE_HUB_TOKEN \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  -v /home/joey.david/reasoning-trajectory-private:/workspace \
  -v /home/joey.david/.cache:/root/.cache \
  -w /workspace \
  python:3.11-slim \
  bash -lc "
    set -euo pipefail
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl
    python -V
    if [ ! -x .venv/bin/python ]; then
      python -m venv .venv
      .venv/bin/pip install -U pip setuptools wheel
      .venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128
      .venv/bin/pip install -r requirements.txt
    fi
    .venv/bin/python scripts/data/prepare_dataset.py ${run_path@Q}
    .venv/bin/python scripts/orchestrate.py --job generation --nodes local --devices 0 --run ${run_path@Q}
  "
