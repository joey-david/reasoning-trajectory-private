#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments/scripts/run_remote_gpu.sh --config experiments/configs/r1_distill_sheep30.yaml --name r1_distill_sheep30 --layer 32

Reads .env from the repo root. The script:
  1. syncs the repo to $SSH_SERVER:$REMOTE_REPO_ROOT;
  2. runs rt on $GPU_HOST if set, otherwise directly on $SSH_SERVER;
  3. writes the remote run to $REMOTE_RUN_ROOT/<name>;
  4. copies that run back to $LOCAL_RUN_ROOT/<name>.
USAGE
}

CONFIG=""
RUN_NAME=""
LAYER=""
ANALYZE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --name) RUN_NAME="$2"; shift 2 ;;
    --layer) LAYER="$2"; shift 2 ;;
    --no-analyze) ANALYZE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Missing --config" >&2
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and fill it in." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${SSH_SERVER:?Set SSH_SERVER in .env}"
: "${REMOTE_REPO_ROOT:?Set REMOTE_REPO_ROOT in .env}"
: "${REMOTE_RUN_ROOT:?Set REMOTE_RUN_ROOT in .env}"
: "${LOCAL_RUN_ROOT:=experiments/runs}"
: "${REMOTE_RT_BIN:=rt}"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="$(basename "$CONFIG")"
  RUN_NAME="${RUN_NAME%.*}"
fi

REMOTE_OUT="${REMOTE_RUN_ROOT%/}/${RUN_NAME}"
LOCAL_OUT="${LOCAL_RUN_ROOT%/}/${RUN_NAME}"
REMOTE_CONFIG="${REMOTE_REPO_ROOT%/}/${CONFIG}"

echo "Syncing repo to ${SSH_SERVER}:${REMOTE_REPO_ROOT}"
rsync -az --delete \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'experiments/runs' \
  ./ "${SSH_SERVER}:${REMOTE_REPO_ROOT%/}/"

read -r -d '' REMOTE_SCRIPT <<'REMOTE' || true
set -euo pipefail
cd "$REMOTE_REPO_ROOT"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export TRANSFORMERS_NO_TORCHVISION="${TRANSFORMERS_NO_TORCHVISION:-1}"
export TRANSFORMERS_NO_VISION="${TRANSFORMERS_NO_VISION:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "$(dirname "$REMOTE_OUT")"
cmd=("$REMOTE_RT_BIN" run --config "$REMOTE_CONFIG" --out "$REMOTE_OUT")
if [[ -n "${LAYER:-}" ]]; then
  cmd+=(--layer "$LAYER")
fi
if [[ "${ANALYZE:-1}" == "0" ]]; then
  cmd+=(--no-analyze)
fi
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
REMOTE

remote_payload() {
  printf 'export REMOTE_REPO_ROOT=%q\n' "$REMOTE_REPO_ROOT"
  printf 'export REMOTE_OUT=%q\n' "$REMOTE_OUT"
  printf 'export REMOTE_CONFIG=%q\n' "$REMOTE_CONFIG"
  printf 'export REMOTE_RT_BIN=%q\n' "$REMOTE_RT_BIN"
  printf 'export HF_TOKEN=%q\n' "${HF_TOKEN:-}"
  printf 'export HF_HUB_DISABLE_XET=%q\n' "${HF_HUB_DISABLE_XET:-1}"
  printf 'export HF_HUB_DISABLE_PROGRESS_BARS=%q\n' "${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
  printf 'export TRANSFORMERS_NO_TORCHVISION=%q\n' "${TRANSFORMERS_NO_TORCHVISION:-1}"
  printf 'export TRANSFORMERS_NO_VISION=%q\n' "${TRANSFORMERS_NO_VISION:-1}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES:-0}"
  printf 'export LAYER=%q\n' "$LAYER"
  printf 'export ANALYZE=%q\n' "$ANALYZE"
  printf '%s\n' "$REMOTE_SCRIPT"
}

echo "Running remote experiment ${RUN_NAME}"
if [[ -n "${GPU_HOST:-}" ]]; then
  remote_payload | ssh -o BatchMode=yes "$SSH_SERVER" \
    "ssh -o BatchMode=yes '$GPU_HOST' bash -s"
else
  remote_payload | ssh -o BatchMode=yes "$SSH_SERVER" "bash -s"
fi

echo "Copying result back to ${LOCAL_OUT}"
mkdir -p "$LOCAL_OUT"
rsync -az "${SSH_SERVER}:${REMOTE_OUT%/}/" "${LOCAL_OUT%/}/"
echo "Done: ${LOCAL_OUT}"
