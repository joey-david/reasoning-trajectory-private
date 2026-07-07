#!/usr/bin/env bash
set -u

usage() {
  echo "usage: queue NODE GPU_IDXS COMMAND [ARG ...]" >&2
  echo "example: queue upnquick 0,1 ./scripts/run_job.sh runs/my_run" >&2
}

if (( $# < 3 )); then
  usage
  exit 2
fi

node=$1
gpu_list=$2
shift 2

if [[ ! $gpu_list =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "queue: GPU_IDXS must look like 0 or 0,1" >&2
  exit 2
fi

poll_seconds=${QUEUE_POLL_SECONDS:-10}
if [[ ! $poll_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "queue: QUEUE_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

IFS=, read -r -a requested_gpus <<< "$gpu_list"
local_short=$(hostname -s 2>/dev/null || hostname)
local_full=$(hostname -f 2>/dev/null || hostname)
ready_streak=0

query_utilization() {
  local query=(
    nvidia-smi
    --query-gpu=index,utilization.gpu
    --format=csv,noheader,nounits
  )

  if [[ $node == localhost || $node == "$local_short" || $node == "$local_full" ]]; then
    "${query[@]}"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" \
      'nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits'
  fi
}

while true; do
  if ! output=$(query_utilization 2>&1); then
    ready_streak=0
    printf '\r\033[2K%s: unavailable [0/4]' "$node"
    sleep "$poll_seconds"
    continue
  fi

  declare -A utilization=()
  while IFS=, read -r index percent; do
    index=${index//[[:space:]]/}
    percent=${percent//[[:space:]]/}
    if [[ $index =~ ^[0-9]+$ && $percent =~ ^[0-9]+$ ]]; then
      utilization["$index"]=$percent
    fi
  done <<< "$output"

  ready_now=1
  statuses=()
  for index in "${requested_gpus[@]}"; do
    if [[ -z ${utilization[$index]+present} ]]; then
      statuses+=("gpu$index=missing")
      ready_now=0
    else
      percent=${utilization["$index"]}
      statuses+=("gpu$index=${percent}%")
      if (( percent >= 5 )); then
        ready_now=0
      fi
    fi
  done

  if (( ready_now )); then
    ((ready_streak += 1))
  else
    ready_streak=0
  fi

  printf '\r\033[2K%s: %s [%d/4]' \
    "$node" "${statuses[*]}" "$ready_streak"
  if (( ready_streak >= 4 )); then
    printf '\r\033[2K'
    exec "$@"
  fi

  sleep "$poll_seconds"
done
