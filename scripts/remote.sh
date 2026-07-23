#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | pull [-h|--hidden-states] [--pt] [runs/<model>/<experiment> ...] | pull-stats [runs/<model>/<experiment> ...]"
action="${1:?$usage}"
shift
host="${SSH_SERVER:-lamgate}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/reasoning}"
include_hidden_states=false
include_pt=false
pull_paths=()

# Inputs: none; scans the local runs directory.
# Returns: sorted run paths on standard output and find's pipeline status.
discover_runs() {
  find runs -mindepth 3 -name config.yaml -print |
    sed 's#/config.yaml$##' |
    sort
}

# Inputs: one run path relative to the repository root.
# Returns: rsync's status, or success after reporting a missing remote run.
pull_run() {
  local run_path="$1"
  if ssh "$host" "test -d '$remote_root/$run_path'"; then
    mkdir -p "$run_path"
    local rsync_args=(-avz --progress)
    if [[ "$include_hidden_states" != true ]]; then
      rsync_args+=(--exclude "*/hidden_states/***")
      rsync_args+=(--exclude "*/activations/***")
      rsync_args+=(--exclude "layer_replications/*/checkpoints/***")
      rsync_args+=(--exclude "*.npz")
    fi
    if [[ "$include_pt" != true ]]; then
      rsync_args+=(--exclude "*.pt")
      rsync_args+=(--exclude "*.safetensors")
    fi
    rsync "${rsync_args[@]}" "$host:$remote_root/$run_path/" "$run_path/"
  else
    echo "skip missing remote run: $run_path" >&2
  fi
}

# Inputs: one run path relative to the repository root.
# Returns: rsync's status, limited to lightweight live-progress files.
pull_run_stats() {
  local run_path="$1"
  if ssh "$host" "test -d '$remote_root/$run_path'"; then
    mkdir -p "$run_path"
    rsync -avz --prune-empty-dirs \
      --include '*/' \
      --include 'config.yaml' \
      --include 'generation/generations.jsonl' \
      --include 'generation/metadata.json' \
      --include 'generation/orchestrator_logs/***' \
      --include 'analysis/live_screening_stats.json' \
      --include 'analysis/live_screening_stats.csv' \
      --exclude '*' \
      "$host:$remote_root/$run_path/" "$run_path/"
  else
    echo "skip missing remote run: $run_path" >&2
  fi
}

case "$action" in
push)
  rsync -avz --delete --prune-empty-dirs \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude .tmp/ \
    --include '/runs/' \
    --include '/runs/**/' \
    --include '/runs/**/config.yaml' \
    --include '/runs/**/dataset.jsonl' \
    --include '/runs/**/layer_replications/**/dataset_manifest.json' \
    --exclude '/runs/**' \
    --exclude '__pycache__' \
    ./ "$host:$remote_root/"
  ;;
pull)
  while (($#)); do
    case "$1" in
    -h|--hidden-states)
      include_hidden_states=true
      ;;
    --pt)
      include_pt=true
      ;;
    -*)
      echo "unknown pull option: $1" >&2
      echo "$usage" >&2
      exit 2
      ;;
    *)
      pull_paths+=("$1")
      ;;
    esac
    shift
  done
  if ((${#pull_paths[@]})); then
    for run_path in "${pull_paths[@]}"; do
      pull_run "$run_path"
    done
  else
    while IFS= read -r run_path; do
      pull_run "$run_path"
    done < <(discover_runs)
  fi
  ;;
pull-stats)
  if (($#)); then
    for run_path in "$@"; do
      pull_run_stats "$run_path"
    done
  else
    while IFS= read -r run_path; do
      pull_run_stats "$run_path"
    done < <(discover_runs)
  fi
  ;;
*)
  echo "unknown action: $action" >&2
  echo "$usage" >&2
  exit 2
  ;;
esac
