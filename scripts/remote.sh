#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push | pull [runs/<model>/<experiment> ...]"
action="${1:?$usage}"
shift
host="${SSH_SERVER:-lamgate}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/reasoning}"

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
    rsync -avz --progress "$host:$remote_root/$run_path/" "$run_path/"
  else
    echo "skip missing remote run: $run_path" >&2
  fi
}

case "$action" in
push)
  rsync -avz --delete \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude .tmp/ \
    --exclude 'runs/**/generation/hidden_states' \
    --exclude 'runs/**/analysis/' \
    --exclude '__pycache__' \
    ./ "$host:$remote_root/"
  ;;
pull)
  if (($#)); then
    for run_path in "$@"; do
      pull_run "$run_path"
    done
  else
    while IFS= read -r run_path; do
      pull_run "$run_path"
    done < <(discover_runs)
  fi
  ;;
*)
  echo "unknown action: $action" >&2
  echo "$usage" >&2
  exit 2
  ;;
esac
