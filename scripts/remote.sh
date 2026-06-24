#!/usr/bin/env bash
set -euo pipefail

usage="usage: scripts/remote.sh push [run] | pull runs/<model>/<experiment>"
action="${1:?$usage}"
run_path="${2:-}"
host="${SSH_SERVER:-lamgate}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/reasoning}"

case "$action" in
push)
  rsync -avz --delete \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude .tmp/ \
    --exclude 'runs/*/*/generation/' \
    --exclude 'runs/*/*/analysis/' \
    --exclude '__pycache__' \
    ./ "$host:$remote_root/"
  ;;
pull)
  [[ -n "$run_path" ]] || { echo "$usage" >&2; exit 2; }
  mkdir -p "$run_path"
  rsync -avz --progress "$host:$remote_root/$run_path/" "$run_path/"
  ;;
*)
  echo "unknown action: $action" >&2
  echo "$usage" >&2
  exit 2
  ;;
esac
