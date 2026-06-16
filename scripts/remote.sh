#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/remote.sh push runs/<model>/<experiment>
#   bash scripts/remote.sh pull runs/<model>/<experiment>
#
#   run generation manually after pushing:
#   ssh lamgate
#   cd /home/lamsade/jdavid/research/reasoning
#   source .venv/bin/activate
#   python scripts/generate.py runs/<model>/<experiment>

# NOTE: bash parsing: if the first argument is missing (1:?)
# set the value of the argument to the error message to show
action="${1:?usage: scripts/remote.sh push|pull runs/<model>/<experiment>}"
# resp second
run_path="${2:?usage: scripts/remote.sh push|pull runs/<model>/<experiment>}"
# ENV_VAR:-default
host="${SSH_SERVER:-lamgate}"
remote_root="${REMOTE_REPO_ROOT:-/home/lamsade/jdavid/research/reasoning}"

case "$action" in
push)
  # sync code, the run config, and datasets
  # --delete: make the destination mirror the source
  # NOTE: avz: archive, verbose, compress.
  rsync -avz --delete \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude 'runs/*/*/generation/' \
    --exclude 'runs/*/*/analysis/' \
    ./ "$host:$remote_root/"
  ;;
pull)
  # Pull only the selected run folder back
  mkdir -p "$run_path" # NOTE: -p: if needed
  rsync -avz --progress "$host:$remote_root/$run_path/" "$run_path/"
  ;;
*)
  # use the error messages from before if pull/push weren't parsed
  echo "unknown action: $action" >&2 # NOTE: >&2: send output to stderr
  echo "usage: scripts/remote.sh push|pull runs/<model>/<experiment>" >&2
  exit 2
  ;;
esac
