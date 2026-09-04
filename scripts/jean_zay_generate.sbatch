#!/usr/bin/env bash
#SBATCH --job-name=generate
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --error=slurm_logs/%x-%A.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "${FINEQCOMP_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp}/scripts/jean_zay_env.sh"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p slurm_logs

args=(scripts/generation/generate.py "${RUN_PATH:?set RUN_PATH}")
if [[ -n "${LIMIT:-}" ]]; then
  args+=(--limit "$LIMIT")
fi
exec "$PYTHON" "${args[@]}"
