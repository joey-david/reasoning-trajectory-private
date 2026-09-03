#!/usr/bin/env bash
#SBATCH --job-name=route-heads
#SBATCH --account=fas@h100
#SBATCH --partition=gpu_p6
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail
repo_root="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$repo_root"
source "${FINEQCOMP_ROOT:-/lustre/fswork/projects/rech/fas/uul94gf/fineQComp}/scripts/jean_zay_env.sh"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p slurm_logs
exec "$PYTHON" scripts/experiments/state_routing_heads.py "${ACTION:?set ACTION}" \
  "${RUN_PATH:?set RUN_PATH}" --split "${SPLIT:-development}" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --shards "${SHARDS:-1}"
