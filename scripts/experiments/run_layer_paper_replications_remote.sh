#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

PYTHON="${PYTHON:-.venv/bin/python}"
export HF_LOCAL_CACHE="${HF_LOCAL_CACHE:-$HOME/.cache/huggingface}"
export HF_HOME="$HF_LOCAL_CACHE"
if [[ "${LAYER_REPLICATION_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi
IFS=',' read -r -a GPUS <<<"${LAYER_REPLICATION_GPUS:-0,1}"
if (("${#GPUS[@]}" != 2)); then
  echo "LAYER_REPLICATION_GPUS must name exactly two comma-separated GPUs" >&2
  exit 2
fi

LAD_RUN="runs/Qwen2.5-1.5B/replications/lad_layer_robustness"
YANG_RUN="runs/Qwen2.5-7B/replications/yang_symbolic_mechanisms"
ZHANG_RUN="runs/Qwen3-1.7B-Base/replications/zhang_single_layer_rl"
DRIVER="scripts/experiments/replication/layer_papers.py"
SCAN="${LAYER_RL_SCAN:-core}"
pipeline_pids=()

write_checklist() {
  "$PYTHON" "$DRIVER" checklist || true
}

finish() {
  status=$?
  trap - EXIT
  write_checklist
  if ((status == 0)); then
    echo "Replication campaign complete. See experiments/layer_replication_checklist.md"
  else
    echo "Replication campaign stopped with status $status; rerun this script to resume." >&2
  fi
  exit "$status"
}
trap finish EXIT

terminate_tree() {
  local parent="$1" child
  while IFS= read -r child; do
    [[ -n "$child" ]] && terminate_tree "$child"
  done < <(pgrep -P "$parent" 2>/dev/null || true)
  kill -TERM "$parent" 2>/dev/null || true
}

stop() {
  trap - INT TERM
  local pid
  for pid in "${pipeline_pids[@]}"; do
    terminate_tree "$pid"
  done
  wait || true
  exit 130
}
trap stop INT TERM

validate_or_prepare() {
  local protocol="$1" run="$2"
  if ! "$PYTHON" "$DRIVER" "validate-$protocol" "$run" >/dev/null 2>&1; then
    "$PYTHON" "$DRIVER" "prepare-$protocol" "$run"
  fi
  "$PYTHON" "$DRIVER" "validate-$protocol" "$run"
}

check_hf_cache() {
  "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from src.runtime.config import load_config

runs = (
    Path("runs/Qwen2.5-1.5B/replications/lad_layer_robustness"),
    Path("runs/Qwen2.5-7B/replications/yang_symbolic_mechanisms"),
    Path("runs/Qwen3-1.7B-Base/replications/zhang_single_layer_rl"),
)
missing = []
for run in runs:
    model = load_config(run)["model"]
    name = model["name"]
    revision = model["revision"]
    try:
        common = {
            "revision": revision,
            "local_files_only": True,
            "trust_remote_code": bool(model.get("trust_remote_code", False)),
        }
        AutoConfig.from_pretrained(name, **common)
        AutoTokenizer.from_pretrained(name, **common)
        snapshot = (
            Path(os.environ["HF_HOME"])
            / "hub"
            / f"models--{name.replace('/', '--')}"
            / "snapshots"
            / revision
        )
        indexes = tuple(snapshot.glob("*.safetensors.index.json"))
        if indexes:
            shards = {
                shard
                for index in indexes
                for shard in json.loads(index.read_text())["weight_map"].values()
            }
            absent = sorted(
                shard for shard in shards if not (snapshot / shard).is_file()
            )
            if absent:
                raise FileNotFoundError(f"missing weight shards: {', '.join(absent)}")
        elif not any(snapshot.glob("*.safetensors")):
            raise FileNotFoundError("no safetensors weights or weight index")
    except Exception as error:
        missing.append(f"{name}@{revision}: {error}")
    else:
        print(f"cached: {name} -> {snapshot}")

zhang = load_config(runs[-1])["single_layer_rl"]
dataset_specs = [zhang["dataset"], *zhang["evaluation"]["benchmarks"]]
for spec in dataset_specs:
    try:
        dataset = load_dataset(
            spec["path"],
            spec.get("name"),
            split=spec.get("split", "train"),
            revision=spec["revision"],
        )
    except Exception as error:
        missing.append(f"{spec['path']}@{spec['revision']}: {error}")
    else:
        print(f"cached dataset: {spec['path']} ({len(dataset)} rows)")

if missing:
    details = "\n  ".join(missing)
    raise SystemExit(
        "Pinned Hugging Face artifacts are missing from HF_LOCAL_CACHE:\n  "
        + details
    )
PY
}

run_orchestrated() {
  local gpu="$1" job="$2" run="$3"
  "$PYTHON" scripts/orchestrate.py \
    --job "$job" --nodes local --devices "$gpu" --run "$run"
}

run_rl() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="$gpu" scripts/run_with_hf_download_fix.sh \
    "$PYTHON" "$DRIVER" "$@"
}

train_and_evaluate_layer() {
  local gpu="$1" layer="$2"
  run_rl "$gpu" train-rl "$ZHANG_RUN" --layer "$layer"
  run_rl "$gpu" evaluate-rl "$ZHANG_RUN" --layer "$layer"
}

echo "Hugging Face cache: $HF_LOCAL_CACHE"
scripts/experiments/install_layer_rl_deps.sh
if [[ "${LAYER_REPLICATION_OFFLINE:-0}" == "1" ]]; then
  check_hf_cache
fi
validate_or_prepare robustness "$LAD_RUN"
validate_or_prepare symbolic "$YANG_RUN"
"$PYTHON" "$DRIVER" validate-rl "$ZHANG_RUN"
write_checklist

if [[ "$SCAN" == "core" ]]; then
  layer_text="$("$PYTHON" - <<'PY'
from pathlib import Path
from src.runtime.config import load_config

run = Path("runs/Qwen3-1.7B-Base/replications/zhang_single_layer_rl")
print(" ".join(map(str, load_config(run)["single_layer_rl"]["core_scan_layers"])))
PY
)"
elif [[ "$SCAN" == "full" ]]; then
  layer_text="$(seq -s ' ' 0 27)"
else
  echo "LAYER_RL_SCAN must be core or full" >&2
  exit 2
fi
read -r -a layers <<<"$layer_text"
queue_zero=()
queue_one=()
for index in "${!layers[@]}"; do
  if ((index % 2 == 0)); then
    queue_zero+=("${layers[index]}")
  else
    queue_one+=("${layers[index]}")
  fi
done

run_layer_queue() {
  local gpu="$1"
  shift
  local layer
  for layer in "$@"; do
    train_and_evaluate_layer "$gpu" "$layer"
  done
}

# Each GPU follows a continuous independent pipeline, avoiding cross-paper
# barriers that would leave the faster side idle.
(
  run_orchestrated "${GPUS[0]}" layer_robustness "$LAD_RUN"
  "$PYTHON" "$DRIVER" analyze-robustness "$LAD_RUN"
  run_rl "${GPUS[0]}" evaluate-rl "$ZHANG_RUN" --base
  run_layer_queue "${GPUS[0]}" "${queue_zero[@]}"
) &
pipeline_zero_pid=$!
pipeline_pids+=("$pipeline_zero_pid")
(
  run_orchestrated "${GPUS[1]}" symbolic_screen "$YANG_RUN"
  "$PYTHON" "$DRIVER" select-symbolic "$YANG_RUN"
  run_orchestrated "${GPUS[1]}" symbolic_cma "$YANG_RUN"
  "$PYTHON" "$DRIVER" analyze-symbolic "$YANG_RUN"
  run_rl "${GPUS[1]}" train-rl "$ZHANG_RUN" --full
  run_rl "${GPUS[1]}" evaluate-rl "$ZHANG_RUN" --full
  run_layer_queue "${GPUS[1]}" "${queue_one[@]}"
) &
pipeline_one_pid=$!
pipeline_pids+=("$pipeline_one_pid")
status=0
wait "$pipeline_zero_pid" || status=$?
wait "$pipeline_one_pid" || status=$?
((status == 0)) || exit "$status"

analysis_layers="$(IFS=,; echo "${layers[*]}")"
"$PYTHON" "$DRIVER" analyze-rl "$ZHANG_RUN" --layers "$analysis_layers"
