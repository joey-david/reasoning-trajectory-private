#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-.venv/bin/python}"
TORCH_VERSION="2.6.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

runtime_works() {
  "$PYTHON" - <<'PY' >/dev/null 2>&1
import math_verify
import torch
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import GRPOTrainer

assert torch.__version__.split("+", 1)[0] == "2.6.0"
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
assert trl.__version__ == "1.8.0"
PY
}

remove_if_installed() {
  local package="$1"
  if "$PYTHON" -c \
    "import importlib.util; raise SystemExit(importlib.util.find_spec('$package') is None)"
  then
    if command -v uv >/dev/null 2>&1; then
      uv pip uninstall --python "$PYTHON" "$package"
    else
      "$PYTHON" -m pip uninstall -y "$package"
    fi
  fi
}

# These text-only experiments do not use TorchVision or TorchAudio. Removing
# them prevents optional Transformers imports from loading binary extensions
# compiled for a different Torch/CUDA runtime.
if runtime_works; then
  exit 0
fi

remove_if_installed vllm
remove_if_installed torchvision
remove_if_installed torchaudio

# An optional package can be the only broken component. In particular, TRL
# imports vLLM when it is installed even though this experiment uses the
# Transformers continuous-batching backend.
if runtime_works; then
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PYTHON" --reinstall-package torch \
    --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION"
  uv pip install --python "$PYTHON" --upgrade \
    "torch==$TORCH_VERSION" \
    "trl==1.8.0" \
    "math-verify>=0.5.2,<1"
else
  "$PYTHON" -m ensurepip --upgrade
  "$PYTHON" -m pip install --force-reinstall \
    "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  "$PYTHON" -m pip install --upgrade \
    "torch==$TORCH_VERSION" \
    "trl==1.8.0" \
    "math-verify>=0.5.2,<1"
fi

if ! runtime_works; then
  echo "CUDA 12.4 text-model dependency check failed after installation:" >&2
  "$PYTHON" - <<'PY'
import torch

print({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
})
import math_verify
import trl
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import GRPOTrainer

assert trl.__version__ == "1.8.0"
PY
  exit 1
fi
