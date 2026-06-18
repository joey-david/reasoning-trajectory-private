#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [[ -d datasets ]] && [[ -z "$(find datasets -mindepth 1 -print -quit)" ]]; then
  rmdir datasets
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

venv_python=".venv/bin/python"

uv pip install --python "$venv_python" -r requirements.txt
uv pip install --python "$venv_python" packaging ninja wheel setuptools
"$venv_python" - <<'PY'
import datasets
import torch
print(f"base deps ok: datasets={datasets.__version__} torch={torch.__version__}")
PY

uv pip uninstall --python "$venv_python" -y flash-attn || true
uv pip install --python "$venv_python" -r requirements.txt
flash_abi="$("$venv_python" - <<'PY'
import torch
assert torch.__version__ == "2.6.0+cu124", torch.__version__
print("1" if torch._C._GLIBCXX_USE_CXX11_ABI else "0")
PY
)"
if [[ "$flash_abi" == "1" ]]; then
  export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE
else
  export FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE
fi
export CFLAGS="${CFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=$flash_abi"
export CXXFLAGS="${CXXFLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=$flash_abi"
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-} -D_GLIBCXX_USE_CXX11_ABI=$flash_abi"
"$venv_python" - <<'PY'
import os
import torch
print(
    f"building flash-attn against torch={torch.__version__} "
    f"cuda={torch.version.cuda} cxx11_abi={torch._C._GLIBCXX_USE_CXX11_ABI} "
    f"flash_abi={os.environ['FLASH_ATTENTION_FORCE_CXX11_ABI']}"
)
PY
MAX_JOBS="${MAX_JOBS:-4}" uv pip install \
  --python "$venv_python" \
  --no-build-isolation \
  --no-binary flash-attn \
  --no-cache \
  --no-deps \
  --reinstall \
  flash-attn==2.8.3.post1

if ! "$venv_python" - <<'PY'
from datasets import load_dataset
import torch

assert torch.__version__ == "2.6.0+cu124", torch.__version__
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"datasets.load_dataset={load_dataset.__module__}.{load_dataset.__name__}")
try:
    from flash_attn import flash_attn_func
except Exception as exc:
    print(f"flash_attn=broken: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
else:
    print(f"flash_attn_func={flash_attn_func.__name__}")
PY
then
  echo "flash-attn is not importable with this Torch/CUDA ABI; uninstalling it so generation can fall back to SDPA."
  uv pip uninstall --python "$venv_python" -y flash-attn || true
fi

echo "activate with: source .venv/bin/activate"
