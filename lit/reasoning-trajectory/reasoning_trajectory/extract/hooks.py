from __future__ import annotations

from contextlib import contextmanager
from typing import Callable


@contextmanager
def capture_module_outputs(module, transform: Callable | None = None):
    """Capture PyTorch module outputs during a forward pass."""
    outputs = []

    def hook(_module, _inputs, output):
        outputs.append(transform(output) if transform else output)

    handle = module.register_forward_hook(hook)
    try:
        yield outputs
    finally:
        handle.remove()
