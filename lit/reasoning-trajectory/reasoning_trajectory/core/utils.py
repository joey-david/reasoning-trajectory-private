from __future__ import annotations

import importlib.util
import random
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        return


def dependency_status(names: Iterable[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def as_array(points) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {arr.shape}")
    return arr
