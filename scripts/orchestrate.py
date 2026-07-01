#!/usr/bin/env python3
"""Run any registered orchestration job across local or SSH GPUs."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestration.remote import main


if __name__ == "__main__":
    raise SystemExit(main())
