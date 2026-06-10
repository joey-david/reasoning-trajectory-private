from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(start: str | Path = ".") -> None:
    path = _find_dotenv(Path(start).resolve())
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _find_dotenv(start: Path) -> Path | None:
    here = start if start.is_dir() else start.parent
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None
