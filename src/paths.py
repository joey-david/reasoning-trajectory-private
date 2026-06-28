"""Resolve repository-relative paths against the checkout root."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_text: str | Path) -> Path:
    """Convert a path value into an absolute-or-repository-rooted path.

    Args:
        path_text: Absolute path or path relative to the repository root.

    Returns:
        The original absolute path or the relative path joined to ``REPO_ROOT``.
    """
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path
