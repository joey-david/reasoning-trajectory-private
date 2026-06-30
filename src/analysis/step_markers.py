"""Apply configured token selectors to generations and persist reusable step-marker indices."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.analysis.common import read_generation_rows
from src.analysis.token_selectors import build_token_selector


DEFAULT_MARKER_SELECTORS: dict[str, dict[str, Any]] = {
    "every_8": {"every_n": 8},
    "sentence_end": {"mode": "sentence_end"},
    "deciles": {"mode": "percentiles", "count": 10},
    "reasoning_boundaries": {"mode": "reasoning_boundaries"},
    "first_last": {"mode": "first_last"},
}


def write_step_markers(run_path: Path, cfg: dict[str, Any]) -> None:
    """Write selector-derived token markers for every generation.

    Args:
        run_path: Completed run folder to analyze.
        cfg: Analysis configuration containing optional token selectors.

    Returns:
        None; writes ``analysis/step_markers.json``.
    """
    rows = read_generation_rows(run_path)
    selectors = configured_selectors(cfg)
    out_dir = run_path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row in rows:
        token_count = len(row.get("generated_token_ids", []))
        sample_record: dict[str, Any] = {
            "sample_id": row.get("sample_id"),
            "seed": row.get("seed"),
            "token_count": token_count,
            "reasoning_length": row.get("reasoning_length"),
            "selectors": {},
        }
        for name, spec in selectors.items():
            selector = build_token_selector(spec)
            sample_record["selectors"][name] = selector(row)
        records.append(sample_record)

    write_json(
        out_dir / "step_markers.json", {"selectors": selectors, "records": records}
    )


def configured_selectors(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve named token selectors from analysis configuration.

    Args:
        cfg: Analysis configuration with modern plural or legacy singular options.

    Returns:
        Named selector specifications, including defaults where appropriate.
    """
    selectors = cfg.get("token_selectors")
    if isinstance(selectors, dict) and selectors:
        return {str(name): dict(spec or {}) for name, spec in selectors.items()}

    selector = cfg.get("token_selector")
    if isinstance(selector, dict) and selector:
        merged = dict(DEFAULT_MARKER_SELECTORS)
        merged["configured"] = dict(selector)
        return merged

    return dict(DEFAULT_MARKER_SELECTORS)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Serialize one dictionary as indented JSON with a trailing newline.

    Args:
        path: Existing-parent destination path.
        obj: JSON-compatible dictionary.

    Returns:
        None.
    """
    import json

    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
