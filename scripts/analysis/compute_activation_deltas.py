#!/usr/bin/env python3
"""Post-process runs with hidden states: compute activation_delta (L2 norm of
consecutive hidden-state differences) and patch timesteps into generations.jsonl.

Usage:
    # Single run
    python3 scripts/analysis/compute_activation_deltas.py runs/SmolLM3-3B/screening/frontier_identification/gsm_symb_40_60_full_token_mlx

    # Runs with activations under a prefix
    scripts/analysis/run_analysis_for_all_activation_runs.sh

    # All runs with hidden_states directories
    for d in runs/*/*/generation; do
        [ -d "$d/hidden_states" ] && python3 scripts/analysis/compute_activation_deltas.py "$(dirname "$(dirname "$d")")"
    done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def dequantize_hidden_states(
    q: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Reconstruct approximate float32 hidden states from int8 quantized storage.

    Args:
        q: int8 array, shape ``(tokens, layers, hidden_dim)``.
        scale: float32 array, shape ``(tokens, layers)`` or ``(tokens, layers, 1)``.

    Returns:
        float32 array, same shape as ``q``.
    """
    if scale.ndim == 2:
        scale = scale[..., None]  # (tokens, layers) -> (tokens, layers, 1)
    return q.astype(np.float32) * scale


def compute_activation_deltas(
    hidden: np.ndarray, gen_count: int
) -> list[float]:
    """Compute L2 norm of consecutive hidden-state differences for generated tokens.

    Supports two storage formats:
    1. ``hidden`` is the full sequence including prompt (``shape[0] > gen_count``).
       First generated delta compares against the last prompt hidden state.
    2. ``hidden`` is generated tokens only (``shape[0] == gen_count``).
       First delta is 0.0 (no predecessor).

    Args:
        hidden: float32 array, shape ``(tokens, layers, hidden_dim)``.
        gen_count: Number of generated tokens.

    Returns:
        One activation_delta per generated token.
    """
    total_len = hidden.shape[0]
    if total_len == gen_count:
        # Generated tokens only — first token has no predecessor
        flat = hidden.reshape(total_len, -1)
        diffs = np.linalg.norm(flat[1:] - flat[:-1], axis=1)
        return [0.0] + [float(d) for d in diffs]

    # Full sequence including prompt
    generated = hidden[gen_count:]
    prev = np.concatenate(
        [hidden[gen_count - 1 : gen_count], generated[:-1]]
    )
    diffs = np.linalg.norm(
        (generated - prev).reshape(gen_count, -1), axis=1
    )
    return [float(d) for d in diffs]


def resolve_run_path(run_path: Path) -> Path | None:
    """Find the generation directory for a run path."""
    generation_dir = run_path / "generation"
    hidden_dir = generation_dir / "hidden_states"
    if not hidden_dir.is_dir():
        print(f"  [skip] no hidden_states directory: {hidden_dir}", file=sys.stderr)
        return None
    jsonl_path = generation_dir / "generations.jsonl"
    if not jsonl_path.exists():
        print(f"  [skip] no generations.jsonl: {jsonl_path}", file=sys.stderr)
        return None
    return jsonl_path


def load_generations(jsonl_path: Path) -> list[dict]:
    """Load all generation records from a generations.jsonl file."""
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_generations(rows: list[dict], jsonl_path: Path) -> None:
    """Write generation records back to generations.jsonl."""
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_metadata(run_path: Path) -> None:
    """Add 'activation_delta' to metadata timestep_metrics if not present."""
    meta_path = run_path / "generation" / "metadata.json"
    if not meta_path.exists():
        return
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    metrics = meta.get("timestep_metrics", [])
    if "activation_delta" not in metrics:
        metrics.append("activation_delta")
        meta["timestep_metrics"] = metrics
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  updated metadata: added activation_delta to timestep_metrics")


def process_run(run_path: Path) -> int:
    """Process one run directory. Returns count of rows updated."""
    jsonl_path = resolve_run_path(run_path)
    if jsonl_path is None:
        return 0

    rows = load_generations(jsonl_path)
    if not rows:
        print(f"  [skip] empty generations.jsonl")
        return 0

    hidden_dir = run_path / "generation" / "hidden_states"
    npz_files = sorted(hidden_dir.glob("*.npz"))

    if not npz_files:
        print(f"  [skip] no NPZ files in {hidden_dir}")
        return 0

    # Build lookup: filename stem -> generation row
    row_by_stem: dict[str, dict] = {}
    for row in rows:
        hf = row.get("hidden_states_file", "")
        # hidden_states_file can be basename or relative path
        stem = Path(hf).stem
        if stem:
            row_by_stem[stem] = row

    updated = 0
    for npz_path in tqdm(npz_files, desc=f"  {run_path.name}", unit="npz", leave=False):
        stem = npz_path.stem
        row = row_by_stem.get(stem)
        if row is None:
            # Try matching by first part of stem before __seed
            short_stem = stem.split("__seed")[0] if "__seed" in stem else stem
            candidates = [k for k in row_by_stem if short_stem in k]
            if len(candidates) == 1:
                row = row_by_stem[candidates[0]]
            elif len(candidates) > 1:
                # Try matching by both stem parts
                for candidate in candidates:
                    if stem.endswith(candidate.split("__seed")[-1]) if "__seed" in candidate else False:
                        row = row_by_stem[candidate]
                        break
                if row is None:
                    tqdm.write(f"  [warn] ambiguous match for {stem}, skipping")
                    continue
            else:
                tqdm.write(f"  [warn] no match for {stem}, skipping")
                continue

        data = np.load(npz_path)
        keys = list(data.keys())

        if "hidden_states" in keys:
            # float16 (or float32) format
            hidden = data["hidden_states"].astype(np.float32)
        elif "hidden_states_q" in keys and "hidden_states_scale" in keys:
            # quantized int8 format
            hidden = dequantize_hidden_states(
                data["hidden_states_q"], data["hidden_states_scale"]
            )
        else:
            tqdm.write(f"  [warn] unknown NPZ keys {keys} in {npz_path.name}, skipping")
            continue

        total_len = hidden.shape[0]
        gen_count = len(row.get("generated_token_ids", []))
        if gen_count == 0:
            tqdm.write(f"  [warn] no generated_token_ids for {stem}, skipping")
            continue
        prompt_len = total_len - gen_count

        deltas = compute_activation_deltas(hidden, gen_count)

        # Build or update timesteps
        timesteps = row.get("timesteps")
        if timesteps and len(timesteps) > 0:
            # Update existing timesteps
            if len(timesteps) != gen_count:
                # Timesteps may be per-layer or truncated; pad/truncate deltas
                delta_iter = iter(deltas[: len(timesteps)])
            else:
                delta_iter = iter(deltas)
            for ts in timesteps:
                ts["activation_delta"] = next(delta_iter)
        else:
            # Create timesteps from scratch
            timesteps = []
            generated_ids = row.get("generated_token_ids", [])
            for idx in range(gen_count):
                timesteps.append({
                    "token_idx": idx,
                    "activation_delta": deltas[idx] if idx < len(deltas) else 0.0,
                    "token_str": (
                        "<unknown>"
                        if not generated_ids or idx >= len(generated_ids)
                        else ""
                    ),
                })
            row["timesteps"] = timesteps

        # Remove the hidden_states_file reference so the web UI doesn't try to link it
        # (it's optional — activation_delta is the derived metric we care about)
        updated += 1

    if updated:
        save_generations(rows, jsonl_path)
        update_metadata(run_path)

    return updated


def discover_runs(base_paths: list[Path]) -> list[Path]:
    """Discover run directories with hidden_states under given base paths."""
    runs = []
    for base in base_paths:
        if not base.exists():
            print(f"path not found: {base}", file=sys.stderr)
            continue
        if (base / "generation" / "hidden_states").is_dir():
            runs.append(base)
        else:
            # Walk deeper
            for hidden_dir in base.rglob("generation/hidden_states"):
                run_path = hidden_dir.parent.parent
                if run_path not in runs:
                    runs.append(run_path)
    return sorted(runs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_paths",
        type=Path,
        nargs="*",
        default=[Path("runs")],
        help="Run directories or parent paths to scan (default: runs/)",
    )
    args = parser.parse_args()

    runs = discover_runs(args.run_paths if args.run_paths else [Path("runs")])
    if not runs:
        print("No runs with hidden_states found.")
        return 0

    total_updated = 0
    for run_path in runs:
        count = process_run(run_path)
        total_updated += count
        print(f"{run_path}: {count} rows updated")

    print(f"\nDone. {total_updated} total rows patched with activation_delta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
