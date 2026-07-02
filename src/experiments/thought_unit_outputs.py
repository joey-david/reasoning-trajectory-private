"""Serialize and plot sentence-lattice experiment results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.step_classification.segmentation import build_segments
from src.analysis.token_alignment import build_token_spans
from src.experiments.common import balanced_generation_rows
from src.experiments.thought_unit_types import OBJECTIVES, PRIMARY_FRACTION
from src.runtime.data import write_jsonl


def write_matrix_csv(
    path: Path,
    matrix: dict[str, dict[str, float]],
) -> None:
    """Write a method-by-objective matrix with stable column ordering.

    Args:
        path: Filesystem path to read from or write to.
        matrix: Nested row/column metric mapping to serialize.

    Returns:
        None.
    """
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *OBJECTIVES])
        writer.writeheader()
        for method in sorted(matrix):
            writer.writerow({"method": method, **matrix[method]})


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous report records to CSV.

    Args:
        path: Filesystem path to read from or write to.
        rows: Generation or analysis records to process.

    Returns:
        None.
    """
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_boundary_examples(
    run_path: Path,
    path: Path,
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
    *,
    trace_limit: int = 8,
    boundaries_per_method: int = 8,
) -> None:
    """Write sentence text around representative held-out boundaries.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        path: Filesystem path to read from or write to.
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.
        primary_partitions: Reference objective-specific partitions keyed by trace.
        trace_limit: Maximum number of traces included in the example artifact.
        boundaries_per_method: Maximum example boundaries retained per method.

    Returns:
        None.
    """
    chosen: list[int] = []
    seen_questions: set[str] = set()
    for index in selected_indices:
        record = cache["records"][index]
        question = str(record["sample_id"])
        if record["train"] or question in seen_questions:
            continue
        chosen.append(index)
        seen_questions.add(question)
        if len(chosen) >= trace_limit:
            break

    rows = balanced_generation_rows(run_path, per_sample=10)
    row_lookup = {(str(row["sample_id"]), int(row["seed"])): row for row in rows}
    chosen_rows = [
        row_lookup[
            (
                str(cache["records"][index]["sample_id"]),
                int(cache["records"][index]["seed"]),
            )
        ]
        for index in chosen
    ]
    token_spans = build_token_spans(run_path, chosen_rows)
    examples: list[dict[str, Any]] = []
    for index, row, spans in zip(chosen, chosen_rows, token_spans):
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        for method, boundaries in primary_partitions[index].items():
            keep = np.linspace(
                0,
                len(boundaries) - 1,
                min(len(boundaries), boundaries_per_method),
                dtype=int,
            )
            selected = boundaries[keep] if len(boundaries) else boundaries
            examples.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "seed": int(row["seed"]),
                    "method": method,
                    "sentence_count": len(segments),
                    "boundary_count": len(boundaries),
                    "examples": [
                        {
                            "boundary_after_sentence": int(boundary),
                            "position": float(boundary / max(len(segments) - 2, 1)),
                            "left": segments[int(boundary)].text[:300],
                            "right": segments[int(boundary) + 1].text[:300],
                        }
                        for boundary in selected
                        if int(boundary) + 1 < len(segments)
                    ],
                }
            )
    write_jsonl(path, examples)


def write_partitions(
    path: Path,
    cache: dict[str, Any],
    selected_indices: list[int],
    partitions: dict[int, dict[str, np.ndarray]],
) -> None:
    """Persist every primary matched-budget boundary set for audit and reuse.

    Args:
        path: Filesystem path to read from or write to.
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.
        partitions: Partitions keyed by trace and method.

    Returns:
        None.
    """
    rows = []
    for index in selected_indices:
        record = cache["records"][index]
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "seed": int(record["seed"]),
                "train": bool(record["train"]),
                "sentence_count": int(record["sentences"]),
                "boundary_fraction": PRIMARY_FRACTION,
                "methods": {
                    method: values.astype(int).tolist()
                    for method, values in partitions[index].items()
                },
            }
        )
    write_jsonl(path, rows)


def write_plots(
    out_dir: Path,
    utilities: dict[str, dict[str, float]],
    front: list[str],
) -> None:
    """Render compact utility, regret, and two-objective Pareto figures.

    Args:
        out_dir: Directory in which to write the results.
        utilities: Utility scores for candidate segmentations.
        front: Indices of methods on the Pareto frontier.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    methods = sorted(utilities)
    values = np.asarray(
        [
            [utilities[method][objective] for objective in OBJECTIVES]
            for method in methods
        ]
    )
    for filename, matrix, title, cmap, norm in (
        (
            "objective_matrix.png",
            values,
            "Normalized utility (random=0, oracle=1)",
            "RdYlGn",
            TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        ),
        (
            "regret_matrix.png",
            1.0 - values,
            "Normalized regret (oracle=0)",
            "magma_r",
            Normalize(vmin=0.0, vmax=2.0, clip=True),
        ),
    ):
        fig, axis = plt.subplots(figsize=(7.2, max(3.8, 0.36 * len(methods))))
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(OBJECTIVES)), OBJECTIVES, rotation=20)
        axis.set_yticks(range(len(methods)), methods)
        axis.set_title(title)
        for row in range(len(methods)):
            for column in range(len(OBJECTIVES)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for method in methods:
        axis.scatter(
            utilities[method]["object"],
            utilities[method]["compression"],
            marker="o" if method in front else "x",
        )
        axis.annotate(
            method,
            (
                utilities[method]["object"],
                utilities[method]["compression"],
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set_xlabel("Object utility")
    axis.set_ylabel("Compression utility")
    axis.set_title("Matched-budget Pareto slice")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_object_compression.png", dpi=180)
    plt.close(fig)
