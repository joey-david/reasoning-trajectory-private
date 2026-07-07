"""Concise summaries and qualitative examples for extraction runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import ObjectRecord


def build_object_records(
    *,
    records: list[dict[str, Any]],
    token_ranges: np.ndarray,
    layer: int,
    projection_id: str,
    per_record: dict[int, dict[str, Any]],
    model: str,
) -> list[dict[str, Any]]:
    """Build the stable metadata rows aligned to object vectors."""
    output = []
    for index, row in enumerate(records):
        metrics = per_record.get(index, {})
        record = ObjectRecord(
            record_id=row["record_id"],
            model=model,
            layer=layer,
            trace_id=row["trace_id"],
            token_start=int(token_ranges[index, 0]),
            token_end=int(token_ranges[index, 1]),
            vector_index=index,
            projection_id=projection_id,
            canonical_graph_id=row["canonical_graph_id"],
            gold_graph_id=row["gold_graph_id"],
            edit_id=row["edit_id"],
            edit_type=row["edit_type"],
            split=row["split"],
            question_id=row["question_id"],
            surface=row["surface"],
            expected=row["expected"],
            observed=row["observed"],
            is_correct=bool(row["is_correct"]),
            hard_negative_type=row.get("hard_negative_type"),
            metrics={
                "retrieval_margin": metrics.get("retrieval_margin"),
                "object_decoder_confidence": None,
                "lexical_leakage_score": None,
            },
        )
        output.append(record.to_record())
    return output


def retrieval_examples(
    records: list[dict[str, Any]],
    per_record: dict[int, dict[str, Any]],
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Return a balanced slice of successes and failures for manual audit."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, metrics in per_record.items():
        row = records[index]
        example = {
            "record_id": row["record_id"],
            "split": row["split"],
            "anchor_text": row["anchor_text"],
            "canonical_graph_id": row["canonical_graph_id"],
            "retrieved_graph_id": metrics["retrieved_graph_id"],
            "retrieval_margin": metrics["retrieval_margin"],
            "lexical_family": row["surface"]["lexical_family"],
            "template_id": row["surface"]["template_id"],
            "edit_type": row["edit_type"],
            "is_correct_trace": row["is_correct"],
        }
        target = (
            successes
            if metrics["retrieved_graph_id"] == row["canonical_graph_id"]
            else failures
        )
        target.append(example)
    half = limit // 2
    return failures[:half] + successes[: limit - min(half, len(failures))]


def overall_summary(
    *,
    retrieval: dict[str, Any],
    rsa: dict[str, Any],
    decoder: dict[str, Any],
    patching: dict[str, Any] | None,
) -> dict[str, Any]:
    """Summarize prespecified positive-result checks without overclaiming."""
    selected = retrieval["selected"]
    object_vocab = selected["object"]["heldout_vocab"]["top1"]
    lexical_vocab = selected["lexical"]["heldout_vocab"]["top1"]
    checks = {
        "retrieval_beats_lexical_heldout_vocab": object_vocab > lexical_vocab,
        "rsa_object_exceeds_lexical": rsa["object_minus_lexical"] > 0,
        "decoder_operation_macro_f1_at_least_0_7": (
            decoder["heldout_vocab"]["object"]["operation"]["macro_f1"] >= 0.7
        ),
    }
    if patching is not None:
        object_cells = [
            cell
            for cell in patching["cells"]
            if cell["mode"] == "object_subspace"
            and cell["condition"] != "same_object_different_wording"
        ]
        random_cells = [
            cell
            for cell in patching["cells"]
            if cell["mode"] == "random_subspace"
            and cell["condition"] != "same_object_different_wording"
        ]
        checks["object_patch_exceeds_random"] = (
            sum(
                cell["mean_type_corrected_donor_probability_change"]
                for cell in object_cells
            )
            / max(len(object_cells), 1)
            > sum(
                cell["mean_type_corrected_donor_probability_change"]
                for cell in random_cells
            )
            / max(len(random_cells), 1)
        )
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "interpretation": (
            "Controlled-pilot checks only. Medium sampled traces and continuation "
            "interventions remain necessary before a scientific claim."
        ),
    }
