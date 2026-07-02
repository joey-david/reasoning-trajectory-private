"""Build a compact sentence-level benchmark of typed solution-object edits."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np

from src.analysis.common import read_generation_rows, read_sample_records
from src.analysis.step_classification.segmentation import sentence_spans
from src.experiments.solution_object_edits import admissible_bronze_update, typed_edit
from src.experiments.objective_segmentation import (
    normalized_regret,
    objective_partition,
    typed_object_costs,
)
from src.experiments.sentence_lattice import random_boundaries
from src.runtime.data import load_samples, write_jsonl


_WORD_EQUATION_RE = re.compile(
    r"\d[^.\n]{0,50}\b(?:is|equals|gives)\s+-?\d",
    re.I,
)
_ANSWER_STATEMENT_RE = re.compile(
    r"\b(?:answer|result|therefore)\b[^.\n]{0,50}-?\d",
    re.I,
)


def build_solution_object_benchmark(
    run_path: Path,
    updates_path: Path,
    out_dir: Path,
    *,
    question_limit: int = 50,
    audit_size: int = 120,
    partitions_path: Path | None = None,
) -> Path:
    """Build bronze sentence labels and a stratified pending gold audit queue.

    Args:
        run_path: Completed GSM-Symbolic run containing generation rows.
        updates_path: H2 symbolic update JSONL artifact.
        out_dir: Directory receiving benchmark artifacts.
        question_limit: Number of distinct questions to retain.
        audit_size: Number of sentence records sampled for human audit.
        partitions_path: Optional OCS partitions to score on the typed objective.

    Returns:
        Path to the benchmark metadata report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_traces = None
    if partitions_path is not None:
        allowed_traces = {
            (str(row["sample_id"]), int(row["seed"]))
            for row in load_samples(partitions_path.resolve())
        }
    selected = select_shortest_correct_traces(
        run_path,
        question_limit,
        allowed_traces=allowed_traces,
    )
    sample_records = read_sample_records(run_path)
    updates = load_samples(updates_path.resolve())
    by_trace = index_unique_updates(updates)
    sentences: list[dict[str, Any]] = []
    edit_counts: Counter[str] = Counter()

    for row in selected:
        key = (str(row["sample_id"]), int(row["seed"]))
        text = str(row.get("produced_text", ""))
        spans = sentence_spans(text)
        trace_updates = by_trace.get(key, [])
        edits = []
        before_state = ""
        for update in trace_updates:
            if not admissible_bronze_update(update):
                continue
            edit = typed_edit(update, before_state)
            edits.append(edit)
            before_state = edit.after_state
            edit_counts[edit.edit_type] += 1
        by_sentence: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
        for edit in edits:
            sentence_index = sentence_for_offset(spans, edit.char_end - 1)
            by_sentence[sentence_index].append(edit.to_record())
        sample = sample_records.get(key[0], {})
        question = str(sample.get("question") or sample.get("prompt") or "")
        for sentence_index, (start, end) in enumerate(spans):
            sentences.append(
                {
                    "record_id": f"{key[0]}::{key[1]}::{sentence_index}",
                    "sample_id": key[0],
                    "seed": key[1],
                    "sentence_index": sentence_index,
                    "sentence_count": len(spans),
                    "question": question,
                    "text": text[start:end],
                    "char_start": start,
                    "char_end": end,
                    "bronze_edits": by_sentence[sentence_index],
                    "bronze_tier": "deterministic_verified_arithmetic",
                    "is_correct": bool(row.get("is_correct")),
                    "hidden_states_file": row.get("hidden_states_file"),
                }
            )

    bronze_path = out_dir / "bronze_sentences.jsonl"
    write_jsonl(bronze_path, sentences)
    audit_rows = stratified_audit_sample(sentences, audit_size)
    write_jsonl(out_dir / "gold_audit_queue.jsonl", audit_rows)
    report = {
        "benchmark": "SO-GSM",
        "version": 1,
        "source_run": run_path.as_posix(),
        "source_updates": updates_path.as_posix(),
        "selection": "shortest correct activation-bearing trace per question",
        "questions": len({row["sample_id"] for row in sentences}),
        "traces": len(selected),
        "sentence_candidates": len(sentences),
        "sentences_with_edits": sum(bool(row["bronze_edits"]) for row in sentences),
        "typed_edits": sum(edit_counts.values()),
        "edit_type_counts": dict(sorted(edit_counts.items())),
        "coverage_diagnostics": {
            "unedited_word_equation_candidates": sum(
                not row["bronze_edits"] and bool(_WORD_EQUATION_RE.search(row["text"]))
                for row in sentences
            ),
            "unedited_answer_statement_candidates": sum(
                not row["bronze_edits"]
                and bool(_ANSWER_STATEMENT_RE.search(row["text"]))
                for row in sentences
            ),
            "interpretation": (
                "Heuristic candidates are not gold false negatives; they quantify "
                "why semantic silver labeling is needed."
            ),
        },
        "bronze": bronze_path.as_posix(),
        "gold_audit_queue": (out_dir / "gold_audit_queue.jsonl").as_posix(),
        "gold_status": "pending_human_audit",
    }
    if partitions_path is not None:
        report["typed_object_evaluation"] = evaluate_typed_object_partitions(
            sentences,
            partitions_path,
        )
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def evaluate_typed_object_partitions(
    sentences: list[dict[str, Any]],
    partitions_path: Path,
) -> dict[str, Any]:
    """Score existing OCS partitions against typed object-edit purity.

    Args:
        sentences: Bronze benchmark sentence records.
        partitions_path: Persisted sentence-lattice partition JSONL.

    Returns:
        Mean normalized regret by partition method on the typed object objective.
    """
    partition_rows = load_samples(partitions_path.resolve())
    partitions = {
        (str(row["sample_id"]), int(row["seed"])): row["methods"]
        for row in partition_rows
    }
    by_trace: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in sentences:
        by_trace[(str(row["sample_id"]), int(row["seed"]))].append(row)
    edit_types = ("bind_variable", "derive_value", "verify", "extract_answer")
    regrets: defaultdict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(73)
    evaluated = 0
    for key, rows in by_trace.items():
        if key not in partitions:
            continue
        rows.sort(key=lambda row: int(row["sentence_index"]))
        counts = np.asarray(
            [
                [
                    sum(edit["edit_type"] == edit_type for edit in row["bronze_edits"])
                    for edit_type in edit_types
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        costs = typed_object_costs(counts)
        first_method = next(iter(partitions[key].values()))
        count = len(first_method)
        oracle = objective_partition(costs, count)
        controls = [random_boundaries(len(rows), count, rng) for _ in range(24)]
        for method, boundaries in partitions[key].items():
            regret = normalized_regret(
                costs,
                np.asarray(boundaries, dtype=np.int32),
                oracle,
                controls,
            )
            if regret is not None:
                regrets[method].append(regret)
        regrets["oracle_typed_object"].append(0.0)
        evaluated += 1
    return {
        "traces": evaluated,
        "edit_types": list(edit_types),
        "cost": ("(edit_count-1)^2 + mixed_type_penalty + 0.25*multi_edit_penalty"),
        "mean_normalized_regret": {
            method: float(np.mean(values)) for method, values in sorted(regrets.items())
        },
    }


def select_shortest_correct_traces(
    run_path: Path,
    question_limit: int,
    *,
    allowed_traces: set[tuple[str, int]] | None = None,
) -> list[dict[str, Any]]:
    """Select one concise correct activation-bearing trace per question.

    Args:
        run_path: Completed run containing generation rows.
        question_limit: Maximum number of questions to retain.
        allowed_traces: Optional sample/seed keys required by downstream artifacts.

    Returns:
        Selected generation rows ordered by trace sentence count.
    """
    by_question: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_generation_rows(run_path):
        key = (str(row["sample_id"]), int(row["seed"]))
        if (
            row.get("hidden_states_file")
            and row.get("is_correct")
            and (allowed_traces is None or key in allowed_traces)
        ):
            by_question[str(row["sample_id"])].append(row)
    candidates = [
        min(
            rows,
            key=lambda row: len(sentence_spans(str(row.get("produced_text", "")))),
        )
        for rows in by_question.values()
    ]
    candidates.sort(
        key=lambda row: (
            len(sentence_spans(str(row.get("produced_text", "")))),
            str(row["sample_id"]),
        )
    )
    return candidates[:question_limit]


def index_unique_updates(
    updates: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Deduplicate layer-repeated symbolic updates and index them by trace.

    Args:
        updates: H2 update records, potentially repeated across layers.

    Returns:
        Ordered unique update records keyed by sample ID and seed.
    """
    indexed: defaultdict[tuple[str, int], dict[tuple[Any, ...], dict[str, Any]]] = (
        defaultdict(dict)
    )
    for update in updates:
        trace = (str(update["sample_id"]), int(update["seed"]))
        identity = (
            int(update["char_start"]),
            int(update["char_end"]),
            str(update["operator"]),
            str(update["expression"]),
            float(update["value"]),
        )
        indexed[trace].setdefault(identity, update)
    return {
        trace: sorted(
            records.values(), key=lambda row: (row["char_end"], row["char_start"])
        )
        for trace, records in indexed.items()
    }


def sentence_for_offset(spans: list[tuple[int, int]], offset: int) -> int:
    """Locate the sentence containing a character offset.

    Args:
        spans: Ordered start-inclusive, end-exclusive sentence spans.
        offset: Character offset to locate.

    Returns:
        Index of the containing or nearest final sentence.
    """
    for index, (start, end) in enumerate(spans):
        if start <= offset < end:
            return index
    return max(len(spans) - 1, 0)


def stratified_audit_sample(
    sentences: list[dict[str, Any]],
    audit_size: int,
) -> list[dict[str, Any]]:
    """Sample edited and unedited sentences for reproducible human auditing.

    Args:
        sentences: Bronze sentence records.
        audit_size: Maximum number of audit records.

    Returns:
        Pending audit records stratified by bronze edit type.
    """
    strata: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sentences:
        labels = {edit["edit_type"] for edit in row["bronze_edits"]} or {"none"}
        for label in labels:
            strata[label].append(row)
    rng = random.Random(42)
    selected: dict[str, dict[str, Any]] = {}
    labels = sorted(strata)
    quota = max(audit_size // max(len(labels), 1), 1)
    for label in labels:
        rows = list(strata[label])
        rng.shuffle(rows)
        for row in rows[:quota]:
            selected[row["record_id"]] = row
    remaining = [row for row in sentences if row["record_id"] not in selected]
    rng.shuffle(remaining)
    for row in remaining:
        if len(selected) >= audit_size:
            break
        selected[row["record_id"]] = row
    return [
        {
            **row,
            "audit_status": "pending",
            "audited_edits": None,
            "auditor_notes": None,
        }
        for row in list(selected.values())[:audit_size]
    ]
