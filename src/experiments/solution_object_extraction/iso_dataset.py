"""Deterministic isomorphic arithmetic bank with strict surface splits."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .schemas import CanonicalGraph, make_graph


OPERATIONS = ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE")
EDIT_TYPES = ("BIND", "OPERATE", "VERIFY", "EXTRACT")

VOCABULARIES = {
    "stationery": ("boxes", "pens"),
    "produce": ("crates", "apples"),
    "hardware": ("cases", "screws"),
    "astronomy": ("clusters", "stars"),
    "textiles": ("bundles", "threads"),
    "ceramics": ("trays", "tiles"),
}

SPLIT_SURFACES = {
    "train": (
        ("stationery", "direct"),
        ("produce", "direct"),
        ("stationery", "relational"),
        ("produce", "relational"),
    ),
    "validation": (("hardware", "direct"), ("hardware", "relational")),
    "heldout_vocab": (("astronomy", "direct"), ("astronomy", "relational")),
    "heldout_template": (
        ("stationery", "inverted"),
        ("produce", "inverted"),
    ),
    "heldout_question": (("stationery", "narrative"),),
}


def arithmetic_result(operation: str, a: int, b: int) -> int:
    """Evaluate one controlled integer operation."""
    if operation == "ADD":
        return a + b
    if operation == "SUBTRACT":
        return a - b
    if operation == "MULTIPLY":
        return a * b
    if operation == "DIVIDE":
        if b == 0 or a % b:
            raise ValueError("DIVIDE bank values must divide exactly")
        return a // b
    raise ValueError(f"Unsupported operation: {operation}")


def graph_specs(count: int) -> list[tuple[str, int, int]]:
    """Return balanced, collision-rich operation/value specifications."""
    seeds = {
        "ADD": [(3, 4), (4, 3), (5, 7), (8, 4), (9, 6), (11, 5)],
        "SUBTRACT": [(7, 3), (9, 4), (12, 5), (15, 7), (14, 6), (18, 9)],
        "MULTIPLY": [(3, 4), (4, 3), (3, 5), (2, 6), (4, 5), (6, 3)],
        "DIVIDE": [(12, 3), (12, 4), (15, 3), (18, 6), (20, 4), (24, 6)],
    }
    ordered: list[tuple[str, int, int]] = []
    for index in range(max(len(values) for values in seeds.values())):
        for operation in OPERATIONS:
            values = seeds[operation]
            if index < len(values):
                ordered.append((operation, *values[index]))
    if count > len(ordered):
        raise ValueError(f"graph_count may not exceed {len(ordered)}")
    return ordered[:count]


def build_bank(
    *,
    graph_count: int,
    include_corruptions: bool,
    splits: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical graphs and surface-instantiated trajectory states."""
    selected_splits = splits or list(SPLIT_SURFACES)
    records: list[dict[str, Any]] = []
    graphs: dict[str, CanonicalGraph] = {}
    for graph_index, (operation, a, b) in enumerate(graph_specs(graph_count)):
        result = arithmetic_result(operation, a, b)
        gold_id = state_graph_id(operation, a, b, result)
        bound_id = state_graph_id("NONE", a, b, None)
        graphs[gold_id] = make_graph(
            graph_id=gold_id,
            operation=operation,
            operand_a=a,
            operand_b=b,
            result=result,
        )
        graphs[bound_id] = make_graph(
            graph_id=bound_id,
            operation="NONE",
            operand_a=a,
            operand_b=b,
            result=None,
        )
        for split in selected_splits:
            for vocab, template in SPLIT_SURFACES[split]:
                surface_id = f"g{graph_index:03d}-{vocab}-{template}"
                records.extend(
                    trajectory_records(
                        graph_index=graph_index,
                        operation=operation,
                        a=a,
                        b=b,
                        result=result,
                        split=split,
                        vocab=vocab,
                        template=template,
                        surface_id=surface_id,
                        corrupt_delta=0,
                    )
                )
                if include_corruptions and split in {
                    "train",
                    "validation",
                    "heldout_vocab",
                    "heldout_template",
                }:
                    for delta in (-1, 1):
                        corrupt_result = result + delta
                        corrupt_id = state_graph_id(operation, a, b, corrupt_result)
                        graphs[corrupt_id] = make_graph(
                            graph_id=corrupt_id,
                            operation=operation,
                            operand_a=a,
                            operand_b=b,
                            result=corrupt_result,
                        )
                        records.extend(
                            trajectory_records(
                                graph_index=graph_index,
                                operation=operation,
                                a=a,
                                b=b,
                                result=result,
                                split=split,
                                vocab=vocab,
                                template=template,
                                surface_id=surface_id,
                                corrupt_delta=delta,
                                edit_types=("OPERATE", "VERIFY", "EXTRACT"),
                            )
                        )
    annotate_hard_negatives(records)
    graph_rows = [graphs[key].to_record() for key in sorted(graphs)]
    return graph_rows, records


def trajectory_records(
    *,
    graph_index: int,
    operation: str,
    a: int,
    b: int,
    result: int,
    split: str,
    vocab: str,
    template: str,
    surface_id: str,
    corrupt_delta: int,
    edit_types: tuple[str, ...] = EDIT_TYPES,
) -> list[dict[str, Any]]:
    """Instantiate one ordered correct or corrupted object trajectory."""
    observed_result = result + corrupt_delta
    problem = render_problem(operation, a, b, vocab, template)
    anchors = render_anchors(operation, a, b, observed_result, vocab, template)
    prefix = problem + "\nReasoning:\n"
    rows: list[dict[str, Any]] = []
    accumulated: list[str] = []
    suffix = "correct" if corrupt_delta == 0 else f"corrupt{corrupt_delta:+d}"
    for edit_index, edit_type in enumerate(EDIT_TYPES):
        accumulated.append(anchors[edit_type])
        if edit_type not in edit_types:
            continue
        text = prefix + " ".join(accumulated)
        anchor = anchors[edit_type]
        graph_id = (
            state_graph_id("NONE", a, b, None)
            if edit_type == "BIND"
            else state_graph_id(operation, a, b, observed_result)
        )
        gold_graph_id = (
            state_graph_id("NONE", a, b, None)
            if edit_type == "BIND"
            else state_graph_id(operation, a, b, result)
        )
        trace_id = f"{surface_id}-{suffix}"
        rows.append(
            {
                "record_id": f"{trace_id}-{edit_type.lower()}",
                "trace_id": trace_id,
                "question_id": f"iso_{graph_index:03d}",
                "split": split,
                "text": text,
                "anchor_text": anchor,
                "canonical_graph_id": graph_id,
                "gold_graph_id": gold_graph_id,
                "edit_id": f"{trace_id}-edit{edit_index}",
                "edit_type": edit_type,
                "is_correct": corrupt_delta == 0,
                "corrupt_delta": corrupt_delta,
                "expected": {
                    "operation": operation if edit_type != "BIND" else "NONE",
                    "operand_a": a,
                    "operand_b": b,
                    "result": result if edit_type != "BIND" else None,
                    "target": "result" if edit_type != "BIND" else "operand_b",
                },
                "observed": {
                    "operation": operation if edit_type != "BIND" else "NONE",
                    "operand_a": a,
                    "operand_b": b,
                    "result": observed_result if edit_type != "BIND" else None,
                    "target": "result" if edit_type != "BIND" else "operand_b",
                },
                "surface": {
                    "template_id": template,
                    "lexical_family": vocab,
                    "language": "en",
                    "container_term": VOCABULARIES[vocab][0],
                    "item_term": VOCABULARIES[vocab][1],
                },
                "causal_prefix": render_causal_prefix(operation, a, b, vocab, template),
                "causal_result": result,
            }
        )
    return rows


def state_graph_id(
    operation: str, a: int, b: int, result: int | None
) -> str:
    """Return a readable unique state identifier."""
    tail = "bound" if result is None else f"r{result}"
    return f"graph_{operation.lower()}_a{a}_b{b}_{tail}"


def render_problem(operation: str, a: int, b: int, vocab: str, template: str) -> str:
    """Render a surface-diverse question without changing the canonical roles."""
    containers, items = VOCABULARIES[vocab]
    if operation == "MULTIPLY":
        core = f"{a} {containers} hold {b} {items} each"
    elif operation == "ADD":
        core = f"one shelf has {a} {items} and another has {b} more"
    elif operation == "SUBTRACT":
        core = f"a stock of {a} {items} loses {b} {items}"
    else:
        core = f"{a} {items} are shared equally among {b} {containers}"
    if template == "inverted":
        return f"Find the resulting number of {items}; the situation is: {core}."
    if template == "relational":
        return f"Use the ordered quantities in this relation: {core}. What follows?"
    if template == "narrative":
        return f"During an inventory check, {core}. What quantity results?"
    return f"{core.capitalize()}. What is the resulting quantity?"


def render_anchors(
    operation: str,
    a: int,
    b: int,
    result: int,
    vocab: str,
    template: str,
) -> dict[str, str]:
    """Render cumulative edit sentences with vocabulary-dependent wording."""
    containers, items = VOCABULARIES[vocab]
    connector = {
        "ADD": "combining",
        "SUBTRACT": "removing",
        "MULTIPLY": "scaling",
        "DIVIDE": "sharing",
    }[operation]
    if template == "inverted":
        bind = f"The relevant quantities are {b} for the second role and {a} first."
        operate = f"Putting them in role order, {a} then {b}, gives {result}."
    elif template == "relational":
        bind = f"Read by role rather than mention order: first is {a}, while second is {b}."
        operate = f"The relation applied to first={a} and second={b} yields {result}."
    else:
        bind = f"Bind the first quantity to {a} {items} and the second to {b} {containers}."
        operate = f"By {connector} the role values in order, the result is {result} {items}."
    return {
        "BIND": bind,
        "OPERATE": operate,
        "VERIFY": f"Checking the same relation confirms the value {result}.",
        "EXTRACT": f"Therefore the requested quantity is {result}.",
    }


def render_causal_prefix(
    operation: str, a: int, b: int, vocab: str, template: str
) -> str:
    """Render a prompt whose next token should be the arithmetic result."""
    problem = render_problem(operation, a, b, vocab, template)
    symbols = {
        "ADD": "+",
        "SUBTRACT": "-",
        "MULTIPLY": "*",
        "DIVIDE": "/",
    }
    return (
        f"{problem}\nThe first role is {a}; the second role is {b}. "
        f"The final computation is {a} {symbols[operation]} {b} = "
    )


def annotate_hard_negatives(records: list[dict[str, Any]]) -> None:
    """Mark graph-level hard-negative categories without changing records."""
    full = [row for row in records if row["edit_type"] == "OPERATE"]
    by_answer: defaultdict[float, set[str]] = defaultdict(set)
    by_template: defaultdict[str, set[str]] = defaultdict(set)
    for row in full:
        by_answer[float(row["observed"]["result"])].add(row["canonical_graph_id"])
        by_template[row["surface"]["template_id"]].add(row["canonical_graph_id"])
    for row in records:
        graph_id = row["canonical_graph_id"]
        result = row["observed"]["result"]
        if result is not None and len(by_answer[float(result)]) > 1:
            row["hard_negative_type"] = "same_answer_different_graph"
        elif len(by_template[row["surface"]["template_id"]]) > 1:
            row["hard_negative_type"] = "same_template_different_graph"
        else:
            row["hard_negative_type"] = None
