"""Typed deterministic edits to a partial arithmetic solution object."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.experiments.symbolic import safe_arithmetic_eval, symbolic_relation_atom


_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
EDIT_TYPES = {
    "none",
    "add_entity",
    "add_quantity",
    "bind_variable",
    "add_relation",
    "derive_value",
    "verify",
    "extract_answer",
    "mixed",
}
OPERATOR_TO_EDIT = {
    "BIND": "bind_variable",
    "OPERATE": "derive_value",
    "VERIFY": "verify",
    "EXTRACT": "extract_answer",
}
_UNIT_BINDINGS = {
    "cm",
    "dollars",
    "euros",
    "feet",
    "grams",
    "hours",
    "inches",
    "kg",
    "kilograms",
    "km",
    "liters",
    "meters",
    "miles",
    "minutes",
    "pounds",
    "seconds",
}


@dataclass(slots=True)
class SolutionObjectEdit:
    """Represent one verified change to a partial mathematical solution."""

    edit_type: str
    operation: str
    before_state: str
    after_state: str
    added_relations: tuple[str, ...]
    removed_relations: tuple[str, ...]
    quantities: tuple[float, ...]
    verified: bool
    operator: str
    expression: str
    value: float
    char_start: int
    char_end: int
    token_start: int
    token_end: int

    def to_record(self) -> dict[str, Any]:
        """Serialize one typed edit as a JSON-compatible record.

        Args:
            None.

        Returns:
            The typed edit fields with tuples converted to lists.
        """
        return {
            "edit_type": self.edit_type,
            "operation": self.operation,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "added_relations": list(self.added_relations),
            "removed_relations": list(self.removed_relations),
            "quantities": list(self.quantities),
            "verified": self.verified,
            "operator": self.operator,
            "expression": self.expression,
            "value": self.value,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
        }


def typed_edit(
    update: dict[str, Any],
    before_state: str,
) -> SolutionObjectEdit:
    """Convert one verified symbolic update into a typed state diff.

    Args:
        update: Symbolic update record from H2 extraction.
        before_state: Canonical graph signature before this update.

    Returns:
        A deterministic typed solution-object edit.
    """
    before = graph_relations(before_state)
    operator = str(update["operator"])
    expression = str(update["expression"])
    if operator == "BIND":
        variable = expression.split("=", 1)[0].strip().lower()
        relation = f"BIND:{variable}:{float(update['value']):g}"
    else:
        relation = symbolic_relation_atom(
            operator,
            expression,
            float(update["value"]),
        )
    is_new_relation = operator != "VERIFY" and relation not in before
    after = before | ({relation} if is_new_relation else set())
    after_state = "|".join(sorted(after))
    lexical = update.get("lexical_items", [])
    quantities = tuple(
        sorted(
            {
                float(str(value).replace(",", ""))
                for value in lexical
                if _NUMBER_RE.fullmatch(str(value))
            }
        )
    )
    edit_type = (
        OPERATOR_TO_EDIT[operator]
        if is_new_relation or operator in {"VERIFY", "EXTRACT"}
        else "verify"
    )
    return SolutionObjectEdit(
        edit_type=edit_type,
        operation=str(update["operation_signature"]),
        before_state=before_state,
        after_state=after_state,
        added_relations=tuple(sorted(after - before)),
        removed_relations=tuple(sorted(before - after)),
        quantities=quantities,
        verified=True,
        operator=operator,
        expression=expression,
        value=float(update["value"]),
        char_start=int(update["char_start"]),
        char_end=int(update["char_end"]),
        token_start=int(update["token_start"]),
        token_end=int(update["token_end"]),
    )


def admissible_bronze_update(update: dict[str, Any]) -> bool:
    """Reject regex matches that cannot support a precise bronze edit.

    Args:
        update: Candidate symbolic update record.

    Returns:
        Whether the candidate is suitable for deterministic benchmark labeling.
    """
    if str(update.get("operator")) != "BIND":
        return True
    expression = str(update.get("expression", ""))
    left = expression.split("=", 1)[0].strip().lower()
    return left not in _UNIT_BINDINGS


def graph_relations(signature: str) -> set[str]:
    """Split a canonical graph signature into relation atoms.

    Args:
        signature: Pipe-delimited canonical graph signature.

    Returns:
        The nonempty relation atoms in the graph.
    """
    return {relation for relation in signature.split("|") if relation}


def validate_silver_label(
    sentence: str,
    label: dict[str, Any],
    bronze_edits: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Check an LLM-proposed object edit for schema and numeric grounding.

    Args:
        sentence: Sentence from which the proposal was inferred.
        label: Proposed structured silver label.
        bronze_edits: Deterministically verified edits anchored in the sentence.

    Returns:
        Validation errors; an empty list means the proposal is admissible.
    """
    errors: list[str] = []
    edit_type = label.get("edit_type")
    if edit_type not in EDIT_TYPES:
        errors.append(f"unsupported edit_type: {edit_type!r}")
    if bronze_edits and edit_type == "none":
        errors.append("label contradicts deterministic bronze edits")
    for field in ("entities", "quantities", "relations"):
        if not isinstance(label.get(field), list):
            errors.append(f"{field} must be a list")
    confidence = label.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        errors.append("confidence must be in [0, 1]")

    grounded = {float(value.replace(",", "")) for value in _NUMBER_RE.findall(sentence)}
    for quantity in label.get("quantities", []):
        if not isinstance(quantity, dict):
            errors.append("each quantity must be an object")
            continue
        value = quantity.get("value")
        if value is not None:
            try:
                numeric_value = float(str(value).replace(",", ""))
            except ValueError:
                errors.append(f"unparseable quantity: {value!r}")
                continue
            if numeric_value not in grounded:
                errors.append(f"ungrounded quantity: {value!r}")

    for relation in label.get("relations", []):
        if not isinstance(relation, str):
            errors.append("each relation must be a string")
            continue
        if "=" not in relation:
            continue
        left, right = relation.rsplit("=", 1)
        evaluated = safe_arithmetic_eval(left.strip())
        try:
            expected = float(right.strip().replace(",", ""))
        except ValueError:
            errors.append(f"unparseable relation result: {relation!r}")
            continue
        if evaluated is None or abs(evaluated - expected) > 1e-6:
            errors.append(f"unverified arithmetic relation: {relation!r}")
    return errors
