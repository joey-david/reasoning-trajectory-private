"""Extract and verify arithmetic state updates from free-form reasoning text."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Any

from reasoning_trajectory.token_alignment import TokenSpan, token_range_for_chars


_EQUATION_RE = re.compile(
    r"(?P<lhs>(?<![\w.])[-+()]?\s*\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:\*\*|[+\-*/×÷])\s*[-+()]?\s*\d[\d,]*(?:\.\d+)?"
    r"|\s*[()]){1,12})"
    r"\s*=\s*(?P<rhs>-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
_BIND_RE = re.compile(
    r"(?P<name>\b[A-Za-z][A-Za-z0-9_]{0,30})\s*=\s*"
    r"(?P<rhs>-?\d[\d,]*(?:\.\d+)?)"
)
_ANSWER_RE = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=)\s*"
    r"(?P<rhs>-?\d[\d,]*(?:\.\d+)?)"
)
_VERIFY_WORDS = re.compile(
    r"(?i)\b(?:check|verify|indeed|correct|consistent|recalculate|double-check)\b"
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass(slots=True)
class SymbolicUpdate:
    """One textually completed, deterministically verified graph update."""

    char_start: int
    char_end: int
    token_start: int
    token_end: int
    operator: str
    expression: str
    value: float
    operation_signature: str
    graph_signature: str
    lexical_items: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        """Serialize the dataclass as a JSON-compatible record.

        Args:
            None.

        Returns:
            The resulting keyed records or metrics.
        """
        return {
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "operator": self.operator,
            "expression": self.expression,
            "value": self.value,
            "operation_signature": self.operation_signature,
            "graph_signature": self.graph_signature,
            "lexical_items": list(self.lexical_items),
        }


def extract_symbolic_updates(
    text: str,
    token_spans: list[TokenSpan] | None = None,
    *,
    token_count: int | None = None,
) -> list[SymbolicUpdate]:
    """Extract valid arithmetic relations, bindings, and terminal answers.

    Args:
        text: Generated text to inspect.
        token_spans: Decoded character spans aligned with generated tokens.
        token_count: Number of generated tokens.

    Returns:
        The resulting ordered records or values.
    """
    candidates: list[tuple[int, int, str, str, float, str, tuple[str, ...]]] = []

    # Equations enter the graph only when the restricted evaluator verifies the
    # textual right-hand value; regex shape alone is not symbolic evidence.
    for match in _EQUATION_RE.finditer(text):
        expression = normalize_expression(match.group("lhs"))
        value = parse_number(match.group("rhs"))
        evaluated = safe_arithmetic_eval(expression)
        if evaluated is None or not math.isclose(
            evaluated, value, rel_tol=1e-6, abs_tol=1e-6
        ):
            continue
        signature = operation_signature(expression)
        lexical = lexical_items(match.group(0))
        candidates.append(
            (
                match.start(),
                match.end(),
                "OPERATE",
                match.group(0),
                value,
                signature,
                lexical,
            )
        )

    occupied = [(start, end) for start, end, *_ in candidates]
    for match in _BIND_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in occupied):
            continue
        candidates.append(
            (
                match.start(),
                match.end(),
                "BIND",
                match.group(0),
                parse_number(match.group("rhs")),
                "BIND",
                lexical_items(match.group(0)),
            )
        )

    answer_matches = list(_ANSWER_RE.finditer(text))
    for match in answer_matches[-1:]:
        candidates.append(
            (
                match.start(),
                match.end(),
                "EXTRACT",
                match.group(0),
                parse_number(match.group("rhs")),
                "EXTRACT",
                lexical_items(match.group(0)),
            )
        )

    updates: list[SymbolicUpdate] = []
    seen_relations: set[tuple[str, float]] = set()
    graph_entries: list[str] = []
    # Completion order defines graph state. Repeated relations and explicit
    # checking language are VERIFY events and therefore do not mutate it.
    for start, end, operator, expression, value, signature, lexical in sorted(
        candidates, key=lambda item: (item[1], item[0])
    ):
        relation_atom = symbolic_relation_atom(operator, expression, value)
        relation = (relation_atom, round(value, 8))
        context = text[max(0, start - 80) : min(len(text), end + 40)]
        if operator == "OPERATE" and (
            relation in seen_relations or _VERIFY_WORDS.search(context)
        ):
            operator = "VERIFY"
        if operator != "VERIFY":
            seen_relations.add(relation)
        if operator != "VERIFY":
            graph_entries.append(relation_atom)
        token_range = aligned_token_range(
            token_spans or [], start, end, len(text), token_count
        )
        if token_range is None:
            continue
        updates.append(
            SymbolicUpdate(
                char_start=start,
                char_end=end,
                token_start=token_range[0],
                token_end=token_range[1],
                operator=operator,
                expression=expression,
                value=value,
                operation_signature=signature,
                graph_signature="|".join(sorted(set(graph_entries))),
                lexical_items=lexical,
            )
        )
    return deduplicate_updates(updates)


def safe_arithmetic_eval(expression: str) -> float | None:
    """Evaluate a numeric arithmetic expression through a restricted AST.

    Args:
        expression: Arithmetic expression to parse or normalize.

    Returns:
        The computed scalar metric, or ``None`` when unavailable.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _eval_node(node: ast.AST) -> float:
    """Evaluate a restricted arithmetic AST recursively.

    Args:
        node: Arithmetic AST node to evaluate.

    Returns:
        The computed scalar metric.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and abs(right) <= 8:
            return left**right
    raise ValueError(f"Unsupported arithmetic node: {type(node).__name__}")


def normalize_expression(expression: str) -> str:
    """Normalize common arithmetic glyphs and separators.

    Args:
        expression: Arithmetic expression to parse or normalize.

    Returns:
        The resulting text or classification label.
    """
    return expression.replace(",", "").replace("×", "*").replace("÷", "/").strip()


def canonical_expression(expression: str) -> str:
    """Convert an arithmetic expression to a canonical AST string.

    Args:
        expression: Arithmetic expression to parse or normalize.

    Returns:
        The resulting text or classification label.
    """
    normalized = normalize_expression(expression)
    try:
        return ast.dump(
            ast.parse(normalized, mode="eval").body, include_attributes=False
        )
    except SyntaxError:
        return normalized


def operation_signature(expression: str) -> str:
    """Describe the arithmetic operators used by an expression.

    Args:
        expression: Arithmetic expression to parse or normalize.

    Returns:
        The resulting text or classification label.
    """
    try:
        tree = ast.parse(normalize_expression(expression), mode="eval")
    except SyntaxError:
        return "UNKNOWN"
    operators: list[str] = []
    names = {
        ast.Add: "ADD",
        ast.Sub: "SUBTRACT",
        ast.Mult: "MULTIPLY",
        ast.Div: "DIVIDE",
        ast.Pow: "POWER",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            operators.append(names.get(type(node.op), "OTHER"))
    return "+".join(sorted(operators)) if operators else "BIND"


def symbolic_relation_atom(operator: str, expression: str, value: float) -> str:
    """Canonicalize a graph relation while ignoring superficial formatting.

    Args:
        operator: Symbolic update taxonomy label.
        expression: Arithmetic expression to parse or normalize.
        value: Value to rank, parse, or transform.

    Returns:
        The resulting text or classification label.
    """
    if operator in {"BIND", "EXTRACT"}:
        return f"{operator}:{round(value, 8):g}"
    lhs = expression.rsplit("=", 1)[0]
    return f"{canonical_expression(lhs)}={round(value, 8):g}"


def lexical_items(text: str) -> tuple[str, ...]:
    """Return literals and variable names, excluding the shared operator syntax.

    Args:
        text: Generated text to inspect.

    Returns:
        The computed aligned values described above.
    """
    words = {word.lower() for word in _WORD_RE.findall(text)}
    numbers = {number.replace(",", "") for number in _NUMBER_RE.findall(text)}
    return tuple(sorted(words | numbers))


def aligned_token_range(
    spans: list[TokenSpan],
    char_start: int,
    char_end: int,
    text_length: int,
    count: int | None,
) -> tuple[int, int] | None:
    """Map a character interval to tokens with a proportional fallback.

    Args:
        spans: Token spans used for character alignment.
        char_start: Inclusive character offset.
        char_end: Exclusive character offset.
        text_length: Character length of the decoded generation.
        count: Optional number of generated tokens.

    Returns:
        The computed aligned values described above.
    """
    exact = token_range_for_chars(spans, char_start, char_end)
    if exact is not None:
        return exact
    if not count or text_length <= 0:
        return None
    start = min(int(char_start / text_length * count), count - 1)
    end = min(max(int(math.ceil(char_end / text_length * count)) - 1, start), count - 1)
    return start, end


def deduplicate_updates(updates: list[SymbolicUpdate]) -> list[SymbolicUpdate]:
    """Prefer terminal-answer labels when multiple patterns cover one token.

    Args:
        updates: Symbolic solution-object updates.

    Returns:
        The resulting ordered records or values.
    """
    by_endpoint: dict[tuple[int, str, float], SymbolicUpdate] = {}
    priority = {"OPERATE": 0, "BIND": 1, "VERIFY": 2, "EXTRACT": 3}
    for update in updates:
        key = (update.token_end, update.operation_signature, round(update.value, 8))
        previous = by_endpoint.get(key)
        if previous is None or priority[update.operator] > priority[previous.operator]:
            by_endpoint[key] = update
    return sorted(by_endpoint.values(), key=lambda update: update.token_end)


def parse_number(value: str) -> float:
    """Parse a comma-separated numeric literal.

    Args:
        value: Value to rank, parse, or transform.

    Returns:
        The computed scalar metric.
    """
    return float(value.replace(",", ""))
