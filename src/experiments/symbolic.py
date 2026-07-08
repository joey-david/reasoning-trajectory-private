"""Extract and verify arithmetic state updates from free-form reasoning text."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import re
from typing import Any

from reasoning_trajectory.token_alignment import TokenSpan, token_range_for_chars


_EQUATION_RE = re.compile(
    r"(?P<lhs>(?<![\w.])[-+()€$]?\s*\d[\d,]*(?:\.\d+)?"
    r"(?:\s*(?:\*\*|[+\-*/×÷])\s*[-+()€$]?\s*\d[\d,]*(?:\.\d+)?"
    r"|\s*[()]){1,12})"
    r"\s*=\s*(?P<rhs>[-+€$]?-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
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

# Prose arithmetic: word operators with explicit result (e.g. "24 plus 11 equals 35")
_PROSE_ARITH_EQ_RE = re.compile(
    r"(?i)"
    r"(?P<lhs_num>-?\d[\d,]*(?:\.\d+)?)\s+"
    r"(?P<word_op>plus|minus|times|multiplied\s+by|divided\s+by)\s+"
    r"(?P<rhs_num>-?\d[\d,]*(?:\.\d+)?)"
    r".{0,40}?"
    r"\b(?:is|equals|are|makes|gives|is\s+equal\s+to)\b\s+"
    r"(?P<result>-?\d[\d,]*(?:\.\d+)?)"
)

# Multiplier prose: "double/triple/quadruple of X (is|equals) Y" -> N*X=Y
_PROSE_MULT_RE = re.compile(
    r"(?i)"
    r"(?P<mult_word>double|triple|quadruple|twice|thrice)\s+"
    r"(?:of\s+)?"
    r"(?P<num>-?\d[\d,]*(?:\.\d+)?)\s+"
    r".{0,40}?"
    r"\b(?:is|equals|are|makes)\b\s+"
    r"(?P<result>-?\d[\d,]*(?:\.\d+)?)"
)

# Characters that continue an arithmetic expression (for BIND RHS extension)
_BIND_EXPR_CHARS: frozenset[str] = frozenset("0123456789+-*/.,()×÷% \t")


def _clean_expression(expression: str) -> str:
    """Strip unbalanced leading/trailing parens from an expression.

    Prose context like "(24 + 11 = 35 minutes)" captures "(" in the LHS
    of the equation match, which survives into the stored expression.
    Only strips parens that would create an invalid unbalanced expression.
    """
    expr = expression.strip()
    # Count opening and closing parens
    opens = expr.count("(")
    closes = expr.count(")")
    if opens == closes:
        return expr  # balanced — leave as-is
    # Strip unbalanced leading opens
    while expr.startswith("(") and opens > closes:
        expr = expr[1:].strip()
        opens = expr.count("(")
        closes = expr.count(")")
    # Strip unbalanced trailing closes
    while expr.endswith(")") and closes > opens:
        expr = expr[:-1].strip()
        opens = expr.count("(")
        closes = expr.count(")")
    return expr


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

    # --- Prose arithmetic equations (word operators, e.g., "24 plus 11 equals 35") ---
    _PROSE_OP_MAP = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
    }
    for match in _PROSE_ARITH_EQ_RE.finditer(text):
        word_op = match.group("word_op").strip().lower()
        symbol = _PROSE_OP_MAP.get(word_op)
        if symbol is None:
            continue
        lhs_text = f'{normalize_expression(match.group("lhs_num"))} {symbol} {normalize_expression(match.group("rhs_num"))}'
        raw_rhs = match.group("result")
        value = parse_number(raw_rhs)
        evaluated = safe_arithmetic_eval(lhs_text)
        if evaluated is None or not math.isclose(evaluated, value, rel_tol=1e-6, abs_tol=1e-6):
            continue
        signature = operation_signature(lhs_text)
        full_expr = f"{lhs_text} = {value:g}"
        lexical = lexical_items(match.group(0))
        candidates.append(
            (
                match.start(),
                match.end(),
                "OPERATE",
                full_expr,
                value,
                signature,
                lexical,
            )
        )

    # --- Prose multiplier equations (e.g., "quadruple of 35 is 140") ---
    _MULT_MAP = {"double": 2, "triple": 3, "quadruple": 4, "twice": 2, "thrice": 3}
    for match in _PROSE_MULT_RE.finditer(text):
        mult_word = match.group("mult_word").strip().lower()
        factor = _MULT_MAP.get(mult_word)
        if factor is None:
            continue
        num_str = normalize_expression(match.group("num"))
        value = parse_number(match.group("result"))
        evaluated_val = safe_arithmetic_eval(f"{num_str} * {factor}")
        if evaluated_val is None or not math.isclose(evaluated_val, value, rel_tol=1e-6, abs_tol=1e-6):
            continue
        expr = f"{num_str} * {factor}"
        signature = operation_signature(expr)
        full_expr = f"{expr} = {value:g}"
        lexical = lexical_items(match.group(0))
        candidates.append(
            (
                match.start(),
                match.end(),
                "OPERATE",
                full_expr,
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

    # --- BIND expression RHS extension ---
    # When a BIND matches a simple number but the text continues with an
    # arithmetic expression (e.g. "B = 24 / (4/6)"), evaluate the full RHS
    # and emit both an OPERATE and a corrected BIND.
    _UNIT_SKIP_RE = re.compile(
        r"\s*(?:[A-Za-z]+(?:/[A-Za-z]+)?\s+)+\s*(?=[\d*+/×÷\-])"
    )

    extended_binds: list[tuple[int, int, str, str, float, str, tuple[str, ...]]] = []
    for i, (start, end, operator, expression, value, signature, lexical) in enumerate(
        candidates
    ):
        if operator != "BIND":
            continue
        # Reconstruct the RHS text from the candidate expression (last token after '=')
        rhs_text = expression.split("=", 1)[-1].strip() if "=" in expression else str(value)
        remaining = text[end:]
        stop = 0
        while stop < len(remaining) and remaining[stop] in _BIND_EXPR_CHARS:
            stop += 1
        continuation = remaining[:stop].strip()
        has_op = bool(continuation and any(op in continuation for op in "+-*/×÷"))

        # If basic continuation has no operator, try skipping unit words
        # (e.g. "60 square yards * 36" → skip "square yards" to reach "*")
        if not has_op:
            unit_skip = _UNIT_SKIP_RE.match(remaining[stop:])
            if unit_skip:
                extra = remaining[stop : stop + unit_skip.end()]
                # Continue scanning arithmetic chars after the unit words
                extra_remaining = remaining[stop + unit_skip.end() :]
                extra_stop = 0
                while extra_stop < len(extra_remaining) and extra_remaining[extra_stop] in _BIND_EXPR_CHARS:
                    extra_stop += 1
                extra_cont = extra_remaining[:extra_stop].strip()
                if extra_cont and any(op in extra_cont for op in "+-*/×÷"):
                    continuation = extra + " " + extra_cont
                    stop = stop + unit_skip.end() + extra_stop
                    has_op = True

        if has_op:
            full_expr_str = f"{rhs_text}{continuation}"
            # Strip unit words from expression for evaluation
            clean_expr = _UNIT_SKIP_RE.sub(" ", full_expr_str).strip()
            evaluated = safe_arithmetic_eval(clean_expr)
            if evaluated is not None and not math.isclose(
                evaluated, value, rel_tol=1e-6, abs_tol=1e-6
            ):
                full_match_text = text[start : end + stop]
                op_signature = operation_signature(clean_expr)
                op_lexical = lexical_items(full_match_text)
                candidates[i] = (
                    start,
                    end + stop,
                    "BIND",
                    full_match_text,
                    evaluated,
                    "BIND",
                    op_lexical,
                )
                extended_binds.append(
                    (
                        start,
                        end + stop,
                        "OPERATE",
                        f"{clean_expr} = {evaluated:g}",
                        evaluated,
                        op_signature,
                        op_lexical,
                    )
                )

    candidates.extend(extended_binds)

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
                expression=_clean_expression(expression),
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
        # Fallback: strip unbalanced leading/trailing parens that come from
        # prose context (e.g. "(24 + 11 = 35 minutes)" captures "(" in LHS)
        stripped = expression
        while stripped.startswith("("):
            stripped = stripped[1:]
        while stripped.endswith(")"):
            stripped = stripped[:-1]
        stripped = stripped.strip()
        if stripped and stripped != expression:
            try:
                tree = ast.parse(stripped, mode="eval")
                value = _eval_node(tree.body)
            except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError):
                return None
        else:
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
    return (
        expression.replace(",", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("€", "")
        .replace("$", "")
        .strip()
    )


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
    """Parse a comma-separated numeric literal, stripping currency prefixes.

    Args:
        value: Value to rank, parse, or transform.

    Returns:
        The computed scalar metric.
    """
    return float(value.replace(",", "").replace("€", "").replace("$", ""))
