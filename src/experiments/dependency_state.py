"""Paired executable-code pilots for pending dependencies and stale conclusions."""

from __future__ import annotations

import ast
from collections import defaultdict
from graphlib import TopologicalSorter
import random
import re


def program_facts(code: str) -> dict:
    """Validate owned straight-line code, then use Python as the integer oracle.

    Never pass model output here. No calls, attributes, imports, indexing, loops,
    reassignment, or noninteger literals are allowed.
    """
    tree = ast.parse(code)
    allowed = (
        ast.Module,
        ast.Assign,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Constant,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Tuple,
    )
    if len(code) > 20000 or any(not isinstance(n, allowed) for n in ast.walk(tree)):
        raise ValueError("not a straight-line integer program")
    dependencies = {}
    order = []
    for statement in tree.body:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ValueError("expected a single named assignment")
        name = statement.targets[0].id
        if name in dependencies or name.startswith("__"):
            raise ValueError("reassignment or reserved name")
        parents = {n.id for n in ast.walk(statement.value) if isinstance(n, ast.Name)}
        if not parents <= dependencies.keys():
            raise ValueError("use before definition")
        for node in ast.walk(statement.value):
            if isinstance(node, ast.Constant) and (
                type(node.value) is not int or abs(node.value) > 1000000
            ):
                raise ValueError("expected bounded integer literals")
        dependencies[name] = parents
        order.append(name)
    tuple(TopologicalSorter(dependencies).static_order())
    values = {}
    exec(
        compile(tree, "<owned-dependency-program>", "exec"),
        {"__builtins__": {}},
        values,
    )
    if "result" not in values or not isinstance(values["result"], tuple):
        raise ValueError("program must assign a result tuple")
    positions = {name: index for index, name in enumerate(order)}
    last_use = dict(positions)
    distances = []
    for name, parents in dependencies.items():
        for parent in parents:
            last_use[parent] = max(last_use[parent], positions[name])
            distances.append(positions[name] - positions[parent])
    live = [
        sum(positions[n] <= i < last_use[n] for n in order) for i in range(len(order))
    ]
    return {
        "values": values,
        "dependencies": {k: sorted(v) for k, v in dependencies.items()},
        "live_counts": live,
        "peak_live": max(live, default=0),
        "mean_live": sum(live) / max(1, len(live)),
        "max_use_distance": max(distances, default=0),
        "mean_use_distance": sum(distances) / max(1, len(distances)),
        "operations": sum(isinstance(n, ast.BinOp) for n in ast.walk(tree)),
    }


def _row(family: str, arm: str, code: str, question: str, **metadata) -> dict:
    facts = program_facts(code)
    return {
        "id": f"{family}:{arm}",
        "family_id": family,
        "arm": arm,
        "question": question,
        "gold_answer": str(list(facts["values"]["result"])),
        "expected": list(facts["values"]["result"]),
        "code": code,
        "graph": facts["dependencies"],
        **{k: v for k, v in facts.items() if k not in {"values", "dependencies"}},
        **metadata,
    }


def load_cases(*, count: int, seed: int) -> list[dict]:
    """Same program graph and literals; only legal assignment order differs."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        width = (2, 4, 8)[index % 3]
        depth = (2, 4)[(index // 3) % 2]
        collision = bool((index // 6) % 2)
        common = [rng.randint(11, 29), *[rng.randint(2, 9) for _ in range(depth)]]
        branches = []
        sums = []
        for branch in range(width):
            numbers = (
                common
                if collision
                else [rng.randint(11, 29), *[rng.randint(2, 9) for _ in range(depth)]]
            )
            names = [f"v{branch:02d}_{d}" for d in range(depth + 1)]
            lines = [f"{names[0]} = {numbers[0]}"]
            for d in range(1, depth + 1):
                op = "*" if d % 2 == 0 else "+"
                lines.append(f"{names[d]} = {names[d - 1]} {op} {numbers[d]}")
            branches.append(lines)
            expression = (
                names[-1] if branch == 0 else f"s{branch - 1:02d} + {names[-1]}"
            )
            sums.append(f"s{branch:02d} = {expression}")
        low = [
            line for branch, total in zip(branches, sums) for line in [*branch, total]
        ]
        high = [branch[d] for d in range(depth + 1) for branch in branches] + sums
        family = f"load-{seed}-{index:04d}"
        for arm, lines in (("local", low), ("breadth", high)):
            code = "\n".join([*lines, f"result = (s{width - 1:02d},)"])
            question = f"What is the value of result after this Python code runs?\n```python\n{code}\n```"
            rows.append(
                _row(
                    family,
                    arm,
                    code,
                    question,
                    experiment="load",
                    width=width,
                    depth=depth,
                    collision=collision,
                    template_id=f"load-w{width}-d{depth}-c{int(collision)}",
                )
            )
    return rows


def revision_cases(*, count: int, seed: int) -> list[dict]:
    """Revise an initial input after supplied, oracle-checked partial work."""
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        depth = (2, 4, 8)[index % 3]
        collision = bool((index // 3) % 2)
        initial = rng.randint(11, 29)
        other = initial if collision else initial + 3
        amounts = [rng.randint(2, 5) for _ in range(depth)]
        branches = []
        for prefix, root in (("a", initial), ("b", other)):
            branch = [f"{prefix}00 = {root}"]
            for d, amount in enumerate(amounts, start=1):
                op = "*" if d % 2 == 0 else "+"
                branch.append(f"{prefix}{d:02d} = {prefix}{d - 1:02d} {op} {amount}")
            branches.append(branch)
        suffix = [
            f"joint = a{depth:02d} - b{depth:02d}",
            "final = joint * 2 + a00",
            f"result = (a00, final, b{depth:02d})",
        ]
        lines = [line for pair in zip(*branches) for line in pair]
        original = "\n".join([*lines, *suffix])
        old = program_facts(original)["values"]
        notes = [
            (
                line.split(" = ")[0],
                f"{line.split(' = ')[0]} = {old[line.split(' = ')[0]]}",
            )
            for line in lines
        ]
        for change in (False, True):
            revised = initial + 1 if change else initial
            corrected = original.replace(f"a00 = {initial}\n", f"a00 = {revised}\n", 1)
            family = f"revision-{seed}-{index:04d}-{int(change)}"
            for arm in (
                "intact",
                "drop_affected",
                "drop_independent",
                "drop_root",
                "restart",
            ):
                kept = [
                    (name, text)
                    for name, text in notes
                    if not (
                        (
                            arm == "drop_affected"
                            and name.startswith("a")
                            and name != "a00"
                        )
                        or (
                            arm == "drop_independent"
                            and name.startswith("b")
                            and name != "b00"
                        )
                        or (arm == "drop_root" and name == "a00")
                    )
                ]
                question = f"What is the value of result after this Python code runs?\n```python\n{original}\n```"
                question += "\n\nPartial calculation:\n" + "\n".join(t for _, t in kept)
                question += (
                    f"\n\nCorrection to the problem: the INITIAL assignment to a00 should be "
                    f"a00 = {revised}. All other code stays the same. "
                    "Finish the calculation for the corrected program."
                )
                if arm == "restart":
                    question = f"What is the value of result after this Python code runs?\n```python\n{corrected}\n```"
                rows.append(
                    _row(
                        family,
                        arm,
                        corrected,
                        question,
                        experiment="revision",
                        depth=depth,
                        collision=collision,
                        changed=change,
                        instance_id=f"revision-{seed}-{index:04d}",
                        template_id=f"revision-d{depth}-c{int(collision)}",
                        original_code=original,
                        old_expected=list(old["result"]),
                        stale_descendant_final=(
                            old[f"a{depth:02d}"] - old[f"b{depth:02d}"]
                        )
                        * 2
                        + revised,
                        removed_lines=len(notes) - len(kept)
                        if arm != "restart"
                        else len(notes),
                        retained_work="\n".join(t for _, t in kept)
                        if arm != "restart"
                        else "",
                    )
                )
    return rows


def parse_result(text: str) -> list[int] | None:
    """Require a final Answer line; never guess from other numbers in a trace."""
    lines = text.strip().splitlines()
    if not lines:
        return None
    match = re.fullmatch(
        r"\s*(?:(?:So,\s*)?(?:the\s+)?final\s+)?(?:answer|result)(?:\s+is)?"
        r"\s*:\s*`?(\[.*\]|\(.*\))`?\s*\.?\s*",
        lines[-1],
        re.I,
    )
    if match is None:
        return None
    try:
        result = ast.literal_eval(match[1])
    except (ValueError, SyntaxError):
        return None
    if not isinstance(result, (list, tuple)) or not all(type(v) is int for v in result):
        return None
    return list(result)


def score_case(case: dict, text: str) -> dict:
    prediction = parse_result(text)
    expected = case["expected"]
    valid = prediction is not None and len(prediction) == len(expected)
    scores = {
        "parsed": valid,
        "correct": valid and prediction == expected,
        "prediction": prediction,
    }
    if case["experiment"] == "revision":
        scores.update(
            root_correct=valid and prediction[0] == expected[0],
            downstream_correct=valid and prediction[1] == expected[1],
            independent_correct=valid and prediction[2] == expected[2],
            stale_with_correct_root=bool(
                valid
                and case["changed"]
                and prediction[0] == expected[0]
                and prediction[1] != expected[1]
                and prediction[1]
                in (case["old_expected"][1], case["stale_descendant_final"])
                and prediction[2] == expected[2]
            ),
        )
    return scores


def summarize(
    cases: list[dict], generations: list[dict], *, max_new_tokens: int
) -> dict:
    """Report raw paired pilot effects; no graph-generalization claim or pooled CI."""
    by_id = {row["id"]: row for row in cases}
    seen = set()
    measurements = []
    groups = defaultdict(list)
    families = defaultdict(dict)
    for generation in generations:
        sample_id = generation["sample_id"]
        if sample_id in seen or sample_id not in by_id:
            raise ValueError(f"duplicate or unknown generation: {sample_id}")
        seen.add(sample_id)
        case = by_id[sample_id]
        scores = score_case(case, generation["produced_text"])
        scores["capped"] = len(generation["generated_token_ids"]) >= max_new_tokens
        measurement = {
            "id": sample_id,
            "family_id": case["family_id"],
            "arm": case["arm"],
            **scores,
        }
        measurements.append(measurement)
        key = f"{case['arm']}:changed={case.get('changed', 'na')}"
        groups[key].append(scores)
        families[case["family_id"]][case["arm"]] = scores
    counts = {
        key: {
            "n": len(rows),
            **{
                metric: sum(r[metric] for r in rows)
                for metric in rows[0]
                if type(rows[0][metric]) is bool
            },
        }
        for key, rows in sorted(groups.items())
    }
    pairs = defaultdict(list)
    for family, arms in families.items():
        reference = "intact" if family.startswith("revision") else "breadth"
        if reference not in arms:
            continue
        changed = by_id[f"{family}:{reference}"].get("changed", "na")
        for arm, scores in arms.items():
            if arm != reference:
                pairs[f"{arm}-{reference}:changed={changed}"].append(
                    int(scores["correct"]) - int(arms[reference]["correct"])
                )
    return {
        "expected": len(cases),
        "completed": len(seen),
        "missing": sorted(by_id.keys() - seen),
        "counts": counts,
        "paired": {
            k: {
                "n": len(v),
                "wins": v.count(1),
                "losses": v.count(-1),
                "mean_difference": sum(v) / len(v),
            }
            for k, v in pairs.items()
        },
        "measurements": measurements,
        "status": "pilot_only",
    }
