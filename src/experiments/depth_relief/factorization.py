"""Behavioral factorization of state reading, synthesis, and composition."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .benchmark import (
    answer_symbols,
    apply_rule,
    build_transition_case,
    candidate_token_ids,
    format_model_prompt,
    rule_text,
    state_symbols,
    state_text,
)
from .metrics import bootstrap_mean_ci
from .qualification import (
    evaluate_prompt_conditions_hf,
    evaluate_prompt_conditions_mlx,
)


ASSAYS = ("read", "update", "synthesize", "compose")
FORMATS = ("prose", "assignments")
STATE_REPRESENTATIONS = ("decimal", "symbols")
DIAGNOSTIC_TARGETS = (
    "correct_composition",
    "history_only",
    "final_on_start",
    "identity",
)
DEFAULT_DECISION = {
    "min_control_accuracy_lower": 0.85,
    "min_candidate_mass_lower": 0.80,
    "min_admitted_cases": 50,
    "min_update_minus_synthesize_lower": 0.30,
    "max_synthesize_compose_gap": 0.15,
    "min_format_update_minus_synthesize_lower": 0.20,
    "min_routing_cases": 50,
    "max_compose_given_synthesize_upper": 0.25,
    "min_synthesize_minus_compose_lower": 0.20,
    "min_format_synthesize_minus_compose_lower": 0.15,
}


def _state_path(case: dict[str, Any]) -> list[int]:
    modulus = 2 ** int(case["bits"])
    states = [int(case["initial_state"])]
    for rule in case["history"]:
        states.append(apply_rule(rule, states[-1], modulus))
    return states


def _diagnostic_targets(case: dict[str, Any]) -> dict[str, int]:
    modulus = 2 ** int(case["bits"])
    return {
        "correct_composition": int(case["next_state"]),
        "history_only": int(case["current_state"]),
        "final_on_start": apply_rule(
            case["final_rule"], int(case["initial_state"]), modulus
        ),
        "identity": int(case["initial_state"]),
    }


def build_factorization_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build matched semantic cases with distinguishable error branches."""
    history_families = tuple(
        str(value) for value in config.get("history_families", ("add", "xor", "affine"))
    )
    final_family = str(config.get("final_family", "pointer"))
    widths = tuple(int(value) for value in config.get("bits", (3,)))
    histories = tuple(int(value) for value in config.get("history_steps", (1, 2, 4)))
    formats = tuple(str(value) for value in config.get("formats", FORMATS))
    examples = int(config.get("examples_per_cell", 12))
    seed = int(config.get("seed", 0))
    require_distinct = bool(config.get("require_distinct_diagnostics", True))
    balance_current = bool(config.get("balance_current_states", False))
    state_representation = str(config.get("state_representation", "decimal"))
    if not set(formats).issubset(FORMATS):
        raise ValueError(f"Unknown factorization formats: {sorted(set(formats) - set(FORMATS))}")
    if state_representation not in STATE_REPRESENTATIONS:
        raise ValueError(f"Unknown state representation: {state_representation!r}")
    if any(width < 3 for width in widths) and require_distinct:
        raise ValueError("Distinct four-way diagnostics require at least three state bits")
    configured_symbols = tuple(str(value) for value in config.get("state_symbols", ()))
    if state_representation == "symbols":
        if history_families != ("pointer",) or final_family != "pointer":
            raise ValueError("Symbolic states require explicit pointer permutations")
        if any(len(configured_symbols) != 2**width for width in widths):
            raise ValueError("Symbolic state alphabet must match every configured width")
        if len(set(configured_symbols)) != len(configured_symbols):
            raise ValueError("Symbolic state alphabet must be unique")

    cases = []
    cell_index = 0
    for history_family in history_families:
        for history_steps in histories:
            for width in widths:
                accepted = []
                candidate_index = 0
                modulus = 2**width
                base_quota, remainder = divmod(examples, modulus)
                extra_states = {
                    (seed + cell_index * remainder + index) % modulus
                    for index in range(remainder)
                }
                cell_index += 1
                quotas = {
                    state: base_quota + int(state in extra_states)
                    for state in range(modulus)
                }
                state_counts: Counter[int] = Counter()
                while len(accepted) < examples:
                    case = build_transition_case(
                        history_family=history_family,
                        final_family=final_family,
                        width=width,
                        example_index=candidate_index,
                        seed=seed,
                        history_steps=history_steps,
                    )
                    candidate_index += 1
                    targets = _diagnostic_targets(case)
                    if require_distinct and len(set(targets.values())) != len(targets):
                        continue
                    current = int(case["current_state"])
                    if balance_current and state_counts[current] >= quotas[current]:
                        continue
                    case["history_steps"] = history_steps
                    case["state_path"] = _state_path(case)
                    case["diagnostic_targets"] = targets
                    if state_representation == "symbols":
                        case["state_representation"] = state_representation
                        case["state_symbols"] = list(configured_symbols)
                    accepted.append(case)
                    state_counts[current] += 1
                for representation in formats:
                    for example, case in enumerate(accepted):
                        row = dict(case)
                        row["format"] = representation
                        representation_suffix = (
                            f"_{state_representation}"
                            if state_representation != "decimal"
                            else ""
                        )
                        row["id"] = (
                            f"{history_family}_to_{final_family}_h{history_steps}_b{width}"
                            f"{representation_suffix}"
                            f"_{representation}_{example:04d}"
                        )
                        cases.append(row)
    return cases


def _assignment_rule_text(rule: dict[str, Any], width: int) -> str:
    kind = str(rule["kind"])
    modulus = 2**width
    if kind in {"pointer", "register_dispatch", "proof_action"}:
        table = ", ".join(str(value) for value in rule["mapping"])
        return f"state = table[state], where table = [{table}]"
    if kind == "affine":
        return f"state = ({rule['a']} * state + {rule['c']}) % {modulus}"
    if kind == "xor":
        return f"state = state ^ {rule['mask']}"
    if kind == "add":
        return f"state = (state + {rule['value']}) % {modulus}"
    if kind == "rotate_left":
        return f"state = rotate_left_{width}(state, {rule['amount']})"
    if kind == "register_add":
        return (
            f"R{rule['register']} = "
            f"(R{rule['register']} + {rule['value']}) % 4"
        )
    if kind == "register_xor":
        return f"R{rule['register']} = R{rule['register']} ^ {rule['mask']}"
    if kind == "register_swap":
        return "R0, R1 = R1, R0"
    if kind == "register_cond_add":
        source = int(rule["source"])
        return (
            f"if R{source} == {rule['equals']}: "
            f"R{1 - source} = (R{1 - source} + {rule['value']}) % 4"
        )
    if kind == "horn":
        premises = ", ".join(str(value) for value in rule["premises"])
        return (
            f"if all fact bits in [{premises}] are 1, "
            f"set fact bit {rule['conclusion']} to 1"
        )
    if kind == "proof_query":
        return (
            f"answer = proof_query(state, mask={rule['required_mask']}, "
            f"mode={rule.get('mode', 'all')})"
        )
    raise ValueError(f"Unknown transition kind: {kind!r}")


def render_factorization_rule(case: dict[str, Any], rule: dict[str, Any]) -> str:
    if case.get("state_representation", "decimal") == "symbols":
        if rule["kind"] != "pointer":
            raise ValueError("Symbolic state rules must be explicit permutations")
        symbols = state_symbols(case)
        if case["format"] == "assignments":
            entries = ", ".join(
                f"{symbols[source]}: {symbols[target]}"
                for source, target in enumerate(rule["mapping"])
            )
            return f"state = table[state], where table = {{{entries}}}"
        entries = ", ".join(
            f"{symbols[source]}->{symbols[target]}"
            for source, target in enumerate(rule["mapping"])
        )
        return f"replace the current symbol according to {{{entries}}}"
    if (
        case.get("state_representation") in {"hexadecimal", "opaque_fact_set"}
        and rule["kind"] in {"pointer", "register_dispatch", "proof_action"}
    ):
        symbols = state_symbols(case)
        entries = ", ".join(
            f"{symbols[source]}->{symbols[target]}"
            for source, target in enumerate(rule["mapping"])
        )
        if case["format"] == "assignments":
            return f"state = table[state], where table = {{{entries}}}"
        return f"look up the current state in {{{entries}}}"
    if case["format"] == "prose":
        return rule_text(rule, int(case["bits"]))
    return _assignment_rule_text(rule, int(case["bits"]))


def render_factorization_preamble(case: dict[str, Any]) -> str:
    if case.get("state_representation", "decimal") == "symbols":
        alphabet = ", ".join(state_symbols(case))
        if case["format"] == "assignments":
            return f"Execute the pseudocode exactly. State is one of [{alphabet}].\n"
        return (
            "Follow the state-transition instructions exactly. "
            f"The state is one of [{alphabet}].\n"
        )
    if case.get("state_representation") == "hexadecimal":
        alphabet = ", ".join(state_symbols(case))
        return (
            f"Execute the register program exactly. State is a {case['bits']}-bit "
            f"register shown as one hexadecimal digit in [{alphabet}]. "
            "Operation constants are decimal.\n"
        )
    if case.get("state_representation") == "opaque_fact_set":
        labels = state_symbols(case)
        facts = tuple(chr(ord("A") + bit) for bit in range(int(case["bits"])))
        entries = []
        for state, label in enumerate(labels):
            present = ",".join(
                facts[bit]
                for bit in range(len(facts))
                if state & (1 << bit)
            ) or "none"
            entries.append(f"{label}={{{present}}}")
        return (
            "Apply the proof rules exactly. The state label records established "
            f"facts: {'; '.join(entries)}.\n"
        )
    modulus = 2 ** int(case["bits"])
    if case.get("domain") == "horn_proof":
        facts = ", ".join(chr(ord("A") + bit) for bit in range(int(case["bits"])))
        return (
            f"Apply the proof rules exactly. The state is a decimal bitmask in "
            f"0..{modulus - 1} for established facts [{facts}].\n"
        )
    if case.get("domain") == "mixed_algebra":
        return (
            f"Execute the register program exactly. State is a decimal "
            f"{case['bits']}-bit register in 0..{modulus - 1}.\n"
        )
    if case["format"] == "assignments":
        return (
            f"Execute the pseudocode exactly. State is a decimal integer in 0..{modulus - 1}.\n"
        )
    return (
        f"Follow the state-transition instructions exactly. The state is a decimal "
        f"integer from 0 through {modulus - 1}.\n"
    )


def render_factorization_history(case: dict[str, Any]) -> str:
    return "\n".join(
        f"Step {index}: {render_factorization_rule(case, rule)}."
        for index, rule in enumerate(case["history"], 1)
    )


def _factorization_update_prompt(
    *,
    case: dict[str, Any],
    state: int,
    rule: dict[str, Any],
    name: str,
    label: str,
) -> dict[str, Any]:
    modulus = 2 ** int(case["bits"])
    if not 0 <= int(state) < modulus:
        raise ValueError(f"State {state} is outside [0, {modulus})")
    if label not in {"FINAL", "Operation"}:
        raise ValueError(f"Unsupported operation label: {label!r}")
    instruction = (
        "Apply FINAL exactly once and return the result."
        if label == "FINAL"
        else "Apply the operation exactly once and return the resulting state."
    )
    text = (
        render_factorization_preamble(case)
        + f"Current state: {state_text(case, int(state))}.\n"
        + f"{label}: {render_factorization_rule(case, rule)}.\n"
        + f"{instruction}\nAnswer="
    )
    return {
        "name": name,
        "text": text,
        "input_state": int(state),
        "expected_next_state": apply_rule(rule, int(state), modulus),
        "output_kind": "answer" if label == "FINAL" else "state",
    }


def render_factorization_update_prompt(
    *,
    tokenizer: Any,
    case: dict[str, Any],
    config: dict[str, Any],
    state: int,
    rule: dict[str, Any],
    name: str,
    label: str,
) -> dict[str, Any]:
    """Render one operation on an arbitrary supplied state."""
    prompt = _factorization_update_prompt(
        case=case,
        state=state,
        rule=rule,
        name=name,
        label=label,
    )
    prompt["text"] = format_model_prompt(tokenizer, prompt["text"], config)
    return prompt


def render_factorization_prompts(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Render the four assays and actual-input constituent transition controls."""
    preamble = render_factorization_preamble(case)
    current = int(case["current_state"])
    final = render_factorization_rule(case, case["final_rule"])
    history = render_factorization_history(case)
    prompts = [
        {
            "name": "read",
            "text": (
                preamble
                + f"Current state: {state_text(case, current)}.\n"
                "Return the current state unchanged.\nAnswer="
            ),
            "expected_next_state": current,
            "output_kind": "state",
        },
        {
            "name": "synthesize",
            "text": (
                preamble
                + f"Start state: {state_text(case, int(case['initial_state']))}.\n"
                f"{history}\n"
                "Apply every numbered step in order and return the resulting state.\nAnswer="
            ),
            "expected_next_state": current,
            "output_kind": "state",
        },
        {
            "name": "compose",
            "text": (
                preamble
                + f"Start state: {state_text(case, int(case['initial_state']))}.\n"
                f"{history}\n"
                + f"FINAL: {final}.\nApply every numbered step in order, then apply "
                "FINAL exactly once, and return the result.\nAnswer="
            ),
            "expected_next_state": int(case["next_state"]),
            "output_kind": "answer",
        },
    ]
    prompts.insert(
        1,
        _factorization_update_prompt(
            case=case,
            state=current,
            rule=case["final_rule"],
            name="update",
            label="FINAL",
        ),
    )
    states = case["state_path"]
    for index, rule in enumerate(case["history"], 1):
        prompts.append(
            _factorization_update_prompt(
                case=case,
                state=int(states[index - 1]),
                rule=rule,
                name=f"history_step_{index}",
                label="Operation",
            )
        )
    for prompt in prompts:
        prompt["text"] = format_model_prompt(tokenizer, prompt["text"], config)
    return prompts


def compose_capture_positions(
    *, tokenizer: Any, case: dict[str, Any], text: str
) -> list[dict[str, int | str]]:
    """Locate semantic endpoints in the exact formatted Compose prompt."""
    markers = [
        ("start", f"Start state: {state_text(case, int(case['initial_state']))}."),
        *[
            (f"history_step_{index}", line)
            for index, line in enumerate(render_factorization_history(case).splitlines(), 1)
        ],
        (
            "final_rule",
            f"FINAL: {render_factorization_rule(case, case['final_rule'])}.",
        ),
    ]
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(map(int, pair)) for pair in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets) or not offsets:
        raise ValueError("Tokenizer did not return usable offset mappings")

    positions: list[dict[str, int | str]] = []
    for name, marker in markers:
        if text.count(marker) != 1:
            raise ValueError(f"Compose marker is not unique: {marker!r}")
        marker_end = text.index(marker) + len(marker)
        matches = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < marker_end <= end
        ]
        if len(matches) != 1:
            raise ValueError(f"Could not align Compose marker endpoint: {marker!r}")
        positions.append({"name": name, "token_index": matches[0]})
    positions.append({"name": "answer", "token_index": len(token_ids) - 1})
    indices = [int(row["token_index"]) for row in positions]
    if indices != sorted(set(indices)):
        raise ValueError("Compose semantic endpoints are not strictly ordered")
    return positions


def validate_factorization_case(
    *, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_factorization_prompts(tokenizer=tokenizer, case=case, config=config)
    for prompt in prompts:
        candidates = (
            answer_symbols(case)
            if prompt.get("output_kind") == "answer"
            else state_symbols(case)
        )
        candidate_token_ids(tokenizer, prompt["text"], candidates)
    return {
        "id": case["id"],
        "condition_count": len(prompts),
        "token_count_range": [
            min(len(tokenizer.encode(prompt["text"], add_special_tokens=False)) for prompt in prompts),
            max(len(tokenizer.encode(prompt["text"], add_special_tokens=False)) for prompt in prompts),
        ],
    }


def factorization_record(
    case: dict[str, Any], conditions: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": case["id"],
        "history_family": case["history_family"],
        "final_family": case["final_family"],
        "format": case["format"],
        "bits": int(case["bits"]),
        "state_representation": str(case.get("state_representation", "decimal")),
        "state_symbols": list(state_symbols(case)),
        "answer_symbols": list(answer_symbols(case)),
        "history_steps": int(case["history_steps"]),
        "initial_state": int(case["initial_state"]),
        "current_state": int(case["current_state"]),
        "next_state": int(case["next_state"]),
        "state_path": [int(value) for value in case["state_path"]],
        "diagnostic_targets": case["diagnostic_targets"],
        "conditions": conditions,
    }


def evaluate_factorization_case_hf(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_factorization_prompts(tokenizer=tokenizer, case=case, config=config)
    conditions = {}
    for prompt in prompts:
        candidates = (
            answer_symbols(case)
            if prompt.get("output_kind") == "answer"
            else state_symbols(case)
        )
        conditions.update(
            evaluate_prompt_conditions_hf(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                candidate_symbols=candidates,
            )
        )
    return factorization_record(case, conditions)


def evaluate_factorization_case_mlx(
    *, model: Any, tokenizer: Any, case: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    prompts = render_factorization_prompts(tokenizer=tokenizer, case=case, config=config)
    conditions = {}
    for prompt in prompts:
        candidates = (
            answer_symbols(case)
            if prompt.get("output_kind") == "answer"
            else state_symbols(case)
        )
        conditions.update(
            evaluate_prompt_conditions_mlx(
                model=model,
                tokenizer=tokenizer,
                prompts=[prompt],
                candidate_symbols=candidates,
            )
        )
    return factorization_record(case, conditions)


def _correct(row: dict[str, Any], name: str) -> bool:
    return bool(row["conditions"][name]["is_expected_unconstrained"])


def _history_controls_correct(row: dict[str, Any]) -> bool:
    return all(
        _correct(row, f"history_step_{index}")
        for index in range(1, int(row["history_steps"]) + 1)
    )


def _admitted(row: dict[str, Any]) -> bool:
    return _correct(row, "read") and _correct(row, "update") and _history_controls_correct(row)


def _accuracy(rows: list[dict[str, Any]], name: str, seed: int) -> dict[str, Any]:
    return bootstrap_mean_ci([_correct(row, name) for row in rows], seed=seed)


def _paired_difference(
    rows: list[dict[str, Any]], left: str, right: str, seed: int
) -> dict[str, Any]:
    return bootstrap_mean_ci(
        [int(_correct(row, left)) - int(_correct(row, right)) for row in rows],
        seed=seed,
    )


def _lower(stat: dict[str, Any]) -> float:
    value = stat["ci95"][0]
    return float(value) if value is not None else float("-inf")


def _upper(stat: dict[str, Any]) -> float:
    value = stat["ci95"][1]
    return float(value) if value is not None else float("inf")


def _group_summary(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    admitted = [row for row in rows if _admitted(row)]
    return {
        "case_count": len(rows),
        "admitted_count": len(admitted),
        "accuracy": {
            name: _accuracy(rows, name, seed + index)
            for index, name in enumerate(ASSAYS)
        },
        "admitted_accuracy": {
            name: _accuracy(admitted, name, seed + 10 + index)
            for index, name in enumerate(("synthesize", "compose"))
        },
        "update_minus_synthesize": _paired_difference(
            rows, "update", "synthesize", seed + 20
        ),
        "synthesize_minus_compose": _paired_difference(
            rows, "synthesize", "compose", seed + 21
        ),
    }


def _compose_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    logprobabilities: defaultdict[str, list[float]] = defaultdict(list)
    margins: list[float] = []
    for row in rows:
        condition = row["conditions"]["compose"]
        prediction = condition["unconstrained_prediction"]
        targets = row["diagnostic_targets"]
        label = next(
            (name for name, state in targets.items() if prediction == int(state)),
            "other_or_nonstate",
        )
        counts[label] += 1
        values = condition.get("final_candidate_logprobabilities", [])
        for name, state in targets.items():
            if values:
                logprobabilities[name].append(float(values[int(state)]))
        if values:
            margins.append(
                float(values[int(targets["final_on_start"])])
                - float(values[int(targets["correct_composition"])])
            )
    labels = (*DIAGNOSTIC_TARGETS, "other_or_nonstate")
    return {
        "case_count": len(rows),
        "prediction_counts": {name: counts[name] for name in labels},
        "mean_output_logprobability": {
            name: bootstrap_mean_ci(values, seed=700 + index)
            for index, (name, values) in enumerate(sorted(logprobabilities.items()))
        },
        "final_on_start_minus_correct_logprobability": bootstrap_mean_ci(
            margins, seed=710
        ),
    }


def summarize_factorization_rows(
    rows: list[dict[str, Any]], decision_config: dict[str, Any]
) -> dict[str, Any]:
    """Separate state construction from transition competence and composition."""
    if not rows:
        raise ValueError("Cannot summarize an empty state-factorization result set")
    decision = {**DEFAULT_DECISION, **decision_config}
    overall = _group_summary(rows, seed=100)
    by_format = {
        value: _group_summary(
            [row for row in rows if row["format"] == value], seed=200 + index * 30
        )
        for index, value in enumerate(sorted({str(row["format"]) for row in rows}))
    }
    by_history = {
        str(value): _group_summary(
            [row for row in rows if int(row["history_steps"]) == value],
            seed=300 + value * 30,
        )
        for value in sorted({int(row["history_steps"]) for row in rows})
    }
    by_family = {
        value: _group_summary(
            [row for row in rows if row["history_family"] == value],
            seed=500 + index * 30,
        )
        for index, value in enumerate(
            sorted({str(row["history_family"]) for row in rows})
        )
    }
    controls = {
        "read": overall["accuracy"]["read"],
        "update": overall["accuracy"]["update"],
        "constituent_steps": bootstrap_mean_ci(
            [_history_controls_correct(row) for row in rows], seed=600
        ),
        "candidate_probability_mass": bootstrap_mean_ci(
            [
                float(condition["candidate_probability_mass"])
                for row in rows
                for condition in row["conditions"].values()
            ],
            seed=601,
        ),
    }
    admitted = [row for row in rows if _admitted(row)]
    read_update = [
        row for row in rows if _correct(row, "read") and _correct(row, "update")
    ]
    routing = [row for row in read_update if _correct(row, "synthesize")]
    routing_failures = [row for row in routing if not _correct(row, "compose")]
    taxonomy = Counter(
        f"synthesize_{'correct' if _correct(row, 'synthesize') else 'wrong'}__"
        f"compose_{'correct' if _correct(row, 'compose') else 'wrong'}"
        for row in admitted
    )
    gap = overall["synthesize_minus_compose"]
    common_checks = {
        "read_control": _lower(controls["read"])
        >= float(decision["min_control_accuracy_lower"]),
        "update_control": _lower(controls["update"])
        >= float(decision["min_control_accuracy_lower"]),
        "candidate_mass": _lower(controls["candidate_probability_mass"])
        >= float(decision["min_candidate_mass_lower"]),
    }
    synthesis_checks = {
        **common_checks,
        "constituent_controls": _lower(controls["constituent_steps"])
        >= float(decision["min_control_accuracy_lower"]),
        "admitted_cases": len(admitted) >= int(decision["min_admitted_cases"]),
        "update_synthesis_dissociation": _lower(overall["update_minus_synthesize"])
        >= float(decision["min_update_minus_synthesize_lower"]),
        "synthesis_compose_tracking": max(abs(_lower(gap)), abs(_upper(gap)))
        <= float(decision["max_synthesize_compose_gap"]),
        "format_replication": all(
            _lower(summary["update_minus_synthesize"])
            >= float(decision["min_format_update_minus_synthesize_lower"])
            for summary in by_format.values()
        ),
    }
    routing_compose = bootstrap_mean_ci(
        [_correct(row, "compose") for row in routing], seed=720
    )
    read_update_gap = _paired_difference(
        read_update, "synthesize", "compose", seed=721
    )
    routing_checks = {
        **common_checks,
        "routing_cases": len(routing) >= int(decision["min_routing_cases"]),
        "compose_given_synthesize": _upper(routing_compose)
        <= float(decision["max_compose_given_synthesize_upper"]),
        "synthesize_compose_dissociation": _lower(read_update_gap)
        >= float(decision["min_synthesize_minus_compose_lower"]),
        "format_replication": all(
            _lower(summary["synthesize_minus_compose"])
            >= float(decision["min_format_synthesize_minus_compose_lower"])
            for summary in by_format.values()
        ),
    }
    return {
        "schema_version": 2,
        "case_count": len(rows),
        "formats": sorted(by_format),
        "state_representations": sorted(
            {str(row.get("state_representation", "decimal")) for row in rows}
        ),
        "history_families": sorted(by_family),
        "history_steps": sorted(int(value) for value in by_history),
        "controls": controls,
        "overall": overall,
        "by_format": by_format,
        "by_history": by_history,
        "by_history_family": by_family,
        "competence_admission": {
            "definition": "read AND update AND every actual-input history step",
            "count": len(admitted),
            "rate": len(admitted) / len(rows),
        },
        "admitted_synthesis_compose_taxonomy": dict(sorted(taxonomy.items())),
        "compose_diagnostics_strict_admission": _compose_diagnostics(admitted),
        "routing_analysis": {
            "definition": "read AND update AND synthesize",
            "eligible_count": len(routing),
            "compose_accuracy": routing_compose,
            "synthesize_minus_compose_among_read_update": read_update_gap,
            "failure_diagnostics": _compose_diagnostics(routing_failures),
        },
        "decision": {
            "thresholds": decision,
            "state_synthesis_bottleneck": {
                "checks": synthesis_checks,
                "supported": all(synthesis_checks.values()),
            },
            "serial_integration_failure": {
                "checks": routing_checks,
                "supported": all(routing_checks.values()),
            },
        },
    }
