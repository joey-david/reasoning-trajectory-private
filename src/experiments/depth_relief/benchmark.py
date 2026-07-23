"""Deterministic controlled tasks with exact intermediate-state semantics."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Sequence


TASK_FAMILIES = ("pointer", "affine", "register")
RULE_FAMILIES = ("pointer", "affine", "xor", "add", "rotate_left", "register")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    """A rendered prompt and the character span occupied by its checkpoint."""

    text: str
    checkpoint_start: int
    checkpoint_end: int


def bits(value: int, width: int) -> list[int]:
    """Return ``value`` as a fixed-width most-significant-bit-first vector."""
    return [int(char) for char in f"{value:0{width}b}"]


def bit_text(values: Iterable[int | str]) -> str:
    """Render register symbols with stable whitespace between every bit."""
    return " ".join(str(value) for value in values)


def decimal_state_symbols(count: int) -> tuple[str, ...]:
    """Return the canonical surface alphabet for a decimal state space."""
    return tuple(str(value) for value in range(count))


def hexadecimal_state_symbols(count: int) -> tuple[str, ...]:
    """Return one hexadecimal digit for state spaces of at most 16 values."""
    if not 1 <= count <= 16:
        raise ValueError("Hexadecimal state symbols support at most 16 values")
    return tuple("０１２３４５６７８９ＡＢＣＤＥＦ"[:count])


def state_symbols(case: dict[str, Any]) -> tuple[str, ...]:
    """Return and validate the one-to-one surface alphabet owned by a case."""
    count = 2 ** int(case["bits"])
    symbols = tuple(
        str(value)
        for value in case.get("state_symbols", decimal_state_symbols(count))
    )
    if len(symbols) != count or len(set(symbols)) != count:
        raise ValueError(f"State alphabet must contain {count} unique symbols")
    return symbols


def state_text(case: dict[str, Any], state: int) -> str:
    """Render one internal state through its case-owned surface alphabet."""
    symbols = state_symbols(case)
    if not 0 <= int(state) < len(symbols):
        raise ValueError(f"State {state} is outside [0, {len(symbols)})")
    return symbols[int(state)]


def answer_symbols(case: dict[str, Any]) -> tuple[str, ...]:
    """Return the output alphabet for FINAL, which may differ from state labels."""
    symbols = tuple(
        str(value) for value in case.get("answer_symbols", state_symbols(case))
    )
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Answer alphabet must contain unique symbols")
    return symbols


def answer_text(case: dict[str, Any], answer: int) -> str:
    """Render one FINAL result through its case-owned answer alphabet."""
    symbols = answer_symbols(case)
    if not 0 <= int(answer) < len(symbols):
        raise ValueError(f"Answer {answer} is outside [0, {len(symbols)})")
    return symbols[int(answer)]


def candidate_token_ids(
    tokenizer: Any, prompt: str, symbols: Sequence[str]
) -> list[int]:
    """Validate a unique one-token answer contract for every state symbol."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    result = []
    for symbol in symbols:
        extended = tokenizer.encode(prompt + symbol, add_special_tokens=False)
        if extended[:-1] != prompt_ids or len(extended) != len(prompt_ids) + 1:
            raise ValueError(
                "State candidate is not one token at the Answer= boundary: "
                f"{symbol!r}. Change the benchmark contract, not the scorer."
            )
        result.append(int(extended[-1]))
    if len(set(result)) != len(symbols):
        raise ValueError("State candidates do not map to unique token IDs")
    return result


def format_model_prompt(
    tokenizer: Any, text: str, config: dict[str, Any]
) -> str:
    """Place benchmark content in the model's native prompt contract."""
    prompt_config = config.get("prompt")
    if prompt_config is None:
        prompt_config = config if "mode" in config else {}
    if prompt_config.get("mode", "plain") == "plain":
        return text
    if prompt_config.get("mode") != "chat":
        raise ValueError(f"Unsupported depth-relief prompt mode: {prompt_config.get('mode')!r}")
    boundaries = [suffix for suffix in ("Answer=", "Checkpoint=") if text.endswith(suffix)]
    if len(boundaries) != 1:
        raise ValueError("Chat benchmark prompts need one assistant-side answer boundary")
    boundary = boundaries[0]
    messages = []
    if system := str(prompt_config.get("system", "")).strip():
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text[: -len(boundary)].rstrip()})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **prompt_config.get("chat_template_kwargs", {}),
    )
    return str(rendered) + boundary


def format_prompt_spec(
    tokenizer: Any, spec: PromptSpec, config: dict[str, Any]
) -> PromptSpec:
    """Apply model formatting while preserving the checkpoint character span."""
    text = format_model_prompt(tokenizer, spec.text, config)
    content_start = text.find(spec.text[: spec.checkpoint_start])
    if content_start < 0:
        raise ValueError("Chat template did not preserve benchmark prompt content")
    return PromptSpec(
        text=text,
        checkpoint_start=content_start + spec.checkpoint_start,
        checkpoint_end=content_start + spec.checkpoint_end,
    )


def apply_rule(rule: dict[str, Any], state: int, modulus: int) -> int:
    """Apply one serialized state transition or final state query."""
    kind = str(rule["kind"])
    if kind in {"pointer", "register_dispatch", "proof_action"}:
        return int(rule["mapping"][state])
    if kind == "affine":
        return (int(rule["a"]) * state + int(rule["c"])) % modulus
    if kind == "xor":
        return state ^ int(rule["mask"])
    if kind == "add":
        return (state + int(rule["value"])) % modulus
    if kind == "rotate_left":
        width = modulus.bit_length() - 1
        amount = int(rule["amount"]) % width
        mask = modulus - 1
        return ((state << amount) & mask) | (state >> (width - amount))
    if kind == "horn":
        premises = tuple(int(value) for value in rule["premises"])
        conclusion = int(rule["conclusion"])
        if any(not 0 <= bit < modulus.bit_length() - 1 for bit in premises):
            raise ValueError("Horn premise is outside the fact register")
        if not 0 <= conclusion < modulus.bit_length() - 1:
            raise ValueError("Horn conclusion is outside the fact register")
        if all(state & (1 << bit) for bit in premises):
            return state | (1 << conclusion)
        return state
    if kind == "proof_query":
        required = int(rule["required_mask"])
        mode = str(rule.get("mode", "all"))
        if not 0 <= required < modulus:
            raise ValueError("Proof query mask is outside the fact register")
        if mode == "all":
            return int((state & required) == required)
        if mode == "any":
            return int(bool(state & required))
        if mode == "parity":
            return int((state & required).bit_count() % 2)
        raise ValueError(f"Unknown proof-query mode: {mode!r}")
    raise ValueError(f"Unknown transition kind: {kind!r}")


def rule_text(rule: dict[str, Any], width: int) -> str:
    """Render a transition without hiding any operation needed for recomputation."""
    kind = str(rule["kind"])
    if kind == "pointer":
        entries = ", ".join(
            f"{source}->{target}" for source, target in enumerate(rule["mapping"])
        )
        return f"look up the current decimal state in {{{entries}}}"
    if kind == "register_dispatch":
        entries = ", ".join(
            f"{source}->{target}" for source, target in enumerate(rule["mapping"])
        )
        return f"dispatch the current register state through {{{entries}}}"
    if kind == "proof_action":
        entries = ", ".join(
            f"{source}->{target}" for source, target in enumerate(rule["mapping"])
        )
        return f"choose the next proof action from {{{entries}}}"
    if kind == "affine":
        return f"state = ({rule['a']} * state + {rule['c']}) mod {2**width}"
    if kind == "xor":
        return f"XOR the register with {bit_text(bits(int(rule['mask']), width))}"
    if kind == "add":
        return f"add {rule['value']} modulo {2**width}"
    if kind == "rotate_left":
        return f"rotate the {width}-bit register left by {rule['amount']}"
    if kind == "horn":
        names = tuple(chr(ord("A") + bit) for bit in rule["premises"])
        conclusion = chr(ord("A") + int(rule["conclusion"]))
        if not names:
            return f"establish fact {conclusion} unconditionally"
        if len(names) == 1:
            return f"if fact {names[0]} is established, establish fact {conclusion}"
        premise = " and ".join(names)
        return f"if facts {premise} are established, establish fact {conclusion}"
    if kind == "proof_query":
        names = [
            chr(ord("A") + bit)
            for bit in range(width)
            if int(rule["required_mask"]) & (1 << bit)
        ]
        facts = " and ".join(names)
        mode = str(rule.get("mode", "all"))
        if mode == "all":
            return f"return 1 exactly when facts {facts} are all established, else 0"
        if mode == "any":
            return f"return 1 when any of facts {facts} is established, else 0"
        if mode == "parity":
            return f"return the parity of established facts among {facts}"
        raise ValueError(f"Unknown proof-query mode: {mode!r}")
    raise ValueError(f"Unknown transition kind: {kind!r}")


def _random_rule(family: str, width: int, rng: random.Random) -> dict[str, Any]:
    modulus = 2**width
    if family == "pointer":
        mapping = list(range(modulus))
        rng.shuffle(mapping)
        return {"kind": "pointer", "mapping": mapping}
    if family == "affine":
        return {
            "kind": "affine",
            "a": rng.choice(list(range(1, modulus, 2))),
            "c": rng.randrange(modulus),
        }
    if family == "register":
        family = rng.choice(("xor", "add", "rotate_left"))
    if family == "xor":
        return {"kind": family, "mask": rng.randrange(1, modulus)}
    if family == "add":
        return {"kind": family, "value": rng.randrange(1, modulus)}
    if family == "rotate_left":
        return {"kind": family, "amount": rng.randrange(1, width)}
    raise ValueError(f"Unknown task family: {family!r}")


def build_case(
    *, family: str, width: int, example_index: int, seed: int, history_steps: int
) -> dict[str, Any]:
    """Construct one exact transition chain and its one-bit counterfactual branch."""
    if family not in TASK_FAMILIES:
        raise ValueError(f"Unknown task family: {family!r}")
    if width < 2:
        raise ValueError("State widths must be at least two bits")
    rng = random.Random(seed + 100_003 * width + 10_007 * example_index + 97 * TASK_FAMILIES.index(family))
    modulus = 2**width
    initial = rng.randrange(modulus)
    history = [_random_rule(family, width, rng) for _ in range(history_steps)]
    final_rule = _random_rule(family, width, rng)
    current = initial
    for rule in history:
        current = apply_rule(rule, current, modulus)
    counterfactual = current ^ (1 << rng.randrange(width))
    random_state = rng.randrange(modulus)
    while random_state in {current, counterfactual}:
        random_state = rng.randrange(modulus)
    return {
        "id": f"{family}_b{width}_{example_index:04d}",
        "family": family,
        "bits": width,
        "example_index": example_index,
        "initial_state": initial,
        "history": history,
        "final_rule": final_rule,
        "current_state": current,
        "next_state": apply_rule(final_rule, current, modulus),
        "counterfactual_state": counterfactual,
        "counterfactual_next_state": apply_rule(final_rule, counterfactual, modulus),
        "random_state": random_state,
        "random_next_state": apply_rule(final_rule, random_state, modulus),
    }


def build_transition_case(
    *,
    history_family: str,
    final_family: str,
    width: int,
    example_index: int,
    seed: int,
    history_steps: int,
) -> dict[str, Any]:
    """Construct a chain whose history and final transition have separate owners."""
    if history_family not in RULE_FAMILIES or final_family not in RULE_FAMILIES:
        raise ValueError(
            f"Unknown transition families: {history_family!r}, {final_family!r}"
        )
    if width < 2:
        raise ValueError("State widths must be at least two bits")
    rng = random.Random(
        seed
        + 100_003 * width
        + 10_007 * example_index
        + 997 * history_steps
        + 97 * RULE_FAMILIES.index(history_family)
        + 53 * RULE_FAMILIES.index(final_family)
    )
    modulus = 2**width
    initial = rng.randrange(modulus)
    history = [
        _random_rule(history_family, width, rng) for _ in range(history_steps)
    ]
    final_rule = _random_rule(final_family, width, rng)
    current = initial
    for rule in history:
        current = apply_rule(rule, current, modulus)
    counterfactual = current ^ (1 << rng.randrange(width))
    random_state = rng.randrange(modulus)
    while random_state in {current, counterfactual}:
        random_state = rng.randrange(modulus)
    return {
        "family": f"{history_family}_to_{final_family}",
        "history_family": history_family,
        "final_family": final_family,
        "bits": width,
        "example_index": example_index,
        "initial_state": initial,
        "history": history,
        "final_rule": final_rule,
        "current_state": current,
        "next_state": apply_rule(final_rule, current, modulus),
        "counterfactual_state": counterfactual,
        "counterfactual_next_state": apply_rule(final_rule, counterfactual, modulus),
        "random_state": random_state,
        "random_next_state": apply_rule(final_rule, random_state, modulus),
        "history_steps": history_steps,
    }


def build_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the configured factorial benchmark in a stable order."""
    families = tuple(config.get("families", TASK_FAMILIES))
    widths = tuple(int(value) for value in config.get("bits", (2, 3, 4)))
    examples = int(config.get("examples_per_cell", 8))
    seed = int(config.get("seed", 0))
    history_steps = int(config.get("history_steps", 2))
    return [
        build_case(
            family=str(family),
            width=width,
            example_index=example,
            seed=seed,
            history_steps=history_steps,
        )
        for family in families
        for width in widths
        for example in range(examples)
    ]


def condition_specs(
    case: dict[str, Any], *, self_state: int | None = None
) -> list[dict[str, Any]]:
    """Return matched checkpoint conditions and their rule-consistent targets."""
    width = int(case["bits"])
    modulus = 2**width
    current = int(case["current_state"])

    def slots(possible: Iterable[int]) -> list[str]:
        values = [f"{value:02d}" for value in possible]
        return values + ["99"] * (modulus - len(values))

    specs = [
        {
            "name": "none",
            "symbols": slots(range(modulus)),
            "revealed_bits": 0,
            "state": None,
            "expected_next_state": int(case["next_state"]),
        }
    ]
    specs.extend(
        {
            "name": f"partial_{revealed}",
            "symbols": slots(
                state
                for state in range(modulus)
                if bits(state, width)[:revealed] == bits(current, width)[:revealed]
            ),
            "revealed_bits": revealed,
            "state": None,
            "expected_next_state": int(case["next_state"]),
        }
        for revealed in range(1, width)
    )
    for name, key, target_key in (
        ("gold", "current_state", "next_state"),
        ("counterfactual", "counterfactual_state", "counterfactual_next_state"),
        ("random", "random_state", "random_next_state"),
    ):
        state = int(case[key])
        specs.append(
            {
                "name": name,
                "symbols": slots([state]),
                "revealed_bits": width if name == "gold" else None,
                "state": state,
                "expected_next_state": int(case[target_key]),
            }
        )
    if self_state is not None:
        state = int(self_state)
        if not 0 <= state < modulus:
            raise ValueError(f"Self-written state {state} is outside [0, {modulus})")
        specs.append(
            {
                "name": "self",
                "symbols": slots([state]),
                "revealed_bits": width if state == int(case["current_state"]) else None,
                "state": state,
                "expected_next_state": apply_rule(case["final_rule"], state, modulus),
            }
        )
    return specs


def _prefix(case: dict[str, Any]) -> str:
    width = int(case["bits"])
    history = "\n".join(
        f"Step {index}: {rule_text(rule, width)}."
        for index, rule in enumerate(case["history"], 1)
    )
    return (
        f"Operate a state from 0 through {2**width - 1}. Apply the numbered steps in "
        "order. The checkpoint is after every numbered step and immediately before "
        "FINAL. It lists possible decimal states; 99 is an empty padded slot. If one "
        "state is listed, it is authoritative and the numbered steps must not be "
        "repeated. If several are listed, use the earlier steps to identify the state.\n"
        f"Initial state: {int(case['initial_state'])}.\n"
        f"{history}\n"
    )


def render_prompt(case: dict[str, Any], condition: dict[str, Any]) -> PromptSpec:
    """Render one next-state prompt and retain the exact checkpoint character span."""
    register = bit_text(condition["symbols"])
    prefix = _prefix(case) + "Checkpoint possible states: ["
    suffix = (
        "]\nApply FINAL exactly once to the identified checkpoint state.\nFINAL: "
        + rule_text(case["final_rule"], int(case["bits"]))
        + ".\nReturn only the final state as a decimal integer.\nAnswer="
    )
    return PromptSpec(
        text=prefix + register + suffix,
        checkpoint_start=len(prefix),
        checkpoint_end=len(prefix) + len(register),
    )


def render_write_prompt(case: dict[str, Any]) -> str:
    """Ask the model to externalize the state immediately before the final rule."""
    return (
        _prefix(case)
        + "Return only the decimal state after those steps and before any further "
        "operation.\nCheckpoint="
    )


def build_qualification_benchmark(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build held-out pointer programs across controlled history lengths."""
    widths = tuple(int(value) for value in config.get("bits", (2, 3)))
    histories = tuple(int(value) for value in config.get("history_steps", (1, 2, 4, 8)))
    examples = int(config.get("examples_per_cell", 12))
    seed = int(config.get("seed", 0))
    cases = []
    for history_steps in histories:
        for width in widths:
            for example in range(examples):
                case = build_case(
                    family="pointer",
                    width=width,
                    example_index=example,
                    seed=seed,
                    history_steps=history_steps,
                )
                case["id"] = f"pointer_h{history_steps}_b{width}_{example:04d}"
                case["history_steps"] = history_steps
                cases.append(case)
    return cases


def qualification_condition_specs(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Define direct and token-matched register qualification conditions."""
    return [
        {
            "name": "direct",
            "checkpoint_value": None,
            "checkpoint_valid": None,
            "expected_next_state": int(case["next_state"]),
        },
        {
            "name": "none",
            "checkpoint_value": int(case["counterfactual_state"]),
            "checkpoint_valid": 0,
            "expected_next_state": int(case["next_state"]),
        },
        {
            "name": "gold",
            "checkpoint_value": int(case["current_state"]),
            "checkpoint_valid": 1,
            "expected_next_state": int(case["next_state"]),
        },
        {
            "name": "counterfactual",
            "checkpoint_value": int(case["counterfactual_state"]),
            "checkpoint_valid": 1,
            "expected_next_state": int(case["counterfactual_next_state"]),
        },
        {
            "name": "invalid",
            "checkpoint_value": int(case["random_state"]),
            "checkpoint_valid": 0,
            "expected_next_state": int(case["next_state"]),
        },
    ]


def render_qualification_prompt(
    case: dict[str, Any], condition: dict[str, Any]
) -> PromptSpec:
    """Render the fixed-shape value/validity register program."""
    if condition["name"] == "direct":
        raise ValueError("The direct control has no register span")
    width = int(case["bits"])
    history = "\n".join(
        f"Step {index}: {rule_text(rule, width)}."
        for index, rule in enumerate(case["history"], 1)
    )
    prefix = (
        "Follow the lookup-table instructions exactly.\n"
        f"Start state: {int(case['initial_state'])}.\n"
        f"{history}\n"
    )
    register = (
        f"Checkpoint value: {int(condition['checkpoint_value'])}.\n"
        f"Checkpoint valid flag: {int(condition['checkpoint_valid'])}."
    )
    suffix = (
        "\nIf the valid flag is 1, replace the current state with the checkpoint "
        "value. If it is 0, ignore the checkpoint value and keep the state from "
        "the numbered steps.\n"
        f"FINAL: {rule_text(case['final_rule'], width)}. Apply FINAL exactly once.\n"
        "Return only the resulting decimal state.\nAnswer="
    )
    return PromptSpec(
        text=prefix + register + suffix,
        checkpoint_start=len(prefix),
        checkpoint_end=len(prefix) + len(register),
    )


def render_qualification_direct_prompt(case: dict[str, Any]) -> str:
    """Render the one-step positive control without history or register parsing."""
    return (
        f"Apply this operation to state {int(case['current_state'])}: "
        f"{rule_text(case['final_rule'], int(case['bits']))}. "
        "Return only the resulting decimal state.\nAnswer="
    )
