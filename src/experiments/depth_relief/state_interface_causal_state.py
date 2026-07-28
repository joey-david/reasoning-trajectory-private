"""Reduce the rate, gauge, and proof-depth causal-state tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.runtime.artifact_store import write_json
from src.runtime.config import load_config

from .metrics import bootstrap_mean_ci, cluster_bootstrap_mean_ci
from .state_interface_challenge import (
    _read,
    _write_summary,
    challenge_dir,
    configured_challenge_profiles,
)
from .state_interface_causal_state_plots import (
    plot_consumer_basis,
    plot_depth_length,
    plot_gauge,
    plot_rate,
)


def _mean(summary: dict[str, Any], *keys: str) -> float:
    value: Any = summary
    for key in keys:
        value = value[key]
    return float(value["mean"] if isinstance(value, dict) and "mean" in value else value)


def _challenge_information(run_path: Path, profile: str) -> dict[str, Any]:
    from .state_handoff_information import (
        conditional_entropy,
        discrete_entropy,
        mutual_information,
    )

    rows = _read(challenge_dir(run_path, profile) / "interface_cases.jsonl")

    def summarize(pairs: list[tuple[int, int]]) -> dict[str, float]:
        return {
            "state_information_bits": mutual_information(pairs),
            "state_given_code_bits": conditional_entropy(pairs),
            "code_given_state_bits": conditional_entropy(
                (code, state) for state, code in pairs
            ),
            "code_entropy_bits": discrete_entropy(code for _, code in pairs),
        }

    true_pairs = [
        (int(row["current_state"]), int(row["true_code"])) for row in rows
    ]
    predicted_pairs = [
        (int(row["current_state"]), int(row["predicted_code"]))
        for row in rows
        if row["predicted_code"] is not None
    ]
    return {
        "true": summarize(true_pairs),
        "predicted": summarize(predicted_pairs),
        "invalid_code_count": len(rows) - len(predicted_pairs),
    }


def _pooled_horizon_advantage(
    run_path: Path,
    profiles: dict[str, dict[str, Any]],
    sources: list[str],
    condition: str,
    horizon: int,
) -> dict[str, Any]:
    differences: list[int] = []
    clusters: list[str] = []
    for source in sources:
        profile = f"{source}__{condition}__length"
        programs = _read(challenge_dir(run_path, profile) / "programs.jsonl")
        interface = {
            str(row["id"]): row
            for row in _read(
                challenge_dir(run_path, profile) / "interface_cases.jsonl"
            )
        }
        owner = str(profiles[profile]["outcome_owner_profile"])
        outcome = {
            str(row["id"]): row
            for row in _read(
                challenge_dir(run_path, owner) / "outcome_cases.jsonl"
            )
        }
        for program in programs:
            if int(program["history_steps"]) != horizon:
                continue
            case_id = str(program["id"])
            interface_correct = bool(
                interface[case_id]["predicted_final"]
                and interface[case_id]["predicted_final"][
                    "is_expected_unconstrained"
                ]
            )
            outcome_correct = bool(
                outcome[case_id]["conditions"]["one_pass_compose"][
                    "is_expected_unconstrained"
                ]
            )
            differences.append(int(interface_correct) - int(outcome_correct))
            clusters.append(f"{source}:{program['program_context']}")
    return cluster_bootstrap_mean_ci(differences, clusters, seed=85_401)


def compare_causal_state_phase(run_path: Path) -> dict[str, Any]:
    """Apply the locked causal-state tests and render their figures."""
    config = load_config(run_path)
    experiment = config["state_interface_causal_state"]
    profiles = configured_challenge_profiles(run_path)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    missing = []
    for profile in profiles:
        summary = _write_summary(run_path, profile)
        if not summary.get("complete"):
            missing.append(profile)
            continue
        source, condition, template = profile.split("__", 2)
        results.setdefault(source, {}).setdefault(condition, {})[template] = summary

    sources = [str(value) for value in experiment["primary_sources"]]
    condition_names = {
        str(key): str(value) for key, value in experiment["conditions"].items()
    }
    complete = not missing
    checks: dict[str, bool] = {"all_profiles_complete": complete}
    metrics: dict[str, Any] = {}
    if complete:
        information: dict[str, dict[str, dict[str, Any]]] = {}
        for source_spec in config["state_interface_challenge_matrix"]["sources"]:
            source = str(source_spec["name"])
            for condition_spec in source_spec["conditions"]:
                condition = (
                    str(condition_spec)
                    if isinstance(condition_spec, str)
                    else str(condition_spec["name"])
                )
                information.setdefault(source, {})[condition] = (
                    _challenge_information(
                        run_path, f"{source}__{condition}__full_support"
                    )
                )
        canonical = condition_names["canonical"]
        aliased = condition_names["aliased"]
        lossy = condition_names["lossy"]
        padded = condition_names["padded"]
        canonical_full = [
            _mean(results[source][canonical]["full_support"], "overall", "exact_state_accuracy")
            for source in sources
        ]
        canonical_full_worst_state = [
            min(
                _mean(
                    results[source][canonical]["full_support"],
                    "by_current_state",
                    str(state),
                    "exact_state_accuracy",
                )
                for state in range(16)
            )
            for source in sources
        ]
        canonical_depth3 = [
            _mean(
                results[source][canonical]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        canonical_h256 = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "256",
                "interface_accuracy",
            )
            for source in sources
        ]
        canonical_h256_exact = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "256",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        canonical_h256_local_exact = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "256",
                "local_exact_state_closure",
            )
            for source in sources
        ]
        canonical_h256_false_positive = [
            float(
                results[source][canonical]["length"][
                    "by_surface_horizon"
                ]["256"]["false_positive_fact_rate"]
            )
            for source in sources
        ]
        length_drop = [
            _mean(
                results[source][canonical]["length"],
                "by_surface_horizon",
                "16",
                "exact_state_accuracy",
            )
            - canonical_h256_exact[index]
            for index, source in enumerate(sources)
        ]
        padded_depth3 = [
            _mean(
                results[source][padded]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        aliased_depth3 = [
            _mean(
                results[source][aliased]["balanced_depth"],
                "by_active_transition_count",
                "3",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        padded_full = [
            _mean(
                results[source][padded]["full_support"],
                "overall",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        aliased_full = [
            _mean(
                results[source][aliased]["full_support"],
                "overall",
                "exact_state_accuracy",
            )
            for source in sources
        ]
        padded_alias_full_gap = [
            padded_full[index] - aliased_full[index]
            for index in range(len(sources))
        ]
        padded_alias_depth3_gap = [
            padded_depth3[index] - aliased_depth3[index]
            for index in range(len(sources))
        ]
        canonical_alias_depth3_gap = [
            canonical_depth3[index] - aliased_depth3[index]
            for index in range(len(sources))
        ]
        canonical_padded_gaps = []
        for source in sources:
            gaps = [
                abs(
                    _mean(results[source][canonical][template], *keys)
                    - _mean(results[source][padded][template], *keys)
                )
                for template, keys in (
                    ("full_support", ("overall", "exact_state_accuracy")),
                    (
                        "balanced_depth",
                        (
                            "by_active_transition_count",
                            "3",
                            "exact_state_accuracy",
                        ),
                    ),
                    (
                        "length",
                        (
                            "by_surface_horizon",
                            "256",
                            "exact_state_accuracy",
                        ),
                    ),
                )
            ]
            canonical_padded_gaps.append(max(gaps))
        gold_code_accuracy = {
            condition: [
                _mean(
                    results[source][condition]["full_support"],
                    "overall",
                    "gold_code_continuation_accuracy",
                )
                for source in sources
            ]
            for condition in (canonical, padded, aliased)
        }
        seed1 = sources[0]
        lossy_gap = canonical_full[0] - _mean(
            results[seed1][lossy]["full_support"],
            "overall",
            "exact_state_accuracy",
        )
        h256_advantage = _pooled_horizon_advantage(
            run_path, profiles, sources, canonical, 256
        )
        padded_alias_ci = bootstrap_mean_ci(padded_alias_full_gap, seed=85_106)
        information_contract = (
            abs(
                information[seed1][lossy]["true"]["state_given_code_bits"]
                - 1.0
            )
            < 1e-9
            and all(
                abs(
                    information[source][condition]["true"][
                        "state_given_code_bits"
                    ]
                )
                < 1e-9
                for source in sources
                for condition in (canonical, padded, aliased)
            )
            and all(
                abs(
                    information[source][padded]["true"][
                        "code_given_state_bits"
                    ]
                )
                < 1e-9
                and abs(
                    information[source][aliased]["true"][
                        "code_given_state_bits"
                    ]
                    - 1.0
                )
                < 1e-9
                for source in sources
            )
        )
        metrics = {
            "canonical_full_support_exact": {
                "per_seed": canonical_full,
                **bootstrap_mean_ci(canonical_full, seed=85_101),
            },
            "canonical_full_support_worst_state_exact": {
                "per_seed": canonical_full_worst_state,
                **bootstrap_mean_ci(canonical_full_worst_state, seed=85_109),
            },
            "canonical_depth3_exact": {
                "per_seed": canonical_depth3,
                **bootstrap_mean_ci(canonical_depth3, seed=85_102),
            },
            "canonical_h256_final": {
                "per_seed": canonical_h256,
                **bootstrap_mean_ci(canonical_h256, seed=85_103),
            },
            "canonical_h256_exact": {
                "per_seed": canonical_h256_exact,
                **bootstrap_mean_ci(canonical_h256_exact, seed=85_104),
            },
            "canonical_h256_local_exact_closure": {
                "per_seed": canonical_h256_local_exact,
                **bootstrap_mean_ci(canonical_h256_local_exact, seed=85_111),
            },
            "canonical_h256_false_positive_fact_rate": {
                "per_seed": canonical_h256_false_positive,
                **bootstrap_mean_ci(canonical_h256_false_positive, seed=85_112),
            },
            "canonical_h256_minus_outcome": h256_advantage,
            "canonical_h16_minus_h256_exact": {
                "per_seed": length_drop,
                **bootstrap_mean_ci(length_drop, seed=85_105),
            },
            "padded_minus_aliased_full_support_exact": {
                "per_seed": padded_alias_full_gap,
                **padded_alias_ci,
            },
            "padded_minus_aliased_depth3_exact": {
                "per_seed": padded_alias_depth3_gap,
                **bootstrap_mean_ci(padded_alias_depth3_gap, seed=85_110),
            },
            "canonical_minus_aliased_depth3_exact": {
                "per_seed": canonical_alias_depth3_gap,
                **bootstrap_mean_ci(canonical_alias_depth3_gap, seed=85_107),
            },
            "canonical_padded_max_exact_gap": {
                "per_seed": canonical_padded_gaps,
                **bootstrap_mean_ci(canonical_padded_gaps, seed=85_108),
            },
            "gold_code_continuation": gold_code_accuracy,
            "canonical_minus_lossy_full_support_exact_seed1": lossy_gap,
            "full_support_code_information": information,
        }
        gates = experiment["gates"]
        checks.update(
            canonical_full_support=all(
                value >= float(gates["min_canonical_full_support_exact_each_seed"])
                for value in canonical_full
            ),
            canonical_full_support_each_state=all(
                value
                >= float(
                    gates["min_canonical_full_support_exact_each_state"]
                )
                for value in canonical_full_worst_state
            ),
            canonical_active_depth=all(
                value >= float(gates["min_canonical_depth3_exact_each_seed"])
                for value in canonical_depth3
            ),
            canonical_long_horizon_final=all(
                value >= float(gates["min_canonical_h256_final_each_seed"])
                for value in canonical_h256
            ),
            canonical_long_horizon_exact=all(
                value >= float(gates["min_canonical_h256_exact_each_seed"])
                for value in canonical_h256_exact
            ),
            canonical_transition_closure=all(
                value
                >= float(
                    gates[
                        "min_canonical_h256_local_exact_closure_each_seed"
                    ]
                )
                for value in canonical_h256_local_exact
            ),
            canonical_noop_control=all(
                value
                <= float(
                    gates[
                        "max_canonical_h256_false_positive_fact_rate_each_seed"
                    ]
                )
                for value in canonical_h256_false_positive
            ),
            canonical_beats_one_pass=(
                h256_advantage["mean"]
                >= float(gates["min_mean_canonical_h256_minus_outcome"])
                and h256_advantage["ci95"][0]
                > float(gates["min_canonical_h256_advantage_ci_low"])
            ),
            surface_length_stable=(
                sum(length_drop) / len(length_drop)
                <= float(gates["max_mean_canonical_h16_to_h256_drop"])
            ),
            below_rate_is_lossy=(
                lossy_gap >= float(gates["min_lossy_exact_gap_seed1"])
            ),
            unused_capacity_is_not_harmful=(
                sum(canonical_padded_gaps) / len(canonical_padded_gaps)
                <= float(gates["max_mean_canonical_padded_exact_gap"])
            ),
            aliases_are_harmful=(
                padded_alias_ci["mean"]
                >= float(gates["min_mean_padded_alias_exact_gap"])
                and padded_alias_ci["ci95"][0]
                > float(gates["min_padded_alias_exact_gap_ci_low"])
            ),
            gold_codes_are_causal_and_sufficient=all(
                value >= float(gates["min_gold_code_continuation_each_seed"])
                for values in gold_code_accuracy.values()
                for value in values
            ),
            exact_information_contract=information_contract,
        )

    status = "pending" if not complete else ("passed" if all(checks.values()) else "failed")
    summary = {
        "schema_version": 1,
        "hypothesis": (
            "Reusable proof state requires a sufficient, canonical, "
            "transition-closed interface; active state changes, not padded "
            "surface length, set the extrapolation burden."
        ),
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "missing_profiles": missing,
        "profile_count": len(profiles),
        "complete_profile_count": len(profiles) - len(missing),
    }
    output = run_path / "evaluation"
    write_json(output / "causal_state_phase_summary.json", summary)
    if complete:
        plot_rate(output / "rate_sufficiency.png", results)
        plot_gauge(
            output / "gauge_closure.png",
            results=results,
            information=information,
            sources=sources,
        )
        plot_depth_length(
            output / "depth_length.png", results=results, sources=sources
        )
        plot_consumer_basis(
            output / "consumer_basis.png", run_path=run_path
        )
    return summary
