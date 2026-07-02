"""Run objective-relative segmentation over token-aligned reasoning sentences."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.experiments.thought_unit_cache import (
    build_feature_cache,
    evenly_select_traces,
    load_feature_cache,
    load_partitions,
    trace_view,
)
from src.experiments.thought_unit_features import (
    accumulated_gram_spectra,
    compare_answer_curves,
    cosine_distance,
    cross_rollout_answer_scores,
    fit_sentence_pca,
    linear_hsic_alignment,
    load_h4_projection,
    normalize_rows,
    question_split,
    raw_geometry,
    sentence_means,
    sentence_update_counts,
    terminal_answer_sentence,
)
from src.experiments.thought_unit_outputs import (
    write_boundary_examples,
    write_matrix_csv,
    write_partitions,
    write_plots,
    write_records_csv,
)
from src.experiments.thought_unit_partitions import (
    best_worst_case,
    candidate_partitions,
    evaluate_partitions,
    grouped_bootstrap_summary,
    objective_costs,
    objective_rank_correlations,
)
from src.experiments.thought_unit_probes import (
    boundary_features,
    boundary_labels,
    evaluate_projection_coherence,
    evaluate_supervised_boundaries,
    fit_boundary_model,
    matched_probability_labels,
)
from src.experiments.thought_unit_signals import (
    answer_proxy_diagnostics,
    apply_gold_answer_scores,
    evaluate_parser_robustness,
    fit_correctness_curves,
    fit_gram_state_scores,
    merge_short_sentence_cache,
    prefix_features,
    weighted_group_means,
)
from src.experiments.thought_unit_types import (
    OBJECTIVES,
    ORACLE_NAMES,
    PRIMARY_FRACTION,
    TraceSpec,
    TraceView,
)


def run_thought_units(
    run_path: Path,
    *,
    projection_path: Path | None = None,
    gold_answer_run: Path | None = None,
    per_sample: int = 10,
    pca_dim: int = 64,
    gram_dim: int = 16,
    rebuild_features: bool = False,
    max_traces: int | None = None,
) -> Path:
    """Build sentence features, run matched-budget tests, and write reports.

    Args:
        run_path: Run directory containing the configuration and artifacts.
        projection_path: Path to a saved projection artifact.
        gold_answer_run: Run directory containing gold-answer activation captures.
        per_sample: Maximum number of trajectories retained per sample.
        pca_dim: Maximum PCA output dimension.
        gram_dim: Number of Gram-spectrum dimensions to retain.
        rebuild_features: Whether to ignore and replace the sentence feature cache.
        max_traces: Maximum number of traces to evaluate.

    Returns:
        The path of the written or discovered artifact.
    """
    out_dir = run_path / "analysis" / "experiments" / "thought_units"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "features.npz"
    trace_path = out_dir / "traces.jsonl"
    if rebuild_features or not feature_path.exists() or not trace_path.exists():
        build_feature_cache(
            run_path,
            out_dir,
            projection_path=projection_path,
            per_sample=per_sample,
            pca_dim=pca_dim,
            gram_dim=gram_dim,
        )

    cache = load_feature_cache(feature_path, trace_path)
    feature_metadata = json.loads(
        (out_dir / "feature_metadata.json").read_text(encoding="utf-8")
    )
    trace_count = len(cache["records"])
    selected_indices = list(range(trace_count))
    if max_traces is not None and max_traces < trace_count:
        selected_indices = evenly_select_traces(cache["records"], max_traces)
    answer_information = {
        "status": "proxy_only",
        "metric": (
            "linear-kernel normalized HSIC between each sentence mean and "
            "a correct terminal-answer sentence mean from another rollout "
            "of the same question"
        ),
        "limitation": (
            "The run lacks separately teacher-forced gold-answer activations, "
            "so this is not a replication of Qian et al.'s Gaussian-HSIC curve."
        ),
        "diagnostics": answer_proxy_diagnostics(cache, selected_indices),
    }
    if gold_answer_run is not None:
        cache, answer_information = apply_gold_answer_scores(
            cache,
            gold_answer_run,
            selected_indices=selected_indices,
            output_path=out_dir / "gold_answer_scores.npz",
        )
    correctness_curves, correctness_report = fit_correctness_curves(
        cache, selected_indices
    )
    gram_scores, gram_report = fit_gram_state_scores(cache, selected_indices)
    evaluation = evaluate_partitions(
        cache,
        selected_indices,
        correctness_curves,
        gram_scores,
    )
    supervised = evaluate_supervised_boundaries(
        cache,
        selected_indices,
        evaluation["primary_partitions"],
    )
    projection_results = evaluate_projection_coherence(
        cache,
        selected_indices,
        evaluation["primary_partitions"],
    )
    parser_robustness = evaluate_parser_robustness(cache, selected_indices)

    write_matrix_csv(
        out_dir / "objective_matrix.csv",
        evaluation["primary"]["utilities"],
    )
    write_records_csv(
        out_dir / "boundary_agreement.csv",
        evaluation["primary"]["boundary_agreement"],
    )
    write_records_csv(
        out_dir / "supervised_transfer.csv",
        supervised["rows"],
    )
    write_boundary_examples(
        run_path,
        out_dir / "boundary_examples.jsonl",
        cache,
        selected_indices,
        evaluation["primary_partitions"],
    )
    write_partitions(
        out_dir / "partitions.jsonl",
        cache,
        selected_indices,
        evaluation["primary_partitions"],
    )
    write_plots(
        out_dir,
        evaluation["primary"]["utilities"],
        evaluation["primary"]["pareto_front"],
    )

    report = {
        "experiment": "sentence_lattice_objective_relative_thought_units",
        "source_run": run_path.as_posix(),
        "scope": (
            "sentence-lattice first pass; this does not establish a token-level "
            "no-free-lunch theorem"
        ),
        "selection": {
            "trajectories": len(selected_indices),
            "available_trajectories": trace_count,
            "train_trajectories": sum(
                bool(cache["records"][index]["train"]) for index in selected_indices
            ),
            "test_trajectories": sum(
                not bool(cache["records"][index]["train"]) for index in selected_indices
            ),
            "questions": len(
                {cache["records"][index]["sample_id"] for index in selected_indices}
            ),
        },
        "feature_extraction": feature_metadata,
        "controls": {
            "question_disjoint_split": True,
            "matched_boundary_fractions": [0.1, 0.2, 0.3],
            "primary_boundary_fraction": PRIMARY_FRACTION,
            "random_repetitions": 12,
            "oracle": "exact dynamic programming with a fixed segment count",
            "selection_leakage": (
                "PCA and correctness probe fit only on training questions; "
                "objective labels are used only by named oracles and supervised "
                "adversaries"
            ),
        },
        "answer_information": answer_information,
        "gram_states": gram_report,
        "correctness_probe": correctness_report,
        "partition_evaluation": {
            key: value
            for key, value in evaluation.items()
            if key != "primary_partitions"
        },
        "supervised_change_points": {
            key: value for key, value in supervised.items() if key != "rows"
        },
        "projection_coherence": projection_results,
        "parser_robustness": parser_robustness,
        "artifacts": {
            "features": feature_path.as_posix(),
            "traces": trace_path.as_posix(),
            "objective_matrix": (out_dir / "objective_matrix.csv").as_posix(),
            "boundary_agreement": (out_dir / "boundary_agreement.csv").as_posix(),
            "boundary_examples": (out_dir / "boundary_examples.jsonl").as_posix(),
            "partitions": (out_dir / "partitions.jsonl").as_posix(),
            "supervised_transfer": (out_dir / "supervised_transfer.csv").as_posix(),
            "objective_plot": (out_dir / "objective_matrix.png").as_posix(),
            "regret_plot": (out_dir / "regret_matrix.png").as_posix(),
            "pareto_plot": (out_dir / "pareto_object_compression.png").as_posix(),
            "gold_answer_scores": (
                (out_dir / "gold_answer_scores.npz").as_posix()
                if (out_dir / "gold_answer_scores.npz").exists()
                else None
            ),
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def run_prompt_transfer(
    source_run: Path,
    target_runs: list[Path],
) -> Path:
    """Train H4 boundary detectors on disjoint questions and test prompt transfer.

    Args:
        source_run: Source run directory containing the original artifacts.
        target_runs: Prompt-transfer run directories to evaluate.

    Returns:
        The path of the written or discovered artifact.
    """
    source_dir = source_run / "analysis" / "experiments" / "thought_units"
    source_cache = load_feature_cache(
        source_dir / "features.npz",
        source_dir / "traces.jsonl",
    )
    source_partitions = load_partitions(source_dir / "partitions.jsonl")
    target_payloads = []
    excluded_questions: set[str] = set()
    for run in target_runs:
        directory = run / "analysis" / "experiments" / "thought_units"
        cache = load_feature_cache(
            directory / "features.npz",
            directory / "traces.jsonl",
        )
        partitions = load_partitions(directory / "partitions.jsonl")
        target_payloads.append((run, cache, partitions))
        excluded_questions.update(
            str(record["sample_id"]) for record in cache["records"]
        )

    train_x: list[np.ndarray] = []
    train_labels: dict[str, list[np.ndarray]] = defaultdict(list)
    train_questions: set[str] = set()
    for index, record in enumerate(source_cache["records"]):
        question = str(record["sample_id"])
        if question in excluded_questions:
            continue
        trace = trace_view(source_cache, index)
        features = boundary_features(trace.h4)
        methods = source_partitions[(question, int(record["seed"]))]
        train_x.append(features)
        train_questions.add(question)
        for objective in OBJECTIVES:
            train_labels[objective].append(
                boundary_labels(
                    len(features),
                    methods[ORACLE_NAMES[objective]],
                )
            )
    x_train = np.concatenate(train_x).astype(np.float32)
    y_train = {
        objective: np.concatenate(values) for objective, values in train_labels.items()
    }
    models = {
        objective: fit_boundary_model(x_train, y_train[objective], nonlinear=True)
        for objective in OBJECTIVES
    }

    rows: list[dict[str, Any]] = []
    for run, cache, partitions in target_payloads:
        test_x: list[np.ndarray] = []
        test_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        slices: list[tuple[int, int, int]] = []
        cursor = 0
        for index, record in enumerate(cache["records"]):
            trace = trace_view(cache, index)
            features = boundary_features(trace.h4)
            methods = partitions[(str(record["sample_id"]), int(record["seed"]))]
            test_x.append(features)
            for objective in OBJECTIVES:
                test_labels[objective].append(
                    boundary_labels(
                        len(features),
                        methods[ORACLE_NAMES[objective]],
                    )
                )
            positives = len(methods[ORACLE_NAMES["answer"]])
            slices.append((cursor, cursor + len(features), positives))
            cursor += len(features)
        x_test = np.concatenate(test_x).astype(np.float32)
        y_test = {
            objective: np.concatenate(values)
            for objective, values in test_labels.items()
        }
        for trained_on, model in models.items():
            probabilities = model.predict_proba(x_test)[:, 1]
            predicted = matched_probability_labels(probabilities, slices)
            for evaluated_on in OBJECTIVES:
                rows.append(
                    {
                        "target_run": run.name,
                        "trained_on": trained_on,
                        "evaluated_on": evaluated_on,
                        "roc_auc": float(
                            roc_auc_score(y_test[evaluated_on], probabilities)
                        ),
                        "average_precision": float(
                            average_precision_score(y_test[evaluated_on], probabilities)
                        ),
                        "matched_budget_f1": float(
                            f1_score(y_test[evaluated_on], predicted)
                        ),
                    }
                )

    out_dir = source_dir
    write_records_csv(out_dir / "prompt_transfer.csv", rows)
    report = {
        "experiment": "question_disjoint_prompt_transfer",
        "space": "shared H4 operation-supervised projection",
        "source_run": source_run.as_posix(),
        "target_runs": [run.as_posix() for run in target_runs],
        "excluded_target_questions": len(excluded_questions),
        "training_questions": len(train_questions),
        "training_boundaries": len(x_train),
        "targets": {
            run.name: {
                "trajectories": len(cache["records"]),
                "questions": len(
                    {str(record["sample_id"]) for record in cache["records"]}
                ),
            }
            for run, cache, _partitions in target_payloads
        },
        "rows": rows,
    }
    report_path = out_dir / "prompt_transfer.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


__all__ = [
    "run_thought_units",
    "run_prompt_transfer",
    "build_feature_cache",
    "question_split",
    "fit_sentence_pca",
    "load_h4_projection",
    "sentence_means",
    "accumulated_gram_spectra",
    "raw_geometry",
    "cosine_distance",
    "normalize_rows",
    "linear_hsic_alignment",
    "cross_rollout_answer_scores",
    "apply_gold_answer_scores",
    "compare_answer_curves",
    "terminal_answer_sentence",
    "sentence_update_counts",
    "load_feature_cache",
    "load_partitions",
    "trace_view",
    "evenly_select_traces",
    "merge_short_sentence_cache",
    "weighted_group_means",
    "evaluate_parser_robustness",
    "prefix_features",
    "fit_correctness_curves",
    "fit_gram_state_scores",
    "answer_proxy_diagnostics",
    "objective_costs",
    "candidate_partitions",
    "evaluate_partitions",
    "grouped_bootstrap_summary",
    "objective_rank_correlations",
    "best_worst_case",
    "boundary_features",
    "evaluate_supervised_boundaries",
    "fit_boundary_model",
    "boundary_labels",
    "matched_probability_labels",
    "evaluate_projection_coherence",
    "write_matrix_csv",
    "write_records_csv",
    "write_boundary_examples",
    "write_partitions",
    "write_plots",
    "TraceSpec",
    "TraceView",
    "OBJECTIVES",
    "ORACLE_NAMES",
    "PRIMARY_FRACTION",
]
