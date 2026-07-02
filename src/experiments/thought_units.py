"""Evaluate objective-relative partitions over token-aligned reasoning sentences."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import torch

from src.analysis.step_classification.segmentation import StepSegment, build_segments
from src.analysis.token_alignment import build_token_spans
from src.experiments.common import balanced_generation_rows
from src.experiments.sentence_lattice import (
    boundary_f1,
    boundary_jaccard,
    fixed_boundaries,
    object_update_costs,
    optimal_partition,
    pareto_front,
    partition_cost,
    partition_variation_of_information,
    random_boundaries,
    squared_error_costs,
    top_boundaries,
)
from src.experiments.symbolic import SymbolicUpdate, extract_symbolic_updates
from src.runtime.artifact_store import load_hidden_states_npz
from src.runtime.data import write_jsonl


OBJECTIVES = ("answer", "object", "correctness", "compression")
ORACLE_NAMES = {objective: f"oracle_{objective}" for objective in OBJECTIVES}
PRIMARY_FRACTION = 0.2
_LIST_MARKER_RE = re.compile(r"^(?:\d+|[A-Za-z])[.)]$")


@dataclass(slots=True)
class TraceSpec:
    """Hold one generation and its fixed token-aligned sentence lattice."""

    row: dict[str, Any]
    segments: list[StepSegment]
    updates: list[SymbolicUpdate]
    train: bool


@dataclass(slots=True)
class TraceView:
    """Expose compact per-sentence features for one cached trajectory."""

    sample_id: str
    seed: int
    is_correct: bool
    train: bool
    raw: np.ndarray
    pca: np.ndarray
    h4: np.ndarray
    gram: np.ndarray
    raw_geometry: np.ndarray
    answer_score: np.ndarray
    update_count: np.ndarray
    token_count: np.ndarray


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
    """Build sentence features, run matched-budget tests, and write reports."""
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
                not bool(cache["records"][index]["train"])
                for index in selected_indices
            ),
            "questions": len(
                {
                    cache["records"][index]["sample_id"]
                    for index in selected_indices
                }
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
            "boundary_agreement": (
                out_dir / "boundary_agreement.csv"
            ).as_posix(),
            "boundary_examples": (
                out_dir / "boundary_examples.jsonl"
            ).as_posix(),
            "partitions": (out_dir / "partitions.jsonl").as_posix(),
            "supervised_transfer": (
                out_dir / "supervised_transfer.csv"
            ).as_posix(),
            "objective_plot": (
                out_dir / "objective_matrix.png"
            ).as_posix(),
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
    """Train H4 boundary detectors on disjoint questions and test prompt transfer."""
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
        objective: np.concatenate(values)
        for objective, values in train_labels.items()
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
            methods = partitions[
                (str(record["sample_id"]), int(record["seed"]))
            ]
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
                            average_precision_score(
                                y_test[evaluated_on], probabilities
                            )
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


def build_feature_cache(
    run_path: Path,
    out_dir: Path,
    *,
    projection_path: Path | None,
    per_sample: int,
    pca_dim: int,
    gram_dim: int,
) -> None:
    """Stream activations into compact sentence-level projection features."""
    rows = balanced_generation_rows(run_path, per_sample=per_sample)
    token_spans = build_token_spans(run_path, rows)
    train_ids = question_split(rows)
    specs: list[TraceSpec] = []
    for row, spans in zip(rows, token_spans):
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        updates = extract_symbolic_updates(
            str(row.get("produced_text", "")),
            spans,
            token_count=len(row.get("generated_token_ids", [])),
        )
        if len(segments) >= 3:
            specs.append(
                TraceSpec(
                    row=row,
                    segments=segments,
                    updates=updates,
                    train=str(row["sample_id"]) in train_ids,
                )
            )
    pca = fit_sentence_pca(run_path, specs, pca_dim=pca_dim)
    h4_weight, resolved_projection = load_h4_projection(
        run_path, projection_path=projection_path
    )

    raw_rows: list[np.ndarray] = []
    pca_rows: list[np.ndarray] = []
    h4_rows: list[np.ndarray] = []
    gram_rows: list[np.ndarray] = []
    geometry_rows: list[np.ndarray] = []
    update_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    offsets = [0]
    parser_counts: Counter[str] = Counter()

    for spec in specs:
        states, layers = load_hidden_states_npz(
            run_path / spec.row["hidden_states_file"]
        )
        if -1 not in layers:
            raise ValueError(f"Last layer missing for {spec.row['sample_id']}")
        layer_states = states[:, layers.index(-1)].astype(np.float32)
        means, token_counts = sentence_means(layer_states, spec.segments)
        pca_means = pca.transform(means).astype(np.float32)
        h4_means = normalize_rows(means @ h4_weight.T)
        gram = accumulated_gram_spectra(
            layer_states,
            spec.segments,
            pca,
            dimension=gram_dim,
        )
        geometry = raw_geometry(means)
        answer_index = terminal_answer_sentence(spec)
        update_counts = sentence_update_counts(spec.segments, spec.updates)

        raw_rows.append(means.astype(np.float16))
        pca_rows.append(pca_means.astype(np.float16))
        h4_rows.append(h4_means.astype(np.float16))
        gram_rows.append(gram.astype(np.float16))
        geometry_rows.append(geometry.astype(np.float32))
        update_rows.append(update_counts.astype(np.int16))
        token_rows.append(token_counts.astype(np.int16))
        offsets.append(offsets[-1] + len(means))

        fragment_count = int(np.sum(token_counts <= 2))
        marker_count = sum(
            bool(_LIST_MARKER_RE.fullmatch(segment.text.strip()))
            for segment in spec.segments
        )
        parser_counts.update(
            {
                "sentences": len(means),
                "short_fragments": fragment_count,
                "list_markers": marker_count,
            }
        )
        records.append(
            {
                "sample_id": str(spec.row["sample_id"]),
                "seed": int(spec.row["seed"]),
                "is_correct": bool(spec.row["is_correct"]),
                "train": spec.train,
                "sentences": len(means),
                "updates": int(update_counts.sum()),
                "answer_sentence": answer_index,
                "short_fragments": fragment_count,
                "list_markers": marker_count,
            }
        )

    answer_rows, answer_target_counts = cross_rollout_answer_scores(
        raw_rows, records
    )
    np.savez_compressed(
        out_dir / "features.npz",
        offsets=np.asarray(offsets, dtype=np.int64),
        raw=np.concatenate(raw_rows).astype(np.float16),
        pca=np.concatenate(pca_rows).astype(np.float16),
        h4=np.concatenate(h4_rows).astype(np.float16),
        gram=np.concatenate(gram_rows).astype(np.float16),
        raw_geometry=np.concatenate(geometry_rows).astype(np.float32),
        answer_score=np.concatenate(answer_rows).astype(np.float32),
        update_count=np.concatenate(update_rows).astype(np.int16),
        token_count=np.concatenate(token_rows).astype(np.int16),
    )
    write_jsonl(out_dir / "traces.jsonl", records)
    metadata = {
        "source_run": run_path.as_posix(),
        "layer": -1,
        "pca": {
            "dimensions": int(pca.n_components_),
            "fit": "at most 24 evenly spaced sentences per training trajectory",
            "questions_disjoint": True,
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        },
        "gram": {
            "dimensions": gram_dim,
            "definition": (
                "top eigenvalues of the accumulated token Gram matrix after "
                "training-only PCA whitening"
            ),
        },
        "h4_projection": resolved_projection.as_posix(),
        "parser_audit": {
            **dict(parser_counts),
            "short_fragment_rate": parser_counts["short_fragments"]
            / max(parser_counts["sentences"], 1),
            "list_marker_rate": parser_counts["list_markers"]
            / max(parser_counts["sentences"], 1),
            "policy": (
                "Preserve the repository sentence parser exactly; fragments are "
                "reported rather than silently merged."
            ),
        },
        "answer_targets": {
            **answer_target_counts,
            "policy": (
                "Prefer a different correct rollout of the same question; use "
                "the trace itself only when no correct donor exists."
            ),
        },
    }
    (out_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def question_split(rows: list[dict[str, Any]]) -> set[str]:
    """Choose deterministic train questions without splitting their rollouts."""
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    unique = np.asarray(sorted(set(sample_ids)))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train, _test = next(splitter.split(unique, groups=unique))
    return set(unique[train])


def fit_sentence_pca(
    run_path: Path,
    specs: list[TraceSpec],
    *,
    pca_dim: int,
) -> PCA:
    """Fit a bounded training-only PCA sample over sentence means."""
    sampled: list[np.ndarray] = []
    for spec in specs:
        if not spec.train:
            continue
        states, layers = load_hidden_states_npz(
            run_path / spec.row["hidden_states_file"]
        )
        means, _counts = sentence_means(
            states[:, layers.index(-1)].astype(np.float32),
            spec.segments,
        )
        indices = np.linspace(
            0,
            len(means) - 1,
            min(len(means), 24),
            dtype=int,
        )
        sampled.append(means[indices])
    fit_matrix = np.concatenate(sampled).astype(np.float32)
    dimension = min(pca_dim, len(fit_matrix) - 1, fit_matrix.shape[1])
    model = PCA(
        n_components=dimension,
        whiten=True,
        svd_solver="randomized",
        random_state=42,
    )
    model.fit(fit_matrix)
    return model


def load_h4_projection(
    run_path: Path,
    *,
    projection_path: Path | None,
) -> tuple[np.ndarray, Path]:
    """Load the existing operation-supervised H4 projection matrix."""
    path = projection_path or (
        run_path
        / "analysis"
        / "experiments"
        / "h4_structural_contrast"
        / "layer-1_projection.pt"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    weight = payload["weight"].detach().cpu().numpy().astype(np.float32)
    return weight, path


def sentence_means(
    states: np.ndarray,
    segments: list[StepSegment],
) -> tuple[np.ndarray, np.ndarray]:
    """Average last-layer token states inside every sentence span."""
    means: list[np.ndarray] = []
    counts: list[int] = []
    for segment in segments:
        start = min(max(segment.token_start, 0), len(states) - 1)
        end = min(max(segment.token_end, start), len(states) - 1)
        values = states[start : end + 1]
        means.append(values.mean(axis=0))
        counts.append(len(values))
    return np.stack(means).astype(np.float32), np.asarray(counts, dtype=np.int32)


def accumulated_gram_spectra(
    states: np.ndarray,
    segments: list[StepSegment],
    pca: PCA,
    *,
    dimension: int,
) -> np.ndarray:
    """Compute Yu-style accumulated Gram spectra on the sentence lattice."""
    dimension = min(dimension, pca.n_components_)
    scale = np.sqrt(np.maximum(pca.explained_variance_[:dimension], 1e-8))
    projected = (
        (states - pca.mean_) @ pca.components_[:dimension].T
    ) / scale
    accumulated = np.zeros((dimension, dimension), dtype=np.float64)
    spectra: list[np.ndarray] = []
    for segment in segments:
        start = min(max(segment.token_start, 0), len(projected) - 1)
        end = min(max(segment.token_end, start), len(projected) - 1)
        values = projected[start : end + 1].astype(np.float64)
        accumulated += values.T @ values / max(len(values), 1)
        eigenvalues = np.linalg.eigvalsh(accumulated)[::-1]
        spectra.append(np.log1p(np.maximum(eigenvalues, 0.0)))
    return np.stack(spectra).astype(np.float32)


def raw_geometry(means: np.ndarray) -> np.ndarray:
    """Summarize raw-space magnitude, direction change, and curvature."""
    count = len(means)
    output = np.zeros((count, 4), dtype=np.float32)
    output[:, 0] = np.linalg.norm(means, axis=1)
    if count < 2:
        return output
    deltas = means[1:] - means[:-1]
    output[1:, 1] = np.linalg.norm(deltas, axis=1)
    output[1:, 2] = cosine_distance(means[:-1], means[1:])
    if count > 2:
        output[2:, 3] = cosine_distance(deltas[:-1], deltas[1:])
    return output


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise cosine distance with stable zero-norm handling."""
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.einsum("ij,ij->i", left, right) / np.maximum(denominator, 1e-8)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows while preserving all-zero vectors."""
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def linear_hsic_alignment(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute normalized linear-kernel HSIC against one target vector."""
    centered = values - values.mean(axis=1, keepdims=True)
    target_centered = target - target.mean()
    numerator = np.square(centered @ target_centered)
    denominator = np.einsum("ij,ij->i", centered, centered) * float(
        target_centered @ target_centered
    )
    return numerator / np.maximum(denominator, 1e-12)


def cross_rollout_answer_scores(
    raw_rows: list[np.ndarray],
    records: list[dict[str, Any]],
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Pair each trace with a correct same-question terminal answer state."""
    by_question: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_question[str(record["sample_id"])].append(index)
    scores: list[np.ndarray] = []
    counts: Counter[str] = Counter()
    for index, (means, record) in enumerate(zip(raw_rows, records)):
        candidates = [
            candidate
            for candidate in by_question[str(record["sample_id"])]
            if candidate != index and records[candidate]["is_correct"]
        ]
        if candidates:
            donor = candidates[index % len(candidates)]
            counts["cross_rollout_correct"] += 1
        else:
            donor = index
            counts["self_fallback_no_correct_donor"] += 1
        donor_answer = int(records[donor]["answer_sentence"])
        target = raw_rows[donor][donor_answer].astype(np.float32)
        scores.append(
            linear_hsic_alignment(means.astype(np.float32), target).astype(
                np.float32
            )
        )
        record["answer_target_seed"] = int(records[donor]["seed"])
        record["answer_target_is_self"] = donor == index
    return scores, dict(counts)


def apply_gold_answer_scores(
    cache: dict[str, Any],
    gold_answer_run: Path,
    *,
    selected_indices: list[int],
    output_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace proxy answer curves with captured gold-solution alignment curves."""
    manifest_path = gold_answer_run / "gold_answers" / "manifest.jsonl"
    manifest: dict[str, dict[str, Any]] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest[str(row["sample_id"])] = row

    targets: dict[str, np.ndarray] = {}
    for sample_id in {str(row["sample_id"]) for row in cache["records"]}:
        if sample_id not in manifest:
            raise ValueError(f"Gold-answer capture missing sample {sample_id}")
        row = manifest[sample_id]
        states, layers = load_hidden_states_npz(
            gold_answer_run / str(row["hidden_states_file"])
        )
        if -1 not in layers:
            raise ValueError(f"Gold-answer capture lacks layer -1 for {sample_id}")
        targets[sample_id] = states[:, layers.index(-1)].astype(np.float32).mean(
            axis=0
        )

    proxy = cache["answer_score"].astype(np.float32)
    score_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for index, record in enumerate(cache["records"]):
        start, end = cache["offsets"][index : index + 2]
        target = targets[str(record["sample_id"])]
        raw = cache["raw"][int(start) : int(end)].astype(np.float32)
        if raw.shape[1] != len(target):
            raise ValueError(
                f"Gold target width {len(target)} does not match "
                f"sentence width {raw.shape[1]}"
            )
        score_rows.append(linear_hsic_alignment(raw, target).astype(np.float32))
        target_rows.append(target)

    scores = np.concatenate(score_rows).astype(np.float32)
    updated = dict(cache)
    updated["answer_score"] = scores
    updated["_answer_targets"] = target_rows
    if output_path is not None:
        np.savez_compressed(
            output_path,
            offsets=cache["offsets"],
            answer_score=scores,
        )

    comparison = compare_answer_curves(
        cache["offsets"],
        proxy,
        scores,
        selected_indices,
        cache["records"],
    )
    metadata_path = gold_answer_run / "gold_answers" / "metadata.json"
    capture_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    information = {
        "status": "gold_solution_alignment",
        "metric": (
            "linear-kernel normalized HSIC between each generated sentence mean "
            "and the mean teacher-forced gold-solution state"
        ),
        "capture_run": gold_answer_run.as_posix(),
        "capture_context": capture_metadata.get("alignment"),
        "limitation": (
            "Gold solutions were captured after a lone BOS/EOS token without the "
            "question prompt. This is a canonical-solution alignment target, not "
            "a Gaussian-HSIC or mutual-information replication of Qian et al."
        ),
        "questions": len(targets),
        "diagnostics": answer_proxy_diagnostics(updated, selected_indices),
        "proxy_comparison": comparison,
    }
    return updated, information


def compare_answer_curves(
    offsets: np.ndarray,
    proxy: np.ndarray,
    gold: np.ndarray,
    selected_indices: list[int],
    records: list[dict[str, Any]],
) -> dict[str, float]:
    """Compare cross-rollout and gold-solution answer curves on held-out traces."""
    correlations: list[float] = []
    jaccards: list[float] = []
    for index in selected_indices:
        if bool(records[index]["train"]):
            continue
        start, end = offsets[index : index + 2]
        proxy_curve = proxy[int(start) : int(end)]
        gold_curve = gold[int(start) : int(end)]
        if np.std(proxy_curve) > 0 and np.std(gold_curve) > 0:
            correlation = spearmanr(proxy_curve, gold_curve).statistic
            if np.isfinite(correlation):
                correlations.append(float(correlation))
        boundary_count = max(int(round((len(proxy_curve) - 1) * PRIMARY_FRACTION)), 1)
        proxy_boundaries = top_boundaries(
            np.abs(np.diff(proxy_curve)),
            boundary_count,
        )
        gold_boundaries = top_boundaries(
            np.abs(np.diff(gold_curve)),
            boundary_count,
        )
        jaccards.append(boundary_jaccard(proxy_boundaries, gold_boundaries))
    return {
        "held_out_curve_spearman": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "held_out_top_boundary_jaccard": float(np.mean(jaccards)),
    }


def terminal_answer_sentence(spec: TraceSpec) -> int:
    """Locate the sentence containing the terminal extracted answer."""
    extracts = [update for update in spec.updates if update.operator == "EXTRACT"]
    target_token = extracts[-1].token_end if extracts else None
    if target_token is not None:
        for index, segment in enumerate(spec.segments):
            if segment.token_start <= target_token <= segment.token_end:
                return index
    for index in range(len(spec.segments) - 1, -1, -1):
        text = spec.segments[index].text.lower()
        if "answer:" in text or "final answer" in text:
            return index
    return len(spec.segments) - 1


def sentence_update_counts(
    segments: list[StepSegment],
    updates: list[SymbolicUpdate],
) -> np.ndarray:
    """Count symbolic update completions inside each sentence."""
    ends = np.asarray([segment.token_end for segment in segments])
    counts = np.zeros(len(segments), dtype=np.int32)
    for update in updates:
        sentence = int(np.searchsorted(ends, update.token_end, side="left"))
        if sentence < len(counts):
            counts[sentence] += 1
    return counts


def load_feature_cache(feature_path: Path, trace_path: Path) -> dict[str, Any]:
    """Load compact arrays and trace metadata into memory."""
    with np.load(feature_path) as data:
        cache = {key: data[key].copy() for key in data.files}
    with trace_path.open(encoding="utf-8") as handle:
        cache["records"] = [json.loads(line) for line in handle if line.strip()]
    return cache


def load_partitions(path: Path) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    """Load persisted primary partitions keyed by trajectory identity."""
    output: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            output[(str(row["sample_id"]), int(row["seed"]))] = {
                method: np.asarray(boundaries, dtype=np.int32)
                for method, boundaries in row["methods"].items()
            }
    return output


def trace_view(cache: dict[str, Any], index: int) -> TraceView:
    """Slice concatenated cache arrays for one trajectory."""
    start, end = cache["offsets"][index : index + 2]
    record = cache["records"][index]
    section = slice(int(start), int(end))
    return TraceView(
        sample_id=str(record["sample_id"]),
        seed=int(record["seed"]),
        is_correct=bool(record["is_correct"]),
        train=bool(record["train"]),
        raw=cache["raw"][section].astype(np.float32),
        pca=cache["pca"][section].astype(np.float32),
        h4=cache["h4"][section].astype(np.float32),
        gram=cache["gram"][section].astype(np.float32),
        raw_geometry=cache["raw_geometry"][section].astype(np.float32),
        answer_score=cache["answer_score"][section].astype(np.float32),
        update_count=cache["update_count"][section].astype(np.float32),
        token_count=cache["token_count"][section].astype(np.float32),
    )


def evenly_select_traces(records: list[dict[str, Any]], limit: int) -> list[int]:
    """Retain train and test traces across as many questions as possible."""
    by_question: defaultdict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_question[str(record["sample_id"])].append(index)
    selected: list[int] = []
    cursor = 0
    questions = sorted(by_question)
    while len(selected) < limit:
        added = False
        for question in questions:
            rows = by_question[question]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        cursor += 1
    return sorted(selected)


def merge_short_sentence_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Build a secondary lattice by merging short fragments into their successor."""
    raw_rows: list[np.ndarray] = []
    pca_rows: list[np.ndarray] = []
    h4_rows: list[np.ndarray] = []
    gram_rows: list[np.ndarray] = []
    update_rows: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    offsets = [0]

    for index, source_record in enumerate(cache["records"]):
        trace = trace_view(cache, index)
        groups: list[list[int]] = []
        pending: list[int] = []
        for sentence, token_count in enumerate(trace.token_count):
            if token_count <= 2:
                pending.append(sentence)
                continue
            groups.append([*pending, sentence])
            pending = []
        if pending:
            if groups:
                groups[-1].extend(pending)
            else:
                groups.append(pending)

        token_counts = np.asarray(
            [trace.token_count[group].sum() for group in groups],
            dtype=np.float32,
        )
        raw = weighted_group_means(trace.raw, trace.token_count, groups)
        pca = weighted_group_means(trace.pca, trace.token_count, groups)
        h4 = normalize_rows(
            weighted_group_means(trace.h4, trace.token_count, groups)
        )
        gram = np.stack([trace.gram[group[-1]] for group in groups])
        updates = np.asarray(
            [trace.update_count[group].sum() for group in groups],
            dtype=np.float32,
        )
        answer_sentence = int(source_record["answer_sentence"])
        merged_answer = next(
            group_index
            for group_index, group in enumerate(groups)
            if answer_sentence in group
        )
        record = dict(source_record)
        record["sentences"] = len(groups)
        record["answer_sentence"] = merged_answer
        record["short_fragments_merged"] = int(
            len(trace.token_count) - len(groups)
        )

        raw_rows.append(raw.astype(np.float16))
        pca_rows.append(pca.astype(np.float16))
        h4_rows.append(h4.astype(np.float16))
        gram_rows.append(gram.astype(np.float16))
        update_rows.append(updates.astype(np.int16))
        token_rows.append(token_counts.astype(np.int16))
        records.append(record)
        offsets.append(offsets[-1] + len(groups))

    if "_answer_targets" in cache:
        answer_rows = [
            linear_hsic_alignment(values.astype(np.float32), target)
            for values, target in zip(raw_rows, cache["_answer_targets"])
        ]
    else:
        answer_rows, _target_counts = cross_rollout_answer_scores(raw_rows, records)
    geometry_rows = [
        raw_geometry(values.astype(np.float32)) for values in raw_rows
    ]
    merged = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "raw": np.concatenate(raw_rows).astype(np.float16),
        "pca": np.concatenate(pca_rows).astype(np.float16),
        "h4": np.concatenate(h4_rows).astype(np.float16),
        "gram": np.concatenate(gram_rows).astype(np.float16),
        "raw_geometry": np.concatenate(geometry_rows).astype(np.float32),
        "answer_score": np.concatenate(answer_rows).astype(np.float32),
        "update_count": np.concatenate(update_rows).astype(np.int16),
        "token_count": np.concatenate(token_rows).astype(np.int16),
        "records": records,
    }
    if "_answer_targets" in cache:
        merged["_answer_targets"] = cache["_answer_targets"]
    return merged


def weighted_group_means(
    values: np.ndarray,
    token_counts: np.ndarray,
    groups: list[list[int]],
) -> np.ndarray:
    """Aggregate sentence features with their generated-token counts."""
    output = []
    for group in groups:
        weights = token_counts[group].astype(np.float32)
        output.append(
            np.average(values[group].astype(np.float32), axis=0, weights=weights)
        )
    return np.stack(output).astype(np.float32)


def evaluate_parser_robustness(
    cache: dict[str, Any],
    selected_indices: list[int],
) -> dict[str, Any]:
    """Re-evaluate the core matrix after merging one- and two-token fragments."""
    merged = merge_short_sentence_cache(cache)
    correctness, probe = fit_correctness_curves(merged, selected_indices)
    gram_scores, gram_report = fit_gram_state_scores(merged, selected_indices)
    evaluation = evaluate_partitions(
        merged,
        selected_indices,
        correctness,
        gram_scores,
    )
    original_sentences = sum(
        int(cache["records"][index]["sentences"]) for index in selected_indices
    )
    merged_sentences = sum(
        int(merged["records"][index]["sentences"]) for index in selected_indices
    )
    primary = evaluation["primary"]
    return {
        "rule": (
            "Merge each run of <=2-token parser fragments into the following "
            "full sentence; trailing fragments merge backward."
        ),
        "original_sentences": original_sentences,
        "merged_sentences": merged_sentences,
        "reduction_fraction": 1.0 - merged_sentences / original_sentences,
        "correctness_probe": probe,
        "gram_states": gram_report,
        "primary_utilities": primary["utilities"],
        "boundary_agreement": primary["boundary_agreement"],
        "best_worst_case": primary["best_worst_case"],
        "budget_sweep_best_worst_case": {
            fraction: evaluation[fraction]["best_worst_case"]
            for fraction in ("fraction_0.1", "fraction_0.2", "fraction_0.3")
        },
    }


def prefix_features(values: np.ndarray) -> np.ndarray:
    """Build cumulative mean and variance features at every sentence."""
    count = np.arange(1, len(values) + 1, dtype=np.float32)[:, None]
    first = np.cumsum(values, axis=0) / count
    second = np.cumsum(np.square(values), axis=0) / count
    variance = np.maximum(second - np.square(first), 0.0)
    return np.concatenate([first, variance], axis=1)


def fit_correctness_curves(
    cache: dict[str, Any],
    selected_indices: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Fit a train-question correctness probe and score every sentence prefix."""
    train_x: list[np.ndarray] = []
    train_y: list[np.ndarray] = []
    for index in selected_indices:
        trace = trace_view(cache, index)
        if not trace.train:
            continue
        features = prefix_features(trace.pca)
        keep = np.linspace(0, len(features) - 1, min(len(features), 100), dtype=int)
        train_x.append(features[keep])
        train_y.append(np.full(len(keep), int(trace.is_correct), dtype=np.int8))
    x = np.concatenate(train_x)
    y = np.concatenate(train_y)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.2,
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        ),
    )
    model.fit(x, y)
    curves: dict[int, np.ndarray] = {}
    test_labels: list[int] = []
    test_scores: list[float] = []
    for index in selected_indices:
        trace = trace_view(cache, index)
        scores = model.decision_function(prefix_features(trace.pca))
        curves[index] = np.asarray(scores, dtype=np.float32)
        if not trace.train:
            test_labels.append(int(trace.is_correct))
            test_scores.append(float(scores[-1]))
    auc = roc_auc_score(test_labels, test_scores)
    return curves, {
        "model": "training-only PCA prefix mean+variance, L2 logistic regression",
        "train_prefixes": len(y),
        "train_class_counts": dict(Counter(int(value) for value in y)),
        "held_out_terminal_roc_auc": float(auc),
        "held_out_trajectories": len(test_labels),
    }


def fit_gram_state_scores(
    cache: dict[str, Any],
    selected_indices: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Fit Yu-style five-state Gram clusters and score their transitions."""
    train_rows: list[np.ndarray] = []
    for index in selected_indices:
        trace = trace_view(cache, index)
        if not trace.train:
            continue
        keep = np.linspace(
            0,
            len(trace.gram) - 1,
            min(len(trace.gram), 100),
            dtype=int,
        )
        train_rows.append(trace.gram[keep])
    train = np.concatenate(train_rows).astype(np.float32)
    scaler = StandardScaler()
    transformed = scaler.fit_transform(train)
    model = MiniBatchKMeans(
        n_clusters=5,
        batch_size=2048,
        n_init=20,
        random_state=42,
    ).fit(transformed)

    scores: dict[int, np.ndarray] = {}
    transition_rates: list[float] = []
    normalized_positions: defaultdict[int, list[float]] = defaultdict(list)
    for index in selected_indices:
        trace = trace_view(cache, index)
        values = scaler.transform(trace.gram)
        labels = model.predict(values)
        distance = np.linalg.norm(np.diff(values, axis=0), axis=1)
        changed = labels[1:] != labels[:-1]
        scale = float(distance.max()) + 1.0
        scores[index] = distance + changed.astype(np.float32) * scale
        transition_rates.append(float(np.mean(changed)))
        for sentence, label in enumerate(labels):
            normalized_positions[int(label)].append(
                sentence / max(len(labels) - 1, 1)
            )
    return scores, {
        "clusters": 5,
        "fit": "training questions, at most 100 sentences per trajectory",
        "mean_transition_rate": float(np.mean(transition_rates)),
        "cluster_mean_normalized_position": {
            str(label): float(np.mean(positions))
            for label, positions in sorted(normalized_positions.items())
        },
        "warning": (
            "Strong position ordering would indicate accumulated Gram states "
            "mostly track trace progress rather than objective-independent thought."
        ),
    }


def answer_proxy_diagnostics(
    cache: dict[str, Any],
    selected_indices: list[int],
) -> dict[str, float]:
    """Quantify terminal-position bias in the local answer-information proxy."""
    correlations: list[float] = []
    top_positions: list[float] = []
    final_quarter_shares: list[float] = []
    peak_rates: list[float] = []
    for index in selected_indices:
        record = cache["records"][index]
        if bool(record["train"]):
            continue
        start, end = cache["offsets"][index : index + 2]
        answer_score = cache["answer_score"][int(start) : int(end)].astype(
            np.float32
        )
        position = np.linspace(0.0, 1.0, len(answer_score))
        if np.std(answer_score) > 0:
            correlations.append(
                float(np.corrcoef(position, answer_score)[0, 1])
            )
        boundary_scores = np.abs(np.diff(answer_score))
        budget = max(int(round(len(boundary_scores) * PRIMARY_FRACTION)), 1)
        top = top_boundaries(boundary_scores, budget)
        normalized = top / max(len(boundary_scores) - 1, 1)
        top_positions.extend(normalized.tolist())
        final_quarter_shares.append(float(np.mean(normalized >= 0.75)))
        q1, q3 = np.percentile(answer_score, [25, 75])
        threshold = q3 + 1.5 * (q3 - q1)
        peak_rates.append(float(np.mean(answer_score > threshold)))
    return {
        "mean_score_position_correlation": float(np.mean(correlations)),
        "mean_selected_boundary_position": float(np.mean(top_positions)),
        "mean_selected_share_in_final_quarter": float(
            np.mean(final_quarter_shares)
        ),
        "mean_iqr_peak_rate": float(np.mean(peak_rates)),
    }


def objective_costs(
    trace: TraceView,
    correctness_curve: np.ndarray,
) -> dict[str, np.ndarray]:
    """Construct the four additive sentence-segment objective costs."""
    return {
        "answer": squared_error_costs(trace.answer_score),
        "object": object_update_costs(trace.update_count),
        "correctness": squared_error_costs(correctness_curve),
        "compression": squared_error_costs(trace.pca),
    }


def candidate_partitions(
    trace: TraceView,
    costs: dict[str, np.ndarray],
    *,
    fraction: float,
    rng: np.random.Generator,
    gram_transition_scores: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """Build matched-budget heuristic and exact-oracle partitions."""
    count = len(trace.pca)
    boundary_count = min(max(int(round((count - 1) * fraction)), 1), count - 1)
    segment_count = boundary_count + 1
    answer_change = np.abs(np.diff(trace.answer_score))
    h4_change = np.linalg.norm(np.diff(trace.h4, axis=0), axis=1)
    curvature = trace.raw_geometry[1:, 3]
    partitions = {
        "fixed_windows": fixed_boundaries(count, boundary_count),
        "raw_curvature": top_boundaries(curvature, boundary_count),
        "answer_peaks": top_boundaries(answer_change, boundary_count),
        "gram_transitions": top_boundaries(
            gram_transition_scores, boundary_count
        ),
        "h4_transitions": top_boundaries(h4_change, boundary_count),
    }
    for objective, cost in costs.items():
        partitions[ORACLE_NAMES[objective]] = optimal_partition(cost, segment_count)
    random_samples = [
        random_boundaries(count, boundary_count, rng) for _ in range(12)
    ]
    partitions["random"] = random_samples[0]
    return partitions, random_samples


def evaluate_partitions(
    cache: dict[str, Any],
    selected_indices: list[int],
    correctness_curves: dict[int, np.ndarray],
    gram_scores: dict[int, np.ndarray],
) -> dict[str, Any]:
    """Evaluate all candidate partitions on held-out questions and budget sweeps."""
    rng = np.random.default_rng(42)
    test_indices = [
        index
        for index in selected_indices
        if not bool(cache["records"][index]["train"])
    ]
    all_primary: dict[int, dict[str, np.ndarray]] = {}
    result: dict[str, Any] = {}
    for fraction in (0.1, 0.2, 0.3):
        utilities: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        utility_groups: defaultdict[
            str, defaultdict[str, list[str]]
        ] = defaultdict(lambda: defaultdict(list))
        agreements: defaultdict[tuple[str, str], list[tuple[float, float, float]]] = (
            defaultdict(list)
        )
        for index in selected_indices:
            trace = trace_view(cache, index)
            costs = objective_costs(trace, correctness_curves[index])
            partitions, random_samples = candidate_partitions(
                trace,
                costs,
                fraction=fraction,
                rng=rng,
                gram_transition_scores=gram_scores[index],
            )
            if fraction == PRIMARY_FRACTION:
                all_primary[index] = partitions
            if index not in test_indices:
                continue
            for objective, cost in costs.items():
                oracle_cost = partition_cost(
                    cost, partitions[ORACLE_NAMES[objective]]
                )
                random_cost = float(
                    np.mean(
                        [
                            partition_cost(cost, boundaries)
                            for boundaries in random_samples
                        ]
                    )
                )
                denominator = random_cost - oracle_cost
                if denominator <= 1e-10:
                    continue
                for method, boundaries in partitions.items():
                    method_cost = (
                        random_cost
                        if method == "random"
                        else partition_cost(cost, boundaries)
                    )
                    utility = 1.0 - (method_cost - oracle_cost) / denominator
                    utilities[method][objective].append(float(utility))
                    utility_groups[method][objective].append(trace.sample_id)

            oracle_pairs = [
                (left, right)
                for i, left in enumerate(OBJECTIVES)
                for right in OBJECTIVES[i + 1 :]
            ]
            for left, right in oracle_pairs:
                left_boundaries = partitions[ORACLE_NAMES[left]]
                right_boundaries = partitions[ORACLE_NAMES[right]]
                agreements[(left, right)].append(
                    (
                        boundary_jaccard(left_boundaries, right_boundaries),
                        boundary_f1(
                            left_boundaries, right_boundaries, tolerance=1
                        ),
                        partition_variation_of_information(
                            left_boundaries,
                            right_boundaries,
                            len(trace.pca),
                        ),
                    )
                )

        mean_utilities = {
            method: {
                objective: float(np.mean(values))
                for objective, values in objective_values.items()
            }
            for method, objective_values in utilities.items()
        }
        utility_intervals = {
            method: {
                objective: grouped_bootstrap_summary(
                    values,
                    utility_groups[method][objective],
                )
                for objective, values in objective_values.items()
            }
            for method, objective_values in utilities.items()
        }
        agreement_rows = [
            {
                "left": left,
                "right": right,
                "boundary_fraction": fraction,
                "jaccard": float(np.mean([value[0] for value in values])),
                "f1_tolerance_1": float(
                    np.mean([value[1] for value in values])
                ),
                "variation_of_information": float(
                    np.mean([value[2] for value in values])
                ),
            }
            for (left, right), values in agreements.items()
        ]
        key = f"fraction_{fraction:.1f}"
        result[key] = {
            "utilities": mean_utilities,
            "utility_question_bootstrap": utility_intervals,
            "boundary_agreement": agreement_rows,
            "pareto_front": pareto_front(mean_utilities),
            "heuristic_rank_correlations": objective_rank_correlations(
                mean_utilities
            ),
            "best_worst_case": best_worst_case(mean_utilities),
            "test_trajectories": len(test_indices),
        }
    result["primary"] = result["fraction_0.2"]
    result["primary_partitions"] = all_primary
    return result


def grouped_bootstrap_summary(
    values: list[float],
    groups: list[str],
    *,
    resamples: int = 4000,
) -> dict[str, Any]:
    """Summarize a metric with a question-level nonparametric bootstrap."""
    by_group: defaultdict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    group_means = np.asarray(
        [np.mean(by_group[group]) for group in sorted(by_group)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(421)
    draws = rng.choice(
        group_means,
        size=(resamples, len(group_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(group_means.mean()),
        "95ci": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
        "questions": len(group_means),
    }


def objective_rank_correlations(
    utilities: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    """Compare heuristic rankings across objectives without oracle tautologies."""
    methods = [
        method
        for method in (
            "fixed_windows",
            "raw_curvature",
            "answer_peaks",
            "gram_transitions",
            "h4_transitions",
            "random",
        )
        if method in utilities
    ]
    rows: list[dict[str, float | str]] = []
    for index, left in enumerate(OBJECTIVES):
        for right in OBJECTIVES[index + 1 :]:
            correlation = spearmanr(
                [utilities[method][left] for method in methods],
                [utilities[method][right] for method in methods],
            ).statistic
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(correlation),
                }
            )
    return rows


def best_worst_case(
    utilities: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Identify the method with the strongest minimum cross-objective utility."""
    minima = {
        method: min(scores.values()) for method, scores in utilities.items()
    }
    method = max(minima, key=minima.get)
    return {
        "method": method,
        "minimum_utility": float(minima[method]),
        "near_optimal_on_all_objectives": minima[method] >= 0.9,
    }


def boundary_features(values: np.ndarray) -> np.ndarray:
    """Describe each transition with signed and absolute coordinate changes."""
    delta = np.diff(values, axis=0)
    position = np.linspace(0.0, 1.0, len(delta), dtype=np.float32)[:, None]
    magnitude = np.linalg.norm(delta, axis=1, keepdims=True)
    return np.concatenate([delta, np.abs(delta), magnitude, position], axis=1)


def evaluate_supervised_boundaries(
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Train objective-specific change-point adversaries and test transfer."""
    spaces = {
        "raw": "raw",
        "pca_whitened": "pca",
        "gram_spectrum": "gram",
        "h4_operation": "h4",
    }
    rows: list[dict[str, Any]] = []
    diagonal: dict[str, dict[str, float]] = defaultdict(dict)
    transfer_ratios: dict[str, dict[str, float]] = defaultdict(dict)
    for space, attribute in spaces.items():
        train_x: list[np.ndarray] = []
        test_x: list[np.ndarray] = []
        train_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        test_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        test_slices: list[tuple[int, int, int]] = []
        cursor = 0
        for index in selected_indices:
            trace = trace_view(cache, index)
            features = boundary_features(getattr(trace, attribute))
            labels = {
                objective: boundary_labels(
                    len(features), primary_partitions[index][ORACLE_NAMES[objective]]
                )
                for objective in OBJECTIVES
            }
            if trace.train:
                train_x.append(features)
                for objective, values in labels.items():
                    train_labels[objective].append(values)
            else:
                test_x.append(features)
                for objective, values in labels.items():
                    test_labels[objective].append(values)
                positives = len(primary_partitions[index][ORACLE_NAMES["answer"]])
                test_slices.append((cursor, cursor + len(features), positives))
                cursor += len(features)
        x_train = np.concatenate(train_x).astype(np.float32)
        x_test = np.concatenate(test_x).astype(np.float32)
        y_train = {
            objective: np.concatenate(values)
            for objective, values in train_labels.items()
        }
        y_test = {
            objective: np.concatenate(values)
            for objective, values in test_labels.items()
        }
        for trained_on in OBJECTIVES:
            model = fit_boundary_model(
                x_train,
                y_train[trained_on],
                nonlinear=space != "raw",
            )
            probabilities = model.predict_proba(x_test)[:, 1]
            for evaluated_on in OBJECTIVES:
                expected = y_test[evaluated_on]
                predicted = matched_probability_labels(probabilities, test_slices)
                auc = float(roc_auc_score(expected, probabilities))
                average_precision = float(
                    average_precision_score(expected, probabilities)
                )
                f1 = float(f1_score(expected, predicted))
                rows.append(
                    {
                        "space": space,
                        "trained_on": trained_on,
                        "evaluated_on": evaluated_on,
                        "roc_auc": auc,
                        "average_precision": average_precision,
                        "matched_budget_f1": f1,
                    }
                )
                if trained_on == evaluated_on:
                    diagonal[space][trained_on] = auc
            own_auc = diagonal[space][trained_on]
            cross = [
                row["roc_auc"]
                for row in rows
                if row["space"] == space
                and row["trained_on"] == trained_on
                and row["evaluated_on"] != trained_on
            ]
            transfer_ratios[space][trained_on] = float(np.mean(cross) / own_auc)
    return {
        "model": (
            "raw: balanced logistic SGD on full transition features; projected "
            "spaces: nonlinear histogram gradient boosting"
        ),
        "primary_boundary_fraction": PRIMARY_FRACTION,
        "diagonal_roc_auc": dict(diagonal),
        "mean_cross_to_in_domain_auc_ratio": dict(transfer_ratios),
        "rows": rows,
    }


def fit_boundary_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nonlinear: bool,
) -> Any:
    """Fit a nonlinear projected-space detector or scalable raw-space probe."""
    if nonlinear:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=100,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
    else:
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                early_stopping=True,
                n_iter_no_change=8,
                random_state=42,
            ),
        )
    model.fit(x, y)
    return model


def boundary_labels(count: int, boundaries: np.ndarray) -> np.ndarray:
    """Convert selected boundary indices into a binary target vector."""
    labels = np.zeros(count, dtype=np.int8)
    labels[np.asarray(boundaries, dtype=int)] = 1
    return labels


def matched_probability_labels(
    probabilities: np.ndarray,
    slices: list[tuple[int, int, int]],
) -> np.ndarray:
    """Threshold each trace at its exact oracle boundary budget."""
    labels = np.zeros(len(probabilities), dtype=np.int8)
    for start, end, positives in slices:
        local = top_boundaries(probabilities[start:end], positives)
        labels[start + local] = 1
    return labels


def evaluate_projection_coherence(
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
) -> dict[str, Any]:
    """Measure whether each oracle also compresses each projection space."""
    spaces = {
        "raw": "raw",
        "pca_whitened": "pca",
        "gram_spectrum": "gram",
        "h4_operation": "h4",
    }
    rng = np.random.default_rng(904)
    scores: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index in selected_indices:
        trace = trace_view(cache, index)
        if trace.train:
            continue
        for space, attribute in spaces.items():
            costs = squared_error_costs(getattr(trace, attribute))
            boundary_count = len(
                primary_partitions[index][ORACLE_NAMES["compression"]]
            )
            oracle = optimal_partition(costs, boundary_count + 1)
            oracle_cost = partition_cost(costs, oracle)
            random_cost = float(
                np.mean(
                    [
                        partition_cost(
                            costs,
                            random_boundaries(
                                len(trace.pca), boundary_count, rng
                            ),
                        )
                        for _ in range(12)
                    ]
                )
            )
            denominator = random_cost - oracle_cost
            if denominator <= 1e-10:
                continue
            for objective in OBJECTIVES:
                boundaries = primary_partitions[index][ORACLE_NAMES[objective]]
                utility = 1.0 - (
                    partition_cost(costs, boundaries) - oracle_cost
                ) / denominator
                scores[space][objective].append(float(utility))
    return {
        space: {
            objective: float(np.mean(values))
            for objective, values in objective_scores.items()
        }
        for space, objective_scores in scores.items()
    }


def write_matrix_csv(
    path: Path,
    matrix: dict[str, dict[str, float]],
) -> None:
    """Write a method-by-objective matrix with stable column ordering."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", *OBJECTIVES])
        writer.writeheader()
        for method in sorted(matrix):
            writer.writerow({"method": method, **matrix[method]})


def write_records_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write homogeneous report records to CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_boundary_examples(
    run_path: Path,
    path: Path,
    cache: dict[str, Any],
    selected_indices: list[int],
    primary_partitions: dict[int, dict[str, np.ndarray]],
    *,
    trace_limit: int = 8,
    boundaries_per_method: int = 8,
) -> None:
    """Write sentence text around representative held-out boundaries."""
    chosen: list[int] = []
    seen_questions: set[str] = set()
    for index in selected_indices:
        record = cache["records"][index]
        question = str(record["sample_id"])
        if record["train"] or question in seen_questions:
            continue
        chosen.append(index)
        seen_questions.add(question)
        if len(chosen) >= trace_limit:
            break

    rows = balanced_generation_rows(run_path, per_sample=10)
    row_lookup = {
        (str(row["sample_id"]), int(row["seed"])): row for row in rows
    }
    chosen_rows = [
        row_lookup[
            (
                str(cache["records"][index]["sample_id"]),
                int(cache["records"][index]["seed"]),
            )
        ]
        for index in chosen
    ]
    token_spans = build_token_spans(run_path, chosen_rows)
    examples: list[dict[str, Any]] = []
    for index, row, spans in zip(chosen, chosen_rows, token_spans):
        segments = build_segments(
            row,
            "sentence",
            {"mode": "sentence", "group_size": 1},
            token_spans=spans,
        )
        for method, boundaries in primary_partitions[index].items():
            keep = np.linspace(
                0,
                len(boundaries) - 1,
                min(len(boundaries), boundaries_per_method),
                dtype=int,
            )
            selected = boundaries[keep] if len(boundaries) else boundaries
            examples.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "seed": int(row["seed"]),
                    "method": method,
                    "sentence_count": len(segments),
                    "boundary_count": len(boundaries),
                    "examples": [
                        {
                            "boundary_after_sentence": int(boundary),
                            "position": float(
                                boundary / max(len(segments) - 2, 1)
                            ),
                            "left": segments[int(boundary)].text[:300],
                            "right": segments[int(boundary) + 1].text[:300],
                        }
                        for boundary in selected
                        if int(boundary) + 1 < len(segments)
                    ],
                }
            )
    write_jsonl(path, examples)


def write_partitions(
    path: Path,
    cache: dict[str, Any],
    selected_indices: list[int],
    partitions: dict[int, dict[str, np.ndarray]],
) -> None:
    """Persist every primary matched-budget boundary set for audit and reuse."""
    rows = []
    for index in selected_indices:
        record = cache["records"][index]
        rows.append(
            {
                "sample_id": str(record["sample_id"]),
                "seed": int(record["seed"]),
                "train": bool(record["train"]),
                "sentence_count": int(record["sentences"]),
                "boundary_fraction": PRIMARY_FRACTION,
                "methods": {
                    method: values.astype(int).tolist()
                    for method, values in partitions[index].items()
                },
            }
        )
    write_jsonl(path, rows)


def write_plots(
    out_dir: Path,
    utilities: dict[str, dict[str, float]],
    front: list[str],
) -> None:
    """Render compact utility, regret, and two-objective Pareto figures."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    methods = sorted(utilities)
    values = np.asarray(
        [
            [utilities[method][objective] for objective in OBJECTIVES]
            for method in methods
        ]
    )
    for filename, matrix, title, cmap, norm in (
        (
            "objective_matrix.png",
            values,
            "Normalized utility (random=0, oracle=1)",
            "RdYlGn",
            TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0),
        ),
        (
            "regret_matrix.png",
            1.0 - values,
            "Normalized regret (oracle=0)",
            "magma_r",
            Normalize(vmin=0.0, vmax=2.0, clip=True),
        ),
    ):
        fig, axis = plt.subplots(figsize=(7.2, max(3.8, 0.36 * len(methods))))
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(OBJECTIVES)), OBJECTIVES, rotation=20)
        axis.set_yticks(range(len(methods)), methods)
        axis.set_title(title)
        for row in range(len(methods)):
            for column in range(len(OBJECTIVES)):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.colorbar(image, ax=axis, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    for method in methods:
        axis.scatter(
            utilities[method]["object"],
            utilities[method]["compression"],
            marker="o" if method in front else "x",
        )
        axis.annotate(
            method,
            (
                utilities[method]["object"],
                utilities[method]["compression"],
            ),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    axis.set_xlabel("Object utility")
    axis.set_ylabel("Compression utility")
    axis.set_title("Matched-budget Pareto slice")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_object_compression.png", dpi=180)
    plt.close(fig)
