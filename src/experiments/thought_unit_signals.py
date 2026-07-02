"""Parser robustness, answer alignment, correctness, and Gram-state signals."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.sentence_lattice import top_boundaries
from src.experiments.thought_unit_cache import trace_view
from src.experiments.thought_unit_features import (
    compare_answer_curves,
    cross_rollout_answer_scores,
    linear_hsic_alignment,
    normalize_rows,
    raw_geometry,
)
from src.experiments.thought_unit_types import PRIMARY_FRACTION
from src.runtime.artifact_store import load_hidden_states_npz


def apply_gold_answer_scores(
    cache: dict[str, Any],
    gold_answer_run: Path,
    *,
    selected_indices: list[int],
    output_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace proxy answer curves with captured gold-solution alignment curves.

    Args:
        cache: Cached arrays or records used by the computation.
        gold_answer_run: Run directory containing gold-answer activation captures.
        selected_indices: Indices of traces selected for evaluation.
        output_path: Destination path for the generated artifact.

    Returns:
        The computed aligned values described above.
    """
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
        targets[sample_id] = states[:, layers.index(-1)].astype(np.float32).mean(axis=0)

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


def merge_short_sentence_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Build a secondary lattice by merging short fragments into their successor.

    Args:
        cache: Cached arrays or records used by the computation.

    Returns:
        The resulting keyed records or metrics.
    """
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
        h4 = normalize_rows(weighted_group_means(trace.h4, trace.token_count, groups))
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
        record["short_fragments_merged"] = int(len(trace.token_count) - len(groups))

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
    geometry_rows = [raw_geometry(values.astype(np.float32)) for values in raw_rows]
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
    """Aggregate sentence features with their generated-token counts.

    Args:
        values: Values to summarize or transform.
        token_counts: Per-sentence token counts used as weights.
        groups: Group labels used to prevent cross-question leakage.

    Returns:
        The resulting numeric array or tensor.
    """
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
    """Re-evaluate the core matrix after merging one- and two-token fragments.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.

    Returns:
        The resulting keyed records or metrics.
    """
    from src.experiments.thought_unit_partitions import evaluate_partitions

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
    """Build cumulative mean and variance features at every sentence.

    Args:
        values: Values to summarize or transform.

    Returns:
        The resulting numeric array or tensor.
    """
    count = np.arange(1, len(values) + 1, dtype=np.float32)[:, None]
    first = np.cumsum(values, axis=0) / count
    second = np.cumsum(np.square(values), axis=0) / count
    variance = np.maximum(second - np.square(first), 0.0)
    return np.concatenate([first, variance], axis=1)


def fit_correctness_curves(
    cache: dict[str, Any],
    selected_indices: list[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Fit a train-question correctness probe and score every sentence prefix.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.

    Returns:
        The computed aligned values described above.
    """
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
    """Fit Yu-style five-state Gram clusters and score their transitions.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.

    Returns:
        The computed aligned values described above.
    """
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
            normalized_positions[int(label)].append(sentence / max(len(labels) - 1, 1))
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
    """Quantify terminal-position bias in the local answer-information proxy.

    Args:
        cache: Cached arrays or records used by the computation.
        selected_indices: Indices of traces selected for evaluation.

    Returns:
        The resulting keyed records or metrics.
    """
    correlations: list[float] = []
    top_positions: list[float] = []
    final_quarter_shares: list[float] = []
    peak_rates: list[float] = []
    for index in selected_indices:
        record = cache["records"][index]
        if bool(record["train"]):
            continue
        start, end = cache["offsets"][index : index + 2]
        answer_score = cache["answer_score"][int(start) : int(end)].astype(np.float32)
        position = np.linspace(0.0, 1.0, len(answer_score))
        if np.std(answer_score) > 0:
            correlations.append(float(np.corrcoef(position, answer_score)[0, 1]))
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
        "mean_selected_share_in_final_quarter": float(np.mean(final_quarter_shares)),
        "mean_iqr_peak_rate": float(np.mean(peak_rates)),
    }
