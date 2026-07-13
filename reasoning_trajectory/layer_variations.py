"""Plot adjacent-layer metrics for captured hidden-state traces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

from reasoning_trajectory.artifacts import load_hidden_states_npz, read_generation_rows
from reasoning_trajectory.config import load_run_config


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Euclidean distance between two feature vectors."""
    return float(np.linalg.norm(a - b))


def normalized_residual_update_magnitude(a: np.ndarray, b: np.ndarray) -> float:
    """Compute residual update magnitude after normalizing each layer vector."""
    return float(np.linalg.norm(_unit_vector(b) - _unit_vector(a)))


def plot_npz_metric(
    npz_path: str | Path,
    tok_idxs: Sequence[int],
    metric_fn: MetricFn = cosine_similarity,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Load one activation NPZ and plot an adjacent-layer metric."""
    states, layers = load_hidden_states_npz(npz_path)
    states = states.astype(np.float32, copy=False)
    if save_path is None:
        save_path = Path(npz_path).with_name(
            f"{Path(npz_path).stem}_{metric_fn.__name__}.png"
        )
    return plot_metric(
        states,
        tok_idxs,
        metric_fn,
        Path(save_path),
        layers=layers,
        title=title,
    )


def write_layer_plots(run_path: Path, cfg: dict) -> None:
    """Write simple 2D and 3D adjacent-layer plots for completed traces."""
    rows = [
        row for row in read_generation_rows(run_path) if row.get("hidden_states_file")
    ]
    if not rows:
        return

    plot_cfg = cfg.get("layer_variations", {})
    if plot_cfg is True:
        plot_cfg = {}
    out_dir = run_path / "analysis" / "layer_variations"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token_2d = int(plot_cfg.get("token_2d", 0))
    max_3d_tokens = int(plot_cfg.get("max_3d_tokens", 160))
    metric_fns = {
        "cosine": cosine_similarity,
        "distance": euclidean_distance,
        "normalized_residual_update_magnitude": normalized_residual_update_magnitude,
    }
    transition_metrics = {"directional_persistence": "Directional persistence"}
    model_name = _model_name(run_path)
    manifest = []
    web_payloads: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        source_path = run_path / row["hidden_states_file"]
        states, layers = load_hidden_states_npz(source_path)
        states = states.astype(np.float32, copy=False)
        if states.shape[0] == 0 or states.shape[1] < 2:
            continue

        stem = _plot_stem(row, row_index, model_name)
        trace_dir = out_dir / stem
        trace_dir.mkdir(parents=True, exist_ok=True)
        token_idx = min(max(token_2d, 1), states.shape[0] - 1)
        sampled_tokens = [
            idx for idx in _sample_token_indices(states.shape[0], max_3d_tokens)
            if idx >= 1
        ]
        for metric_name, metric_fn in metric_fns.items():
            path_2d = trace_dir / f"{metric_name}_token{token_idx}_2d.png"
            path_3d = trace_dir / f"{metric_name}_tokens_3d.png"
            title_prefix = f"{model_name} on {row.get('sample_id', stem)}"
            plot_metric(
                states,
                [token_idx],
                metric_fn,
                path_2d,
                layers=layers,
                title=f"{title_prefix}: {metric_name} at token {token_idx}",
            )
            plot_metric(
                states,
                sampled_tokens,
                metric_fn,
                path_3d,
                layers=layers,
                title=f"{title_prefix}: {metric_name} across generated tokens",
            )
            manifest.extend(
                [
                    {
                        "sample_id": row.get("sample_id"),
                        "seed": row.get("seed"),
                        "metric": metric_name,
                        "kind": "2d",
                        "token": token_idx,
                        "path": path_2d.relative_to(run_path).as_posix(),
                    },
                    {
                        "sample_id": row.get("sample_id"),
                        "seed": row.get("seed"),
                        "metric": metric_name,
                        "kind": "3d",
                        "tokens": len(sampled_tokens),
                        "path": path_3d.relative_to(run_path).as_posix(),
                    },
                ]
            )
            payload = web_payloads.setdefault(
                metric_name,
                _empty_web_payload(metric_name, metric_fn.__name__, max_3d_tokens),
            )
            _append_web_points(
                payload,
                row,
                row_index,
                states,
                layers,
                sampled_tokens,
                metric_fn,
            )
        for metric_name, metric_label in transition_metrics.items():
            transition_tokens = [idx for idx in sampled_tokens if idx < states.shape[0] - 1]
            if not transition_tokens:
                continue
            token_idx = min(token_idx, states.shape[0] - 2)
            path_2d = trace_dir / f"{metric_name}_token{token_idx}_2d.png"
            path_3d = trace_dir / f"{metric_name}_tokens_3d.png"
            title_prefix = f"{model_name} on {row.get('sample_id', stem)}"
            _plot_matrix(
                _directional_persistence_matrix(states, [token_idx]),
                [token_idx],
                path_2d,
                layers=layers,
                metric_label=metric_label,
                title=f"{title_prefix}: {metric_name} at token {token_idx}",
            )
            _plot_matrix(
                _directional_persistence_matrix(states, transition_tokens),
                transition_tokens,
                path_3d,
                layers=layers,
                metric_label=metric_label,
                title=f"{title_prefix}: {metric_name} across generated tokens",
            )
            manifest.extend(
                [
                    {
                        "sample_id": row.get("sample_id"),
                        "seed": row.get("seed"),
                        "metric": metric_name,
                        "kind": "2d",
                        "token": token_idx,
                        "path": path_2d.relative_to(run_path).as_posix(),
                    },
                    {
                        "sample_id": row.get("sample_id"),
                        "seed": row.get("seed"),
                        "metric": metric_name,
                        "kind": "3d",
                        "tokens": len(transition_tokens),
                        "path": path_3d.relative_to(run_path).as_posix(),
                    },
                ]
            )
            payload = web_payloads.setdefault(
                metric_name,
                _empty_web_payload(metric_name, metric_label, max_3d_tokens),
            )
            _append_web_matrix_points(
                payload,
                row,
                row_index,
                states,
                layers,
                transition_tokens,
                _directional_persistence_matrix(states, transition_tokens),
            )

    (out_dir / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    web_manifest = []
    for metric_name, payload in web_payloads.items():
        path = out_dir / f"{metric_name}_layerwise.json"
        points = [
            point
            for trace in payload["traces"]
            for point in trace["points"]
        ]
        payload["points"] = points
        payload["traces"] = [
            {key: value for key, value in trace.items() if key != "points"}
            for trace in payload["traces"]
        ]
        payload["trace_count"] = len(payload["traces"])
        payload["point_count"] = len(points)
        path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        web_manifest.append(
            {
                "metric": metric_name,
                "metric_label": payload["metric_label"],
                "traces": payload["trace_count"],
                "points": payload["point_count"],
                "max_3d_tokens": max_3d_tokens,
                "path": path.relative_to(run_path).as_posix(),
            }
        )
    (out_dir / "web_index.json").write_text(
        json.dumps(web_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_correctness_group_plots(run_path: Path, cfg: dict) -> None:
    """Write layerwise metric plots averaged over correct and incorrect groups."""
    rows = [
        row for row in read_generation_rows(run_path)
        if row.get("hidden_states_file") and row.get("is_correct") in {True, False}
    ]
    groups = {
        "correct": [row for row in rows if row["is_correct"] is True],
        "incorrect": [row for row in rows if row["is_correct"] is False],
    }
    plot_cfg = cfg.get("layer_variations", {})
    if plot_cfg is True:
        plot_cfg = {}
    max_3d_tokens = int(plot_cfg.get("max_3d_tokens", 160))
    out_dir = run_path / "analysis" / "layer_variations_by_correctness"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_fns = {
        "cosine": (cosine_similarity, "Cosine similarity"),
        "distance": (euclidean_distance, "Euclidean distance"),
        "normalized_residual_update_magnitude": (
            normalized_residual_update_magnitude,
            "Normalized residual update magnitude",
        ),
    }
    group_summary = {}
    for group_name, group_rows in groups.items():
        group_dir = out_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        valid_rows = _valid_group_rows(run_path, group_rows)
        summary = {
            "group": group_name,
            "rollout_count": len(valid_rows),
            "metrics": {},
        }
        if not valid_rows:
            (group_dir / "metrics.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            group_summary[group_name] = summary
            continue
        layers = load_hidden_states_npz(
            run_path / valid_rows[0]["hidden_states_file"]
        )[1]
        for metric_name, (metric_fn, metric_label) in metric_fns.items():
            means, counts = _aggregate_group_metric(run_path, valid_rows, metric_fn)
            _write_group_metric_plots(
                group_dir,
                metric_name,
                metric_label,
                means,
                counts,
                layers,
                max_3d_tokens,
                start_token=1,
            )
            trimmed_means = means[1:]
            trimmed_counts = counts[1:]
            summary["metrics"][metric_name] = {
                "metric_label": metric_label,
                "token_indices": list(range(1, int(means.shape[0]))),
                "rollout_counts": trimmed_counts.tolist(),
                "mean_by_token_and_layer_pair": np.round(trimmed_means, 6).tolist(),
            }

        persistence, persistence_counts = _aggregate_group_metric(
            run_path,
            valid_rows,
            None,
            transition=True,
        )
        persistence_mean = np.nanmean(persistence, axis=1)
        valid = np.isfinite(persistence_mean)
        token_indices = np.arange(persistence_mean.shape[0])[valid & (np.arange(persistence_mean.shape[0]) >= 1)]
        persistence_values = persistence_mean[token_indices]
        correlation = _pearson_correlation(token_indices, persistence_values)
        inlier_mask = _iqr_inlier_mask(persistence_values)
        filtered_correlation = _pearson_correlation(
            token_indices[inlier_mask], persistence_values[inlier_mask]
        )
        _write_group_metric_plots(
            group_dir,
            "directional_persistence",
            "Directional persistence",
            persistence,
            persistence_counts,
            layers,
            max_3d_tokens,
            start_token=1,
        )
        trimmed_persistence = persistence[1:]
        trimmed_persistence_counts = persistence_counts[1:]
        summary["metrics"]["directional_persistence"] = {
            "metric_label": "Directional persistence",
            "token_indices": list(range(1, int(persistence.shape[0]))),
            "rollout_counts": trimmed_persistence_counts.tolist(),
            "mean_by_token_and_layer_pair": np.round(trimmed_persistence, 6).tolist(),
        }
        summary["directional_persistence_token_correlation"] = {
            "method": "pearson",
            "token_index": "absolute generated token index",
            "r": correlation,
            "n_tokens": int(token_indices.size),
            "outlier_filtered": {
                "method": "remove values outside 1.5 IQR of the token-level mean persistence",
                "r": filtered_correlation,
                "n_tokens": int(inlier_mask.sum()),
            },
            "mean_persistence_by_token": [
                {"token_idx": int(token), "value": round(float(value), 6)}
                for token, value in zip(token_indices, persistence_values)
            ],
        }
        (group_dir / "metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        group_summary[group_name] = summary
    (out_dir / "index.json").write_text(
        json.dumps(group_summary, indent=2) + "\n", encoding="utf-8"
    )


def _valid_group_rows(
    run_path: Path,
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_rows = []
    for row in rows:
        with np.load(run_path / row["hidden_states_file"]) as data:
            shape = data["hidden_states_q"].shape if "hidden_states_q" in data else data["hidden_states"].shape
        if shape[0] and shape[1] >= 2:
            valid_rows.append(row)
    return valid_rows


def _aggregate_group_metric(
    run_path: Path,
    rows: Sequence[dict[str, Any]],
    metric_fn: MetricFn | None,
    *,
    transition: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    counts = np.zeros(0, dtype=np.int32)
    for row in rows:
        states, _ = load_hidden_states_npz(run_path / row["hidden_states_file"])
        states = states.astype(np.float32, copy=False)
        token_count = states.shape[0] - int(transition)
        if transition:
            matrix = _directional_persistence_matrix(states, range(token_count))
        else:
            matrix = np.asarray(
                [_adjacent_metrics(states[idx], metric_fn) for idx in range(token_count)],
                dtype=np.float32,
            )
        if sums is None:
            sums = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
            counts = np.zeros(matrix.shape[0], dtype=np.int32)
        if matrix.shape[0] > sums.shape[0]:
            extra = matrix.shape[0] - sums.shape[0]
            sums = np.pad(sums, ((0, extra), (0, 0)))
            counts = np.pad(counts, (0, extra))
        sums[:matrix.shape[0]] += matrix
        counts[:matrix.shape[0]] += 1
    if sums is None:
        return np.empty((0, 0), dtype=np.float32), counts
    means = sums / np.maximum(counts[:, None], 1)
    means[counts == 0] = np.nan
    return means.astype(np.float32), counts


def _write_group_metric_plots(
    group_dir: Path,
    metric_name: str,
    metric_label: str,
    means: np.ndarray,
    counts: np.ndarray,
    layers: Sequence[int],
    max_3d_tokens: int,
    start_token: int = 0,
) -> None:
    valid_tokens = np.flatnonzero(np.isfinite(means).all(axis=1))
    valid_tokens = valid_tokens[valid_tokens >= start_token]
    if valid_tokens.size == 0:
        return
    token_indices = _sample_token_indices(int(valid_tokens[-1]) + 1, max_3d_tokens)
    token_indices = [
        idx for idx in token_indices
        if idx >= start_token and np.isfinite(means[idx]).all()
    ]
    if not token_indices:
        token_indices = [int(valid_tokens[0])]
    _plot_matrix(
        means[[token_indices[0]]],
        [token_indices[0]],
        group_dir / f"{metric_name}_token{token_indices[0]}_2d.png",
        layers=layers,
        metric_label=metric_label,
        title=f"{metric_label}: {group_dir.name} at token {token_indices[0]}",
    )
    _plot_matrix(
        means[token_indices],
        token_indices,
        group_dir / f"{metric_name}_tokens_3d.png",
        layers=layers,
        metric_label=metric_label,
        title=f"{metric_label}: {group_dir.name} rollouts averaged",
    )


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _iqr_inlier_mask(values: np.ndarray) -> np.ndarray:
    """Return token-level inliers using Tukey's 1.5-IQR rule."""
    if values.size < 4:
        return np.ones(values.shape, dtype=bool)
    lower, upper = np.percentile(values, [25, 75])
    spread = upper - lower
    return (values >= lower - 1.5 * spread) & (values <= upper + 1.5 * spread)


def plot_metric(
    acts: np.ndarray,
    tok_idxs: Sequence[int],
    metric_fn: MetricFn,
    save_path: Path,
    *,
    layers: Sequence[int] | None = None,
    title: str | None = None,
) -> Path:
    """Plot a 2D or 3D adjacent-layer metric.

    Args:
        acts: Either ``[tokens, layers, hidden]`` or ``[layers, hidden]``.
        tok_idxs: Generated-token indices to plot. If one index is provided,
            writes a 2D line plot. If multiple are provided, writes a 3D line
            plot with token index as the second axis. For 2D ``acts``, pass an
            empty list.
        metric_fn: Function comparing two layer vectors.
        save_path: Destination image path.
        layers: Optional decoder-layer labels.
        title: Optional plot title.

    Returns:
        The saved path.
    """
    values = _metric_matrix(acts, tok_idxs, metric_fn)
    layer_labels = (
        list(layers) if layers is not None else list(range(_layer_count(acts)))
    )
    x = np.arange(values.shape[1])
    x_labels = [str(layer) for layer in layer_labels[:-1]]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if values.shape[0] == 1:
        _plot_2d(values[0], x, x_labels, metric_fn.__name__, save_path, title=title)
    else:
        _plot_3d(
            values,
            x,
            list(tok_idxs),
            x_labels,
            metric_fn.__name__,
            save_path,
            title=title,
        )
    return save_path


def _plot_matrix(
    values: np.ndarray,
    tok_idxs: Sequence[int],
    save_path: Path,
    *,
    layers: Sequence[int] | None = None,
    metric_label: str,
    title: str | None = None,
) -> Path:
    layer_labels = (
        list(layers) if layers is not None else list(range(values.shape[1] + 1))
    )
    x = np.arange(values.shape[1])
    x_labels = [str(layer) for layer in layer_labels[:-1]]

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if values.shape[0] == 1:
        _plot_2d(values[0], x, x_labels, metric_label, save_path, title=title)
    else:
        _plot_3d(
            values,
            x,
            list(tok_idxs),
            x_labels,
            metric_label,
            save_path,
            title=title,
        )
    return save_path


def _metric_matrix(
    acts: np.ndarray,
    tok_idxs: Sequence[int],
    metric_fn: MetricFn,
) -> np.ndarray:
    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim == 2:
        if tok_idxs:
            raise ValueError(
                "tok_idxs must be empty when acts has shape [layers, hidden]"
            )
        return np.asarray([_adjacent_metrics(acts, metric_fn)], dtype=np.float32)
    if acts.ndim != 3:
        raise ValueError(f"Expected [tokens, layers, hidden], got {acts.shape}")
    if not tok_idxs:
        raise ValueError("Pass at least one token index for 3D activation arrays")
    rows = []
    for tok_idx in tok_idxs:
        if not 0 <= int(tok_idx) < acts.shape[0]:
            raise IndexError(
                f"Token index {tok_idx} outside trace length {acts.shape[0]}"
            )
        rows.append(_adjacent_metrics(acts[int(tok_idx)], metric_fn))
    return np.asarray(rows, dtype=np.float32)


def _adjacent_metrics(layer_states: np.ndarray, metric_fn: MetricFn) -> list[float]:
    return [
        metric_fn(layer_states[layer_idx], layer_states[layer_idx + 1])
        for layer_idx in range(layer_states.shape[0] - 1)
    ]


def _directional_persistence_matrix(
    acts: np.ndarray,
    tok_idxs: Sequence[int],
) -> np.ndarray:
    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim != 3:
        raise ValueError(f"Expected [tokens, layers, hidden], got {acts.shape}")
    if acts.shape[0] < 2:
        return np.empty((0, max(acts.shape[1] - 1, 0)), dtype=np.float32)
    values = []
    for tok_idx in tok_idxs:
        idx = int(tok_idx)
        if not 0 <= idx < acts.shape[0] - 1:
            raise IndexError(
                f"Token index {tok_idx} outside transition length {acts.shape[0] - 1}"
            )
        current = _normalized_layer_updates(acts[idx])
        following = _normalized_layer_updates(acts[idx + 1])
        values.append(
            [
                cosine_similarity(current[layer_idx], following[layer_idx])
                for layer_idx in range(current.shape[0])
            ]
        )
    return np.asarray(values, dtype=np.float32)


def _normalized_layer_updates(layer_states: np.ndarray) -> np.ndarray:
    normalized = np.asarray([_unit_vector(row) for row in layer_states], dtype=np.float32)
    return normalized[1:] - normalized[:-1]


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return np.asarray(vector, dtype=np.float32) / norm


def _empty_web_payload(
    metric_name: str,
    metric_label: str,
    max_3d_tokens: int,
) -> dict[str, Any]:
    return {
        "metric": metric_name,
        "metric_label": metric_label,
        "max_3d_tokens": max_3d_tokens,
        "traces": [],
        "points": [],
    }


def _append_web_points(
    payload: dict[str, Any],
    row: dict[str, Any],
    row_index: int,
    states: np.ndarray,
    layers: Sequence[int],
    sampled_tokens: Sequence[int],
    metric_fn: MetricFn,
) -> None:
    """Append one generation trace as web-ready adjacent-layer metric points."""
    layer_pairs = [
        {
            "index": idx,
            "from": int(layers[idx]),
            "to": int(layers[idx + 1]),
            "label": f"{layers[idx]}->{layers[idx + 1]}",
        }
        for idx in range(len(layers) - 1)
    ]
    trace = {
        "sample_id": row.get("sample_id"),
        "seed": row.get("seed"),
        "row_index": row_index,
        "is_correct": row.get("is_correct"),
        "produced_answer": row.get("produced_answer"),
        "token_count": int(states.shape[0]),
        "layer_pairs": layer_pairs,
        "points": [],
    }
    denominator = max(1, states.shape[0] - 1)
    for token_idx in sampled_tokens:
        values = _adjacent_metrics(states[int(token_idx)], metric_fn)
        for pair, value in zip(layer_pairs, values):
            trace["points"].append(
                {
                    "sample_id": row.get("sample_id"),
                    "seed": row.get("seed"),
                    "row_index": row_index,
                    "is_correct": row.get("is_correct"),
                    "produced_answer": row.get("produced_answer"),
                    "token_idx": int(token_idx),
                    "token_fraction": round(float(token_idx) / denominator, 6),
                    "layer_pair_index": pair["index"],
                    "layer_from": pair["from"],
                    "layer_to": pair["to"],
                    "layer_pair": pair["label"],
                    "value": round(float(value), 6),
                    "x": pair["index"],
                    "y": int(token_idx),
                    "z": round(float(value), 6),
                }
            )
    payload["traces"].append(trace)


def _append_web_matrix_points(
    payload: dict[str, Any],
    row: dict[str, Any],
    row_index: int,
    states: np.ndarray,
    layers: Sequence[int],
    sampled_tokens: Sequence[int],
    matrix: np.ndarray,
) -> None:
    """Append precomputed layer-pair values as web-ready points."""
    layer_pairs = [
        {
            "index": idx,
            "from": int(layers[idx]),
            "to": int(layers[idx + 1]),
            "label": f"{layers[idx]}->{layers[idx + 1]}",
        }
        for idx in range(len(layers) - 1)
    ]
    trace = {
        "sample_id": row.get("sample_id"),
        "seed": row.get("seed"),
        "row_index": row_index,
        "is_correct": row.get("is_correct"),
        "produced_answer": row.get("produced_answer"),
        "token_count": int(states.shape[0]),
        "layer_pairs": layer_pairs,
        "points": [],
    }
    denominator = max(1, states.shape[0] - 1)
    for token_idx, values in zip(sampled_tokens, matrix):
        for pair, value in zip(layer_pairs, values):
            trace["points"].append(
                {
                    "sample_id": row.get("sample_id"),
                    "seed": row.get("seed"),
                    "row_index": row_index,
                    "is_correct": row.get("is_correct"),
                    "produced_answer": row.get("produced_answer"),
                    "token_idx": int(token_idx),
                    "token_fraction": round(float(token_idx) / denominator, 6),
                    "layer_pair_index": pair["index"],
                    "layer_from": pair["from"],
                    "layer_to": pair["to"],
                    "layer_pair": pair["label"],
                    "value": round(float(value), 6),
                    "x": pair["index"],
                    "y": int(token_idx),
                    "z": round(float(value), 6),
                }
            )
    payload["traces"].append(trace)


def _plot_2d(
    values: np.ndarray,
    x: np.ndarray,
    x_labels: list[str],
    metric_name: str,
    save_path: Path,
    *,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    points = np.column_stack([x, values])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    collection = LineCollection(segments, cmap="coolwarm", norm=norm, linewidth=2.5)
    collection.set_array((values[:-1] + values[1:]) / 2.0)
    ax.add_collection(collection)
    ax.scatter(x, values, c=values, cmap="coolwarm", norm=norm, s=28, zorder=3)
    ax.autoscale()
    ax.set_ylim(_axis_limits_with_zero(values))
    ax.set_title(title or f"{metric_name} between adjacent layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    fig.colorbar(collection, ax=ax, label=metric_name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _plot_3d(
    values: np.ndarray,
    x: np.ndarray,
    tok_idxs: list[int],
    x_labels: list[str],
    metric_name: str,
    save_path: Path,
    *,
    title: str | None = None,
) -> None:
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    cmap = plt.get_cmap("coolwarm")
    for row_idx, token_values in enumerate(values):
        color = cmap(norm(float(np.mean(token_values))))
        ax.plot(
            x,
            [tok_idxs[row_idx]] * len(x),
            token_values,
            color=color,
            linewidth=1.8,
        )
    ax.set_title(title or f"{metric_name} between adjacent layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token index")
    ax.set_zlabel(metric_name)
    ax.set_zlim(_axis_limits_with_zero(values))
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.65,
        label=metric_name,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _layer_count(acts: np.ndarray) -> int:
    return acts.shape[0] if acts.ndim == 2 else acts.shape[1]


def _axis_limits_with_zero(values: np.ndarray) -> tuple[float, float]:
    low = min(0.0, float(np.min(values)))
    high = max(0.0, float(np.max(values)))
    if low == high:
        pad = 1.0 if low == 0.0 else abs(low) * 0.05
        return low - pad, high + pad
    pad = (high - low) * 0.03
    return low - pad, high + pad


def _sample_token_indices(token_count: int, max_tokens: int) -> list[int]:
    if max_tokens <= 0 or token_count <= max_tokens:
        return list(range(token_count))
    step = max(1, round((token_count - 1) / max(1, max_tokens - 1)))
    indices = list(range(0, token_count, step))
    if len(indices) > max_tokens:
        indices = indices[:max_tokens]
    return indices


def _plot_stem(row: dict, row_index: int, model_name: str) -> str:
    safe_model = _safe_path_part(model_name.split("/")[-1])
    sample_id = str(row.get("sample_id", f"trace_{row_index}"))
    safe_sample = _safe_path_part(sample_id)
    return f"{row_index:04d}_{safe_model}_{safe_sample}_seed{row.get('seed', row_index)}"


def _model_name(run_path: Path) -> str:
    config_path = run_path / "config.yaml"
    if not config_path.exists():
        return run_path.parts[-3] if len(run_path.parts) >= 3 else run_path.name
    model_cfg = load_run_config(run_path).get("model", {})
    return str(model_cfg.get("name") or model_cfg.get("path") or run_path.name)


def _safe_path_part(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


if __name__ == "__main__":
    # choose the number of tokens to plot along the depth dim, since all may be too much
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot adjacent-layer metrics for hidden-state traces."
    )
    parser.add_argument("run_path", type=Path)
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        default=160,
        help="target number of regularly spaced generated tokens in each 3D plot",
    )
    args = parser.parse_args()

    write_layer_plots(
        args.run_path,
        {
            "layer_variations": {
                "max_3d_tokens": args.tokens,
            }
        },
    )
