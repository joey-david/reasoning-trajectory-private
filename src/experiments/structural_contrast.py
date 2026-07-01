"""Learn and evaluate a contrastive projection over symbolic update vectors."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import torch

from src.runtime.data import load_samples, write_jsonl


def run_structural_contrast(
    h2_dir: Path,
    *,
    layer: int = -1,
    max_updates: int = 12000,
    max_pairs: int = 20000,
    epochs: int = 12,
    projection_dim: int = 128,
) -> Path:
    """Run H4 from H2's verified update records and transition vectors."""
    records = load_samples((h2_dir / "updates.jsonl").resolve())
    with np.load(h2_dir / f"layer{layer}_update_vectors.npz") as data:
        vector_key = (
            "net_update_vectors" if "net_update_vectors" in data else "delta_vectors"
        )
        vectors = data[vector_key].astype(np.float32)
    out_dir = h2_dir.parent / "h4_structural_contrast"
    return fit_structural_projection(
        records=records,
        vectors=vectors,
        out_dir=out_dir,
        projection_filename=f"layer{layer}_projection.pt",
        source=h2_dir.as_posix(),
        layer=layer,
        component="residual",
        update_vector=(
            "net displacement across the verified symbolic interval"
            if vector_key == "net_update_vectors"
            else "legacy single-token endpoint delta"
        ),
        max_updates=max_updates,
        max_pairs=max_pairs,
        epochs=epochs,
        projection_dim=projection_dim,
        output_prefix="",
    )


def fit_structural_projection(
    *,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    out_dir: Path,
    projection_filename: str,
    source: str,
    layer: int,
    component: str,
    update_vector: str,
    max_updates: int = 12000,
    max_pairs: int = 20000,
    epochs: int = 12,
    projection_dim: int = 128,
    output_prefix: str = "",
    write_pair_manifests: bool = True,
) -> Path:
    """Fit one controlled structural projection from aligned records and vectors."""
    selected = select_structural_updates(records, max_updates=max_updates)
    record_indices = np.asarray(
        [record["feature_row"] for record in selected], dtype=int
    )
    x = vectors[record_indices]
    groups = np.asarray([str(record["sample_id"]) for record in selected])

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_indices, test_indices = next(split.split(x, groups=groups))
    train_pairs = mine_controlled_pairs(
        selected, train_indices, max_pairs=max_pairs, seed=42
    )
    test_pairs = mine_controlled_pairs(
        selected, test_indices, max_pairs=max_pairs // 3, seed=43
    )
    if not train_pairs or not test_pairs:
        raise ValueError("Lexical controls produced too few contrastive pairs")

    projection, losses = fit_contrastive_projection(
        x,
        train_pairs,
        projection_dim=projection_dim,
        epochs=epochs,
    )
    raw_scores, labels = pair_scores(x, test_pairs)
    projected = project_vectors(x, projection)
    projected_scores, _ = pair_scores(projected, test_pairs)
    singular_values = torch.linalg.svdvals(projection.float())
    matrix_rank = int(torch.linalg.matrix_rank(projection.float()).item())
    condition_number = float(
        singular_values.max() / singular_values.clamp_min(1e-12).min()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_prefix = f"{output_prefix}_" if output_prefix else ""
    if write_pair_manifests:
        write_jsonl(
            out_dir / f"{pair_prefix}train_pairs.jsonl",
            pair_records(train_pairs, selected),
        )
        write_jsonl(
            out_dir / f"{pair_prefix}test_pairs.jsonl",
            pair_records(test_pairs, selected),
        )
    torch.save(
        {
            "weight": projection,
            "input_dim": x.shape[1],
            "projection_dim": projection_dim,
            "layer": layer,
            "component": component,
            "source": source,
            "update_vector": update_vector,
            "matrix_rank": matrix_rank,
            "condition_number": condition_number,
        },
        out_dir / projection_filename,
    )
    report = {
        "hypothesis": "H4_contrastive_structural_discovery",
        "source": source,
        "component": component,
        "layer": layer,
        "update_vector": update_vector,
        "projection_artifact": (out_dir / projection_filename).as_posix(),
        "selection": {
            "updates": len(selected),
            "questions": len(set(groups)),
            "signature_counts": dict(
                Counter(record["operation_signature"] for record in selected)
            ),
        },
        "pairs": {
            "train": pair_summary(train_pairs),
            "test": pair_summary(test_pairs),
            "positive_rule": "same operation, different questions, zero literal overlap",
            "negative_rule": "different operation within the same question",
        },
        "training": {
            "epochs": epochs,
            "projection_dim": projection_dim,
            "losses": losses,
            "matrix_rank": matrix_rank,
            "condition_number": condition_number,
        },
        "evaluation": {
            "question_disjoint": True,
            "raw_cosine_auc": float(roc_auc_score(labels, raw_scores)),
            "projected_cosine_auc": float(roc_auc_score(labels, projected_scores)),
        },
    }
    report_path = out_dir / f"{pair_prefix}report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def select_structural_updates(
    records: list[dict[str, Any]],
    *,
    max_updates: int,
) -> list[dict[str, Any]]:
    """Retain common atomic arithmetic operations and cap classes evenly."""
    allowed = {"ADD", "SUBTRACT", "MULTIPLY", "DIVIDE"}
    by_signature: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        signature = str(record.get("operation_signature"))
        if signature in allowed:
            by_signature[signature].append(record)
    per_class = max(max_updates // max(len(by_signature), 1), 1)
    selected: list[dict[str, Any]] = []
    for signature in sorted(by_signature):
        candidates = sorted(
            by_signature[signature],
            key=lambda record: (
                str(record["sample_id"]),
                int(record["seed"]),
                int(record["token_end"]),
            ),
        )
        selected.extend(candidates[:per_class])
    return selected


def mine_controlled_pairs(
    records: list[dict[str, Any]],
    eligible: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    """Mine positive and hard-negative pairs without quadratic enumeration."""
    rng = np.random.default_rng(seed)
    indices = [int(index) for index in eligible]
    by_signature: defaultdict[str, list[int]] = defaultdict(list)
    by_sample: defaultdict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_signature[str(records[index]["operation_signature"])].append(index)
        by_sample[str(records[index]["sample_id"])].append(index)

    positives: list[tuple[int, int, int]] = []
    negatives: list[tuple[int, int, int]] = []
    shuffled = [int(index) for index in rng.permutation(indices)]
    target = max_pairs // 2
    for anchor in shuffled:
        record = records[anchor]
        lexical = set(record.get("lexical_items", []))
        positive_pool = by_signature[str(record["operation_signature"])]
        for candidate in rng.permutation(positive_pool)[:64]:
            candidate = int(candidate)
            other = records[candidate]
            if (
                candidate != anchor
                and other["sample_id"] != record["sample_id"]
                and lexical.isdisjoint(other.get("lexical_items", []))
            ):
                positives.append((anchor, candidate, 1))
                break

        negative_pool = by_sample[str(record["sample_id"])]
        hard_negative = None
        hard_overlap = -1.0
        for candidate in rng.permutation(negative_pool)[:64]:
            candidate = int(candidate)
            other = records[candidate]
            if other["operation_signature"] == record["operation_signature"]:
                continue
            overlap = lexical_jaccard(lexical, set(other.get("lexical_items", [])))
            if overlap > hard_overlap:
                hard_overlap = overlap
                hard_negative = candidate
        if hard_negative is not None:
            negatives.append((anchor, hard_negative, 0))
        if len(positives) >= target and len(negatives) >= target:
            break
    count = min(len(positives), len(negatives), target)
    combined = positives[:count] + negatives[:count]
    rng.shuffle(combined)
    return combined


def fit_contrastive_projection(
    vectors: np.ndarray,
    pairs: list[tuple[int, int, int]],
    *,
    projection_dim: int,
    epochs: int,
) -> tuple[torch.Tensor, list[float]]:
    """Fit a bias-free linear projection with cosine contrastive loss."""
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    x = torch.from_numpy(vectors).to(device)
    pair_array = np.asarray(pairs, dtype=np.int64)
    model = torch.nn.Linear(x.shape[1], projection_dim, bias=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    losses: list[float] = []
    batch_size = 256
    for _epoch in range(epochs):
        order = torch.randperm(len(pair_array), device=device)
        epoch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            rows = pair_array[order[start : start + batch_size].cpu().numpy()]
            left = torch.nn.functional.normalize(model(x[rows[:, 0]]), dim=1)
            right = torch.nn.functional.normalize(model(x[rows[:, 1]]), dim=1)
            similarity = torch.sum(left * right, dim=1)
            labels = torch.from_numpy(rows[:, 2].astype(np.float32)).to(device)
            positive_loss = labels * (1.0 - similarity)
            negative_loss = (1.0 - labels) * torch.relu(similarity - 0.1)
            loss = (positive_loss + negative_loss).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))
    return model.weight.detach().cpu(), losses


def project_vectors(vectors: np.ndarray, weight: torch.Tensor) -> np.ndarray:
    projected = vectors @ weight.numpy().T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    return projected / np.maximum(norms, 1e-8)


def pair_scores(
    vectors: np.ndarray,
    pairs: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    normalized = vectors / np.maximum(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8
    )
    return (
        np.asarray(
            [float(normalized[left] @ normalized[right]) for left, right, _ in pairs]
        ),
        np.asarray([label for _, _, label in pairs]),
    )


def pair_records(
    pairs: list[tuple[int, int, int]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "left": int(left),
            "right": int(right),
            "positive": bool(label),
            "left_sample_id": records[left]["sample_id"],
            "right_sample_id": records[right]["sample_id"],
            "left_operation": records[left]["operation_signature"],
            "right_operation": records[right]["operation_signature"],
            "lexical_overlap": lexical_jaccard(
                set(records[left].get("lexical_items", [])),
                set(records[right].get("lexical_items", [])),
            ),
        }
        for left, right, label in pairs
    ]


def pair_summary(pairs: list[tuple[int, int, int]]) -> dict[str, int]:
    counts = Counter(label for _, _, label in pairs)
    return {
        "total": len(pairs),
        "positive": counts[1],
        "negative": counts[0],
    }


def lexical_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
