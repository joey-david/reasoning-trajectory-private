"""End-to-end preparation, capture, analysis, and causal execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .decoders import decode_representation
from .features import capture_bank_features, load_activation_model
from .iso_dataset import build_bank
from .mixed_trajectories import analyze_mixed_trajectories
from .patching import causal_reports
from .projections import fit_projection_bundle, project
from .reports import build_object_records, overall_summary, retrieval_examples
from .retrieval import (
    evaluate_by_split,
    evaluate_retrieval,
    lexical_features,
    surface_gap,
)
from .rsa import run_rsa
from .storage import (
    load_experiment_config,
    output_dir,
    read_jsonl,
    write_json,
    write_jsonl,
    write_npz,
)
from .trajectory import reranking_report, trajectory_report


EVALUATION_SPLITS = (
    "validation",
    "heldout_vocab",
    "heldout_template",
    "heldout_question",
)


def run_prepare(run_path: Path) -> dict[str, Any]:
    """Materialize the controlled bank and canonical graph inventory."""
    loaded = load_experiment_config(run_path)
    experiment = loaded["experiment"]
    bank_cfg = experiment["bank"]
    graphs, records = build_bank(
        graph_count=int(bank_cfg["graph_count"]),
        include_corruptions=bool(bank_cfg.get("include_corruptions", True)),
        splits=[str(split) for split in bank_cfg.get("splits", [])] or None,
    )
    validate_bank(records)
    out = output_dir(run_path)
    write_jsonl(run_path / "dataset.jsonl", records)
    write_jsonl(out / "canonical_graphs.jsonl", graphs)
    write_json(
        out / "config.json",
        {
            "run_path": run_path.as_posix(),
            "solution_object_extraction": experiment,
            "model": loaded["run"]["model"],
        },
    )
    report = {
        "graphs": len(graphs),
        "records": len(records),
        "splits": {
            split: sum(row["split"] == split for row in records)
            for split in sorted({row["split"] for row in records})
        },
        "correct_records": sum(bool(row["is_correct"]) for row in records),
        "corrupted_records": sum(not bool(row["is_correct"]) for row in records),
    }
    write_json(out / "bank_report.json", report)
    return report


def validate_bank(records: list[dict[str, Any]]) -> None:
    """Enforce split and class coverage before model work."""
    required = {"train", *EVALUATION_SPLITS}
    present = {str(row["split"]) for row in records}
    if not required.issubset(present):
        raise ValueError(f"Bank is missing splits: {sorted(required - present)}")
    train_graphs = {
        row["canonical_graph_id"] for row in records if row["split"] == "train"
    }
    heldout_graphs = {
        row["canonical_graph_id"]
        for row in records
        if row["split"] in EVALUATION_SPLITS
    }
    missing = heldout_graphs - train_graphs
    if missing:
        raise ValueError(
            f"{len(missing)} evaluation graph states lack train prototypes"
        )
    for row in records:
        text = str(row["text"])
        anchor = str(row["anchor_text"])
        if text.count(anchor) != 1:
            raise ValueError(f"{row['record_id']} has a non-unique anchor")


def run_capture(run_path: Path) -> dict[str, Any]:
    """Capture selected hidden layers for the prepared bank."""
    loaded = load_experiment_config(run_path)
    experiment = loaded["experiment"]
    capture_cfg = experiment["capture"]
    bank_path = run_path / "dataset.jsonl"
    if not bank_path.exists():
        run_prepare(run_path)
    capture_path = output_dir(run_path) / "captured_features.npz"
    expected_records = read_jsonl(bank_path)
    expected_ids = np.asarray(
        [row["record_id"] for row in expected_records], dtype=str
    )
    expected_layers = np.asarray(
        [int(layer) for layer in capture_cfg["layers"]], dtype=np.int32
    )
    if capture_path.exists():
        with np.load(capture_path) as data:
            compatible = (
                {
                    "record_ids",
                    "layers",
                    "h_pool",
                    "h_last",
                    "h_last_two",
                    "h_pre_anchor",
                    "h_delta",
                    "h_text_mean",
                    "token_ranges",
                }.issubset(data.files)
                and np.array_equal(data["record_ids"].astype(str), expected_ids)
                and np.array_equal(data["layers"].astype(np.int32), expected_layers)
                and data["h_pool"].shape[0] == len(expected_ids)
            )
            hidden_size = int(data["h_pool"].shape[-1]) if compatible else None
        if compatible:
            report = {
                "records": len(expected_ids),
                "layers": expected_layers.astype(int).tolist(),
                "hidden_size": hidden_size,
                "reused": True,
                "capture_path": capture_path.as_posix(),
            }
            write_json(output_dir(run_path) / "capture_report.json", report)
            return report
    report = capture_bank_features(
        bank_path=bank_path,
        output_path=capture_path,
        model_cfg=loaded["run"]["model"],
        layers=[int(layer) for layer in capture_cfg["layers"]],
        batch_size=int(capture_cfg.get("batch_size", 4)),
    )
    report["reused"] = False
    write_json(output_dir(run_path) / "capture_report.json", report)
    return report


def run_analysis(run_path: Path) -> dict[str, Any]:
    """Fit projections and run Experiments B-D, G, and H."""
    loaded = load_experiment_config(run_path)
    experiment = loaded["experiment"]
    out = output_dir(run_path)
    records = read_jsonl(run_path / "dataset.jsonl")
    feature_path = out / "captured_features.npz"
    if not feature_path.exists():
        raise FileNotFoundError(f"Run capture first: {feature_path}")
    with np.load(feature_path) as data:
        h_pool = data["h_pool"].astype(np.float32)
        h_last = data["h_last"].astype(np.float32)
        h_text_mean = data["h_text_mean"].astype(np.float32)
        record_ids = data["record_ids"].astype(str)
        layers = data["layers"].astype(int).tolist()
        token_ranges = data["token_ranges"].astype(int)
    expected_ids = np.asarray([row["record_id"] for row in records], dtype=str)
    if not np.array_equal(record_ids, expected_ids):
        raise ValueError("Captured feature rows do not match the prepared bank")
    train_indices = np.asarray(
        [index for index, row in enumerate(records) if row["split"] == "train"],
        dtype=int,
    )
    max_dim = int(experiment["projection"].get("dimension", 32))
    layer_results: list[dict[str, Any]] = []
    bundles: dict[int, dict[str, np.ndarray]] = {}
    per_layer_vectors: dict[int, dict[str, np.ndarray]] = {}
    for layer_col, layer in enumerate(
        tqdm(layers, desc="projection layer sweep", unit="layer")
    ):
        # A current object state depends on the local edit and its accumulated
        # prefix. Equal-weight pooling keeps the vector in the residual space
        # needed for causal interventions while avoiding anchor-only result
        # shortcuts observed in the first local pilot.
        vectors = 0.5 * (
            h_pool[:, layer_col] + h_text_mean[:, layer_col]
        )
        bundle = fit_projection_bundle(
            vectors, records, train_indices, max_dim=max_dim
        )
        bundles[layer] = bundle
        representations = {
            "object": project(
                vectors, bundle["object_mean"], bundle["object_basis"]
            ),
            "pca": project(vectors, bundle["pca_mean"], bundle["pca_basis"]),
            "random": project(
                vectors, bundle["random_mean"], bundle["random_basis"]
            ),
            "raw": vectors,
            "anchor_pool": h_pool[:, layer_col],
            "last_token": h_last[:, layer_col],
            "full_text_mean": h_text_mean[:, layer_col],
        }
        per_layer_vectors[layer] = representations
        reports = {}
        for name, representation in representations.items():
            reports[name], _ = evaluate_by_split(
                train_vectors=representation[train_indices],
                all_vectors=representation,
                records=records,
                train_indices=train_indices,
                splits=EVALUATION_SPLITS,
            )
        reports["object_similarity_gap"] = surface_gap(
            representations["object"],
            records,
            np.arange(len(records)),
        )
        layer_results.append({"layer": layer, "representations": reports})
    selected_layer = max(
        layers,
        key=lambda layer: (
            next(
                row["representations"]["object"]["validation"]["top1"]
                for row in layer_results
                if row["layer"] == layer
            ),
            next(
                row["representations"]["object"]["validation"][
                    "mean_retrieval_margin"
                ]
                or -1e9
                for row in layer_results
                if row["layer"] == layer
            ),
        ),
    )
    bundle = bundles[selected_layer]
    representations = per_layer_vectors[selected_layer]
    object_reports, per_record = evaluate_by_split(
        train_vectors=representations["object"][train_indices],
        all_vectors=representations["object"],
        records=records,
        train_indices=train_indices,
        splits=EVALUATION_SPLITS,
    )
    selected_reports: dict[str, Any] = {"object": object_reports}
    for name in (
        "raw",
        "pca",
        "random",
        "anchor_pool",
        "last_token",
        "full_text_mean",
    ):
        selected_reports[name], _ = evaluate_by_split(
            train_vectors=representations[name][train_indices],
            all_vectors=representations[name],
            records=records,
            train_indices=train_indices,
            splits=EVALUATION_SPLITS,
        )
    lexical_reports = {}
    train_records = [records[index] for index in train_indices]
    for split in EVALUATION_SPLITS:
        split_indices = np.asarray(
            [index for index, row in enumerate(records) if row["split"] == split],
            dtype=int,
        )
        test_records = [records[index] for index in split_indices]
        lexical_train, lexical_test, lexical_meta = lexical_features(
            train_records, test_records
        )
        report, _, _ = evaluate_retrieval(
            lexical_train,
            lexical_test,
            np.asarray(
                [row["canonical_graph_id"] for row in train_records], dtype=str
            ),
            np.asarray(
                [row["canonical_graph_id"] for row in test_records], dtype=str
            ),
            test_records,
        )
        lexical_reports[split] = {**report, **lexical_meta}
    selected_reports["lexical"] = lexical_reports
    retrieval_report = {
        "selection_rule": (
            "highest validation top-1 object retrieval, then validation margin"
        ),
        "selected_layer": selected_layer,
        "layer_sweep": layer_results,
        "selected": selected_reports,
        "positive_threshold": {
            "heldout_vocab_object_top1": object_reports["heldout_vocab"]["top1"],
            "heldout_vocab_lexical_top1": lexical_reports["heldout_vocab"]["top1"],
            "object_beats_lexical": (
                object_reports["heldout_vocab"]["top1"]
                > lexical_reports["heldout_vocab"]["top1"]
            ),
        },
    }
    write_json(out / "retrieval_report.json", retrieval_report)
    projection_id = (
        f"objproj_smollm_l{selected_layer}_d"
        f"{bundle['object_basis'].shape[0]}_v1"
    )
    write_npz(out / "projection.npz", **bundle)
    write_json(
        out / "projection_manifest.json",
        {
            "projection_id": projection_id,
            "selected_layer": selected_layer,
            "input_dim": int(bundle["object_basis"].shape[1]),
            "object_dim": int(bundle["object_basis"].shape[0]),
            "fit_split": "train",
            "selection_split": "validation",
            "selection_metric": "top1_then_margin",
            "pooling": "0.5 * anchor_mean + 0.5 * accumulated_prefix_mean",
            "controls": ["pca", "random", "lexical_family"],
            "projection_file": "projection.npz",
        },
    )
    object_rows = build_object_records(
        records=records,
        token_ranges=token_ranges,
        layer=selected_layer,
        projection_id=projection_id,
        per_record=per_record,
        model=str(loaded["run"]["model"]["name"]),
    )
    write_jsonl(out / "object_records.jsonl", object_rows)
    write_npz(
        out / "object_vectors.npz",
        z_obj=representations["object"].astype(np.float16),
        h_pool=(
            0.5
            * (
                h_pool[:, layers.index(selected_layer)]
                + h_text_mean[:, layers.index(selected_layer)]
            )
        ).astype(np.float16),
        record_ids=record_ids,
    )
    evaluation_indices = np.asarray(
        [
            index
            for index, row in enumerate(records)
            if row["split"] in {"heldout_vocab", "heldout_template"}
        ],
        dtype=int,
    )
    rsa_report = run_rsa(
        representations["object"][evaluation_indices],
        [records[index] for index in evaluation_indices],
    )
    write_json(out / "rsa_report.json", rsa_report)
    decoder_report = run_decoders(
        representations, records, train_indices, EVALUATION_SPLITS
    )
    write_json(out / "decoder_report.json", decoder_report)
    controlled_trajectory = trajectory_report(
        representations["object"], records
    )
    controlled_reranking = reranking_report(representations["object"], records)
    write_json(out / "controlled_trajectory_report.json", controlled_trajectory)
    write_json(out / "controlled_reranking_report.json", controlled_reranking)
    mixed_cfg = experiment["mixed_trajectories"]
    mixed_source = Path(str(mixed_cfg["source_run"]))
    mixed_status = mixed_source_status(mixed_source)
    if mixed_status["available"]:
        mixed_layer = int(mixed_cfg.get("layer", -1))
        projection_layer = layers[-1] if mixed_layer == -1 else mixed_layer
        if projection_layer not in bundles:
            raise ValueError(
                f"Mixed trajectory layer {mixed_layer} requires controlled capture "
                f"layer {projection_layer}"
            )
        mixed_bundle = bundles[projection_layer]
        trajectory, reranking = analyze_mixed_trajectories(
            source_run=mixed_source,
            projection_mean=mixed_bundle["object_mean"],
            projection_basis=mixed_bundle["object_basis"],
            layer=mixed_layer,
            per_sample=int(mixed_cfg.get("per_sample", 10)),
            max_questions=int(mixed_cfg.get("max_questions", 58)),
        )
    elif bool(mixed_cfg.get("required", True)):
        raise FileNotFoundError(str(mixed_status["message"]))
    else:
        skipped = {
            "status": "skipped",
            "reason": str(mixed_status["message"]),
            "source_run": mixed_source.as_posix(),
            "note": (
                "G/H use an existing analysis corpus and may be run after pulling "
                "the medium capture to a checkout that has those hidden states."
            ),
        }
        trajectory = skipped
        reranking = dict(skipped)
    write_json(out / "trajectory_report.json", trajectory)
    write_json(out / "reranking_report.json", reranking)
    write_jsonl(
        out / "examples.jsonl", retrieval_examples(records, per_record)
    )
    summary = overall_summary(
        retrieval=retrieval_report,
        rsa=rsa_report,
        decoder=decoder_report,
        patching=None,
    )
    write_json(out / "summary.json", summary)
    return {
        "selected_layer": selected_layer,
        "object_dim": int(bundle["object_basis"].shape[0]),
        "summary": summary,
    }


def run_decoders(
    representations: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    train_indices: np.ndarray,
    splits: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate factorized probes across prespecified surface splits."""
    report: dict[str, Any] = {}
    train_records = [records[index] for index in train_indices]
    for split in tqdm(splits, desc="factorized decoders", unit="split"):
        test_indices = np.asarray(
            [index for index, row in enumerate(records) if row["split"] == split],
            dtype=int,
        )
        test_records = [records[index] for index in test_indices]
        report[split] = {
            name: decode_representation(
                values[train_indices],
                values[test_indices],
                train_records,
                test_records,
            )
            for name, values in {
                "object": representations["object"],
                "raw": representations["raw"],
                "pca": representations["pca"],
                "random": representations["random"],
            }.items()
        }
        lexical_train, lexical_test, _ = lexical_features(
            train_records, test_records
        )
        report[split]["lexical"] = decode_representation(
            lexical_train, lexical_test, train_records, test_records
        )
    return report


def run_causal(run_path: Path) -> dict[str, Any]:
    """Run Experiments E and F with the selected object subspace."""
    loaded = load_experiment_config(run_path)
    out = output_dir(run_path)
    manifest = json.loads((out / "projection_manifest.json").read_text())
    with np.load(out / "projection.npz") as data:
        bundle = {key: data[key].astype(np.float32) for key in data.files}
    records = read_jsonl(run_path / "dataset.jsonl")
    model, tokenizer, _device = load_activation_model(loaded["run"]["model"])
    causal_cfg = loaded["experiment"].get("causal", {})
    patching, ablation, details = causal_reports(
        model=model,
        tokenizer=tokenizer,
        records=records,
        layer=int(manifest["selected_layer"]),
        object_mean=bundle["object_mean"],
        object_basis=bundle["object_basis"],
        random_mean=bundle["random_mean"],
        random_basis=bundle["random_basis"],
        lexical_mean=bundle["lexical_mean"],
        lexical_basis=bundle["lexical_basis"],
        max_pairs_per_condition=int(causal_cfg.get("max_pairs_per_condition", 4)),
        continuation_tokens=int(causal_cfg.get("continuation_tokens", 0)),
    )
    write_json(out / "patching_report.json", patching)
    write_json(out / "ablation_report.json", ablation)
    write_jsonl(out / "patching_examples.jsonl", details)
    retrieval = json.loads((out / "retrieval_report.json").read_text())
    rsa = json.loads((out / "rsa_report.json").read_text())
    decoder = json.loads((out / "decoder_report.json").read_text())
    summary = overall_summary(
        retrieval=retrieval,
        rsa=rsa,
        decoder=decoder,
        patching=patching,
    )
    write_json(out / "summary.json", summary)
    return {"patching": patching, "ablation": ablation, "summary": summary}


def validate_run(run_path: Path, *, require_capture: bool = False) -> dict[str, Any]:
    """Validate a prepared small or medium run without loading a model."""
    loaded = load_experiment_config(run_path)
    records = read_jsonl(run_path / "dataset.jsonl")
    validate_bank(records)
    layers = [int(layer) for layer in loaded["experiment"]["capture"]["layers"]]
    if len(layers) != len(set(layers)):
        raise ValueError("capture layers must be unique")
    capture_path = output_dir(run_path) / "captured_features.npz"
    if require_capture and not capture_path.exists():
        raise FileNotFoundError(capture_path)
    mixed_source = Path(
        str(loaded["experiment"]["mixed_trajectories"]["source_run"])
    )
    mixed_status = mixed_source_status(mixed_source)
    mixed_required = bool(
        loaded["experiment"]["mixed_trajectories"].get("required", True)
    )
    if mixed_required and not mixed_status["available"]:
        raise FileNotFoundError(str(mixed_status["message"]))
    return {
        "run_path": run_path.as_posix(),
        "records": len(records),
        "layers": layers,
        "capture_present": capture_path.exists(),
        "mixed_trajectory_source": mixed_source.as_posix(),
        "mixed_trajectory_available": mixed_status["available"],
        "mixed_trajectory_required": mixed_required,
        "mixed_trajectory_note": mixed_status["message"],
        "valid": True,
    }


def mixed_source_status(source_run: Path) -> dict[str, Any]:
    """Report whether a completed mixed-trajectory corpus is locally readable."""
    index_path = source_run / "generation" / "generations.jsonl"
    if not index_path.exists():
        return {
            "available": False,
            "message": f"mixed-trajectory index is unavailable: {index_path}",
        }
    rows = read_jsonl(index_path)
    referenced = [
        str(row["hidden_states_file"])
        for row in rows
        if row.get("hidden_states_file")
    ]
    missing = [
        relative
        for relative in referenced
        if not (source_run / relative).exists()
    ]
    if missing:
        return {
            "available": False,
            "message": (
                f"mixed-trajectory source is missing {len(missing)} hidden-state "
                f"files, including {missing[0]}"
            ),
        }
    return {
        "available": True,
        "message": f"mixed-trajectory source has {len(referenced)} hidden-state files",
    }
