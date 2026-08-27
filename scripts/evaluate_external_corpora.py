#!/usr/bin/env python
"""Zero-shot evaluation on frozen Taking Form and Algomus targets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from annotated_piece import notes_to_pc_bins, select_bin_size
from corpus_loaders import load_fugue_piece, load_taking_form_piece
from external_evaluation import (
    best_tree_span_iou, boundary_ap_from_times, span_scores,
    spans_from_boundaries,
)
from greedy_evaluation import (
    boundary_prominence, boundary_prominence_scores, boundary_scores,
    collect_prominent_splits, tree_shape_diagnostics,
)
from neural_clustering import (
    BoundaryDistanceModel, MergePolicy, NeuralEmbeddingDistance,
    build_pitch_class_encoder,
)
from train_deep_clustering import build_tree


METHODS = [
    "key_profile_affinity_greedy", "key_profile_dp",
    "siamese_affinity_greedy", "siamese_dp", "rl_frozen", "rl_joint",
]


def load_frozen_models(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    architecture = saved.get("encoder_config", {}).get(
        "name", "circular_harmonic_cnn")
    metric = BoundaryDistanceModel(build_pitch_class_encoder(architecture)).to(device)
    metric.load_state_dict(saved["metric"])
    metric.eval()
    rl_models = {}
    for variant in ("rl_frozen", "rl_joint"):
        encoder = build_pitch_class_encoder(architecture).to(device)
        encoder.load_state_dict(saved[f"{variant}_encoder"])
        policy = MergePolicy().to(device)
        policy.load_state_dict(saved[f"{variant}_policy"])
        encoder.eval(); policy.eval()
        rl_models[variant] = (encoder, policy)
    return (int(saved["seed"]), NeuralEmbeddingDistance(metric.encoder, device),
            rl_models, saved.get("external_budget_ratio"))


def _tree_spans(root):
    spans = []
    def walk(node):
        children = list(getattr(node, "children", []) or [])
        if children:
            spans.append((float(node.start), float(node.end)))
            for child in children:
                walk(child)
    walk(root)
    return spans


def evaluate_piece(piece, seed, distance, rl_models, ratio, device):
    bin_size = select_bin_size(piece.duration_qb)
    matrix, bounds = notes_to_pc_bins(piece.inference_view(), bin_size)
    item = {"matrix": matrix, "bounds": bounds}
    candidate_count = max(1, len(matrix) - 1)
    budget = min(candidate_count, max(1, int(round(ratio * candidate_count))))
    rows, span_rows, diagnostics = [], [], []
    bundle = piece.annotations
    levels = (sorted({boundary.level for boundary in bundle.boundaries})
              if piece.dataset == "taking_form" else [1])
    for method in METHODS:
        tree, runtime = build_tree(
            method, item, distance, rl_models, torch.device(device))
        predicted = collect_prominent_splits(tree, budget)
        prominence = boundary_prominence_scores(tree, bounds)
        edges = np.r_[bounds[0][0], np.asarray(bounds)[:, 1]]
        for level in levels:
            references = [value.time_qb for value in bundle.boundaries
                          if value.level == level]
            score = boundary_scores(predicted, references, bin_size)
            rows.append({
                "dataset": piece.dataset, "work": piece.work_id,
                "piece": piece.piece_id, "seed": seed, "method": method,
                "level": level, "bin_size_qb": bin_size,
                "n_bins": len(matrix), "budget_ratio": ratio, "budget": budget,
                "boundary_ap": boundary_ap_from_times(
                    edges[1:-1], prominence, references, bin_size),
                "runtime_seconds": runtime,
                **{name: score[name] for name in
                   ("tp", "fp", "fn", "precision", "recall", "f1")},
            })
            if piece.dataset == "taking_form":
                reference_spans = [
                    (value.start_qb, value.end_qb) for value in bundle.spans
                    if value.level == level]
                span_rows.append({
                    "dataset": piece.dataset, "work": piece.work_id,
                    "piece": piece.piece_id, "seed": seed, "method": method,
                    "level": level, "bin_size_qb": bin_size,
                    **span_scores(
                        spans_from_boundaries(predicted, piece.duration_qb),
                        reference_spans, bin_size),
                })
        pedal = [(value.start_qb, value.end_qb) for value in bundle.intervals
                 if value.kind == "pedal"]
        diagnostics.append({
            "dataset": piece.dataset, "work": piece.work_id,
            "piece": piece.piece_id, "seed": seed, "method": method,
            "pedal_best_tree_span_iou": (
                best_tree_span_iou(_tree_spans(tree), pedal) if pedal else np.nan),
            "reference_pedal_count": len(pedal),
            **tree_shape_diagnostics(tree),
        })
    return rows, span_rows, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("results/dissertation_main/data_manifest.csv"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("results/dissertation_main/deep"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/dissertation_main/external"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(args.checkpoint_dir.glob("checkpoint_seed_*.pt"))
    if not checkpoints:
        raise SystemExit(f"No frozen checkpoints found in {args.checkpoint_dir}")
    manifest = pd.read_csv(args.manifest).fillna("")
    selected = manifest[
        manifest.dataset.isin(["taking_form", "algomus_fugue"]) &
        manifest.status.eq("included")].copy()
    if args.quick:
        selected = selected.groupby("dataset", as_index=False).head(1)
        checkpoints = checkpoints[:1]
    if selected.empty:
        raise SystemExit(
            "No validated external score/annotation pairs; inspect exclusion_audit.csv")
    access = [{"sequence": 1, "event": "all_abc_checkpoints_loaded",
               "scope": "model", "details": str(len(checkpoints))}]
    loaded = [load_frozen_models(path, args.device) for path in checkpoints]
    if any(ratio is None for _, _, _, ratio in loaded):
        raise SystemExit(
            "Checkpoint predates frozen external-budget selection; retrain it")
    access.append({"sequence": 2, "event": "target_annotations_loaded",
                   "scope": "external_test", "details": str(len(selected))})
    rows = []; spans = []; diagnostics = []; status = []
    for record in selected.itertuples(index=False):
        try:
            if record.dataset == "taking_form":
                piece = load_taking_form_piece(
                    record.annotation_path, record.score_path,
                    work_id=str(record.work_id), piece_id=record.piece_id)
            else:
                piece = load_fugue_piece(
                    record.annotation_path, record.score_path,
                    work_id=str(record.work_id), piece_id=record.piece_id)
            for seed, distance, rl_models, ratio in loaded:
                current, current_spans, current_diagnostics = evaluate_piece(
                    piece, seed, distance, rl_models, ratio, args.device)
                rows.extend(current); spans.extend(current_spans)
                diagnostics.extend(current_diagnostics)
            status.append({"dataset": record.dataset, "piece": record.piece_id,
                           "status": "success", "reason": ""})
        except Exception as error:
            status.append({"dataset": record.dataset, "piece": record.piece_id,
                           "status": "excluded", "reason": repr(error)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        pd.DataFrame(status).to_csv(args.output_dir / "exclusion_audit.csv", index=False)
        raise SystemExit("All external pieces failed validation")
    work = frame.groupby(
        ["dataset", "seed", "method", "level", "work"], as_index=False).agg(
            precision=("precision", "mean"), recall=("recall", "mean"),
            f1=("f1", "mean"), boundary_ap=("boundary_ap", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            n_pieces=("piece", "nunique"))
    summary = work.groupby(
        ["dataset", "seed", "method", "level"], as_index=False).agg(
            n_works=("work", "nunique"), work_macro_precision=("precision", "mean"),
            work_macro_recall=("recall", "mean"), work_macro_f1=("f1", "mean"),
            work_macro_boundary_ap=("boundary_ap", "mean"))
    frame.to_csv(args.output_dir / "external_per_piece.csv", index=False)
    work.to_csv(args.output_dir / "external_per_work.csv", index=False)
    summary.to_csv(args.output_dir / "external_summary.csv", index=False)
    pd.DataFrame(spans).to_csv(args.output_dir / "taking_form_span_per_piece.csv",
                               index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "tree_diagnostics.csv",
                                     index=False)
    pd.DataFrame(status).to_csv(args.output_dir / "exclusion_audit.csv", index=False)
    pd.DataFrame(access).to_csv(args.output_dir / "access_audit.csv", index=False)
    (args.output_dir / "config.json").write_text(json.dumps({
        "protocol": "frozen_abc_zero_shot_external_validation",
        "methods": METHODS, "target_training": False,
        "budget_selection": "shared ratio selected using ABC validation only",
        "tolerance": "one selected temporal bin",
        "pedal_metric": "mean best IoU with any predicted internal tree span",
        "subjects_and_countersubjects": "excluded because overlapping events are not a tree",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
