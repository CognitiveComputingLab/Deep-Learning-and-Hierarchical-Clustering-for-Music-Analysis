#!/usr/bin/env python
"""Train on one Taking Form corpus and test parametric Greedy/exact DP on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_contrast import RobustContrastScaler, boundary_contrast_features
from hierarchical_evaluation import (
    equal_budget_level_scores,
    full_tree_measure,
    reference_lca_depths,
    tree_measure,
)
from hierarchical_parametric import (
    distance_weights_from_scorer,
    estimate_kernel_scale,
    estimate_representation_scales,
    fit_hierarchical_parametric_scorer,
    hierarchical_boundary_targets,
    make_parametric_distance,
    work_balanced_soft_examples,
)
from ordered_affinity import (
    boundary_aware_tree_objective,
    greedy_boundary_aware_tree,
    optimal_boundary_aware_tree,
    pairwise_affinity,
)
from shared_annotation_format import SharedPiece


DISTANCES = ("euclidean", "circle_of_fifths", "key_profile")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path,
                        default=Path("data/shared_annotations_v1_aligned"))
    parser.add_argument("--train-dataset", choices=["mozart", "beethoven"],
                        default="mozart")
    parser.add_argument("--test-dataset", choices=["mozart", "beethoven"],
                        default="beethoven")
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--balance-weights", nargs="+", type=float,
                        default=[0.0, 0.3, 0.6, 1.0, 2.0])
    parser.add_argument("--affinity-context-radius", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--selection-pieces-per-work", type=int, default=1,
                        help="Deterministic work-balanced subset for objective selection")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/parametric_hierarchy_stage"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def dataset_directory(name: str) -> str:
    return {
        "mozart": "Taking_Form_Mozart",
        "beethoven": "Taking_Form_Beethoven",
    }[name]


def load_corpus(args, name: str):
    from greedy_clustering import distance_specs, load_pc_bins

    paths = sorted((args.aligned_root / dataset_directory(name)).glob("*.json"))
    if args.quick:
        paths = paths[:3]
    rows = []
    statuses = []
    for path in paths:
        piece = SharedPiece.read_json(path)
        try:
            if not piece.score_path or not Path(piece.score_path).exists():
                raise FileNotFoundError(piece.score_path or "missing score")
            matrix, bounds = load_pc_bins(Path(piece.score_path), args.bin_size)
            if len(matrix) > args.max_bins:
                raise ValueError(f"{len(matrix)} bins exceeds max_bins={args.max_bins}")
            specs, key = distance_specs(matrix.sum(axis=0))
            features = boundary_contrast_features(
                matrix, specs, contexts=args.contexts, distances=DISTANCES)
            targets = hierarchical_boundary_targets(piece, bounds, len(matrix))
            rows.append({
                "dataset": name, "piece": piece.piece_id, "work": piece.work_id,
                "shared_piece": piece, "matrix": matrix, "bounds": bounds,
                "distance_specs": specs, "features": features, "targets": targets,
                "estimated_key": key,
            })
            statuses.append({
                "dataset": name, "piece": piece.piece_id, "work": piece.work_id,
                "status": "success", "message": "", "n_bins": len(matrix)})
        except Exception as error:
            statuses.append({
                "dataset": name, "piece": piece.piece_id, "work": piece.work_id,
                "status": "failed", "message": repr(error), "n_bins": 0})
    return rows, statuses


def prepare_predictions(items, distance_weights, representation_scales,
                        kernel_scale, scorer, args):
    for item in items:
        distance = make_parametric_distance(
            item, DISTANCES, distance_weights, representation_scales)
        affinity, metadata = pairwise_affinity(
            item["matrix"], distance, scale=kernel_scale,
            context_radius=args.affinity_context_radius)
        item["parametric_distance"] = distance
        item["affinity"] = affinity
        item["affinity_metadata"] = metadata
        item["contrast"] = scorer.score(item["features"])


def fit_model(train, args):
    scaler = RobustContrastScaler.fit([item["features"] for item in train])
    x, targets, sample_weights, audit = work_balanced_soft_examples(train, scaler)
    scorer, history = fit_hierarchical_parametric_scorer(
        scaler, x, targets, sample_weights, epochs=args.epochs,
        learning_rate=args.learning_rate, regularization=args.regularization)
    distance_weights = distance_weights_from_scorer(scorer, DISTANCES)
    representation_scales = estimate_representation_scales(train, DISTANCES)
    kernel_scale = estimate_kernel_scale(
        train, DISTANCES, distance_weights, representation_scales)
    return (scaler, scorer, distance_weights, representation_scales,
            kernel_scale, history, audit)


def select_objective(train, args):
    from greedy_clustering import ClusterNode

    rows = []
    selected_items = []
    for work in sorted({item["work"] for item in train}):
        selected_items.extend(sorted(
            (item for item in train if item["work"] == work),
            key=lambda item: item["piece"])[:args.selection_pieces_per_work])
    for item in selected_items:
        bounds = np.asarray(item["bounds"], dtype=float)
        edges = (np.r_[bounds[0, 0], bounds[:, 1]]
                 if bounds.ndim == 2 else bounds)
        times = (edges[:-1] + edges[1:]) / 2.0
        item["selection_reference"] = (
            times, reference_lca_depths(item["shared_piece"], times))
    best = None
    best_key = None
    for contrast_weight in args.lambdas:
        for balance_weight in args.balance_weights:
            values = []
            reduced = []
            runtimes = []
            for item in selected_items:
                started = time.perf_counter()
                tree, diagnostics = optimal_boundary_aware_tree(
                    item["matrix"], item["bounds"], item["affinity"],
                    item["contrast"], ClusterNode,
                    contrast_weight=contrast_weight,
                    balance_weight=balance_weight, max_bins=args.max_bins)
                metric = full_tree_measure(
                    tree, item["shared_piece"],
                    prepared_reference=item["selection_reference"])
                values.append(metric["t_measure_full"])
                runtimes.append(time.perf_counter() - started)
                if abs(diagnostics.total_objective - tree.subtree_objective) > 1e-8:
                    raise AssertionError("DP reconstruction objective mismatch")
            row = {
                "contrast_weight": contrast_weight,
                "balance_weight": balance_weight,
                "train_t_measure_full": float(np.mean(values)),
                "train_t_measure_reduced": float("nan"),
                "mean_runtime_seconds": float(np.mean(runtimes)),
                "training_pieces": len(values),
                "training_works": len({item["work"] for item in selected_items}),
            }
            rows.append(row)
            key = (row["train_t_measure_full"],
                   -float(balance_weight), -float(contrast_weight))
            if best is None or key > best_key:
                best = (float(contrast_weight), float(balance_weight))
                best_key = key
    return best, pd.DataFrame(rows)


def evaluate_test(test, selected, args):
    from greedy_clustering import ClusterNode

    contrast_weight, balance_weight = selected
    rows = []
    budget_rows = []
    objective_rows = []
    for item in test:
        trees = {}
        runtimes = {}
        started = time.perf_counter()
        trees["greedy"] = greedy_boundary_aware_tree(
            item["matrix"], item["bounds"], item["affinity"], item["contrast"],
            ClusterNode, contrast_weight=contrast_weight,
            balance_weight=balance_weight)
        runtimes["greedy"] = time.perf_counter() - started
        started = time.perf_counter()
        trees["dp"], dp_diagnostics = optimal_boundary_aware_tree(
            item["matrix"], item["bounds"], item["affinity"], item["contrast"],
            ClusterNode, contrast_weight=contrast_weight,
            balance_weight=balance_weight, max_bins=args.max_bins)
        runtimes["dp"] = time.perf_counter() - started
        objectives = {
            search: boundary_aware_tree_objective(
                tree, item["affinity"], item["contrast"],
                contrast_weight=contrast_weight,
                balance_weight=balance_weight)
            for search, tree in trees.items()}
        if (objectives["dp"].total_objective
                < objectives["greedy"].total_objective - 1e-8):
            raise AssertionError("exact DP objective is below Greedy")
        objective_rows.append({
            "dataset": item["dataset"], "piece": item["piece"],
            "work": item["work"],
            "greedy_objective": objectives["greedy"].total_objective,
            "dp_objective": objectives["dp"].total_objective,
            "dp_minus_greedy": (objectives["dp"].total_objective
                                - objectives["greedy"].total_objective),
            "dp_root_split": dp_diagnostics.root_split,
            "evaluated_splits": dp_diagnostics.evaluated_splits,
        })
        for search, tree in trees.items():
            metrics = tree_measure(tree, item["shared_piece"])
            rows.append({
                "dataset": item["dataset"], "piece": item["piece"],
                "work": item["work"], "search": search,
                "contrast_weight": contrast_weight,
                "balance_weight": balance_weight,
                "objective": objectives[search].total_objective,
                "runtime_seconds": runtimes[search],
                **metrics,
            })
            for level in (1, 2, 3):
                budget_rows.append({
                    "dataset": item["dataset"], "piece": item["piece"],
                    "work": item["work"], "search": search,
                    "reference_max_level": level,
                    **equal_budget_level_scores(
                        tree, item["shared_piece"], max_level=level,
                        tolerance_qb=args.tolerance),
                })
    return pd.DataFrame(rows), pd.DataFrame(budget_rows), pd.DataFrame(objective_rows)


def paired_test(work):
    pivot = work.pivot_table(
        index="work", columns="search", values="t_measure_full", aggfunc="mean")
    paired = pivot[["dp", "greedy"]].dropna()
    differences = (paired["dp"] - paired["greedy"]).to_numpy(float)
    rng = np.random.default_rng(20260818)
    signs = rng.choice([-1.0, 1.0], size=(10000, len(differences)))
    observed = abs(float(np.mean(differences)))
    null = np.abs(np.mean(signs * differences, axis=1))
    return {
        "n_works": len(differences),
        "mean_dp_minus_greedy": float(np.mean(differences)),
        "p_value": float((1 + np.sum(null >= observed)) / 10001),
    }


def main():
    args = parse_args()
    if args.train_dataset == args.test_dataset:
        raise SystemExit("Train and test datasets must differ for held-out evaluation")
    if args.quick:
        args.epochs = min(args.epochs, 50)
        args.lambdas = [0.5, 1.0]
        args.balance_weights = [0.0, 0.6]
        args.selection_pieces_per_work = 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, train_status = load_corpus(args, args.train_dataset)
    test, test_status = load_corpus(args, args.test_dataset)
    if not train or not test:
        raise SystemExit("No usable train/test hierarchy pieces")
    pd.DataFrame(train_status + test_status).to_csv(
        args.output_dir / "run_status.csv", index=False)

    (scaler, scorer, distance_weights, representation_scales,
     kernel_scale, history, audit) = fit_model(train, args)
    prepare_predictions(
        train, distance_weights, representation_scales, kernel_scale, scorer, args)
    prepare_predictions(
        test, distance_weights, representation_scales, kernel_scale, scorer, args)
    selected, grid = select_objective(train, args)
    grid.to_csv(args.output_dir / "training_objective_grid.csv", index=False)
    pd.DataFrame(history).to_csv(
        args.output_dir / "parametric_training_history.csv", index=False)
    pd.DataFrame(audit).to_csv(
        args.output_dir / "training_example_audit.csv", index=False)

    per_piece, budget, objectives = evaluate_test(test, selected, args)
    per_piece.to_csv(args.output_dir / "held_out_per_piece.csv", index=False)
    budget.to_csv(args.output_dir / "held_out_equal_budget.csv", index=False)
    objectives.to_csv(args.output_dir / "objective_comparison.csv", index=False)
    work = (per_piece.groupby(["dataset", "work", "search"], as_index=False)
            .agg(t_measure_full=("t_measure_full", "mean"),
                 t_measure_reduced=("t_measure_reduced", "mean"),
                 t_precision_full=("t_precision_full", "mean"),
                 t_recall_full=("t_recall_full", "mean"),
                 lca_spearman=("lca_spearman", "mean"),
                 objective=("objective", "mean"),
                 runtime_seconds=("runtime_seconds", "mean")))
    work.to_csv(args.output_dir / "held_out_per_work.csv", index=False)
    summary = (work.groupby(["dataset", "search"], as_index=False)
               .agg(works=("work", "nunique"),
                    t_measure_full=("t_measure_full", "mean"),
                    t_measure_reduced=("t_measure_reduced", "mean"),
                    t_precision_full=("t_precision_full", "mean"),
                    t_recall_full=("t_recall_full", "mean"),
                    lca_spearman=("lca_spearman", "mean"),
                    objective=("objective", "mean"),
                    runtime_seconds=("runtime_seconds", "mean")))
    summary.to_csv(args.output_dir / "held_out_summary.csv", index=False)
    pd.DataFrame([paired_test(work)]).to_csv(
        args.output_dir / "paired_dp_vs_greedy.csv", index=False)

    parameters = {
        "training_dataset": args.train_dataset,
        "held_out_test_dataset": args.test_dataset,
        "test_labels_used_during_training_or_selection": False,
        "distance_names": list(DISTANCES),
        "feature_weights": scorer.as_dict(),
        "distance_weights": dict(zip(DISTANCES, map(float, distance_weights))),
        "representation_scales": dict(zip(
            DISTANCES, map(float, representation_scales))),
        "kernel_scale": kernel_scale,
        "selected_contrast_weight": selected[0],
        "selected_balance_weight": selected[1],
        "selection_metric": "training-corpus mean full T-measure for exact DP",
        "optimality_scope": (
            "fixed leaves, temporal contiguity, and the selected additive "
            "normalized-affinity plus hierarchical-boundary objective"),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()},
    }
    (args.output_dir / "model_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(pd.DataFrame([paired_test(work)]).to_string(index=False))


if __name__ == "__main__":
    main()
