#!/usr/bin/env python
"""Nested work-blocked evaluation of boundary-aware Greedy and exact DP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_contrast import (
    DEFAULT_CONTEXTS,
    BoundaryContrastScorer,
    RobustContrastScaler,
    boundary_contrast_features,
    fit_supervised_scorer,
    make_supervised_examples,
)
from dp_clustering import assert_valid_ordered_binary_tree
from greedy_clustering import ClusterNode, distance_specs, load_pc_bins
from greedy_evaluation import (
    bootstrap_summary,
    boundary_salience,
    boundary_scores,
    collect_salient_splits,
    dcml_localkey_segments,
    micro_summary,
    paired_permutation_tests,
    tree_shape_diagnostics,
)
from ordered_affinity import (
    boundary_aware_tree_objective,
    greedy_boundary_aware_tree,
    optimal_boundary_aware_tree,
    pairwise_affinity,
)


MODEL_MODES = ("similarity_only", "unsupervised_contrast", "supervised_contrast")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/boundary_aware_stage"))
    parser.add_argument("--piece", action="append")
    parser.add_argument("--outer-work", action="append",
                        help="evaluate only these held-out works; training still uses all others")
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--affinity-method", default="key_profile")
    parser.add_argument("--affinity-context-radius", type=int, default=0)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[0, 0.1, 0.25, 0.5, 0.75, 1])
    parser.add_argument("--balance-weights", nargs="+", type=float,
                        default=[0, 0.1, 0.25, 0.5, 0.6, 1])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[value / 50 for value in range(51)])
    parser.add_argument("--boundary-budgets", nargs="+", type=int,
                        default=[3, 5, 8, 10, 12, 15, 20])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[2, 4, 8, 12, 24])
    parser.add_argument("--selection-tolerance", type=float, default=8.0)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--tie-break", choices=["earliest", "midpoint", "latest"],
                        default="midpoint")
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def work_id(piece):
    return piece.rsplit("_", 1)[0] if "_" in piece else piece


def discover_pairs(root, selected=None):
    notes = {path.name.replace(".notes.tsv", ""): path
             for path in (root / "notes").glob("*.notes.tsv")}
    harmonies = {path.name.replace(".harmonies.tsv", ""): path
                 for path in (root / "harmonies").glob("*.harmonies.tsv")}
    stems = sorted(set(notes) & set(harmonies))
    if selected:
        stems = [stem for stem in stems if stem in set(selected)]
    return [(stem, notes[stem], harmonies[stem]) for stem in stems]


def load_corpus(pairs, args):
    cache = {}
    statuses = []
    affinity_rows = []
    for number, (piece, notes, harmonies) in enumerate(pairs, 1):
        try:
            segments, total = dcml_localkey_segments(harmonies)
            matrix, bounds = load_pc_bins(notes, args.bin_size)
            if len(matrix) > args.max_bins:
                raise ValueError(f"{len(matrix)} bins exceeds max_bins={args.max_bins}")
            specs, estimated_key = distance_specs(matrix.sum(axis=0))
            if args.affinity_method not in specs:
                raise KeyError(f"unknown affinity method: {args.affinity_method}")
            features = boundary_contrast_features(matrix, specs, contexts=args.contexts)
            affinity, affinity_info = pairwise_affinity(
                matrix, specs[args.affinity_method],
                context_radius=args.affinity_context_radius)
            reference = [float(end) for _, end, _ in segments[:-1]]
            cache[piece] = {
                "piece": piece, "work": work_id(piece), "matrix": matrix,
                "bounds": bounds, "segments": segments, "total": total,
                "reference": reference, "features": features, "affinity": affinity,
                "estimated_key": estimated_key,
            }
            affinity_rows.append({"piece": piece, "work": work_id(piece), **affinity_info})
            statuses.append({"piece": piece, "work": work_id(piece),
                             "status": "success", "message": ""})
            print(f"[load {number}/{len(pairs)}] {piece}: {len(matrix)} bins", flush=True)
        except Exception as error:
            state = "skipped" if "exceeds max_bins=" in str(error) else "failed"
            statuses.append({"piece": piece, "work": work_id(piece),
                             "status": state, "message": repr(error)})
            print(f"[load {number}/{len(pairs)}] {piece}: {state}: {error}",
                  file=sys.stderr, flush=True)
    return cache, statuses, affinity_rows


def items_for_works(cache, works):
    works = set(works)
    return [item for _, item in sorted(cache.items()) if item["work"] in works]


def stable_inner_folds(works, folds, seed, outer_work):
    works = np.asarray(sorted(works), dtype=object)
    if len(works) < 2:
        raise ValueError("nested evaluation requires at least two training works")
    digest = hashlib.sha256(f"{seed}|{outer_work}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    ordered = works[rng.permutation(len(works))]
    return [list(chunk) for chunk in np.array_split(ordered, min(folds, len(works)))
            if len(chunk)]


def scorer_training_items(items):
    return [{
        "work": item["work"], "piece": item["piece"], "features": item["features"],
        "bounds": item["bounds"], "reference": item["reference"],
    } for item in items]


def fit_contrast_models(items, args):
    scaler = RobustContrastScaler.fit([item["features"] for item in items])
    unsupervised = BoundaryContrastScorer.unsupervised(scaler)
    calibrated, labels, sample_weights, audit = make_supervised_examples(
        scorer_training_items(items), scaler)
    supervised, history = fit_supervised_scorer(
        scaler, calibrated, labels, sample_weights, epochs=args.epochs,
        learning_rate=args.learning_rate, regularization=args.regularization)
    return {
        "unsupervised_contrast": unsupervised,
        "supervised_contrast": supervised,
    }, audit, history


def contrast_for_item(mode, scorer, item):
    if mode == "similarity_only":
        return np.ones(max(0, len(item["matrix"]) - 1), dtype=float)
    return scorer.score(item["features"])


def build_trees(item, contrast, contrast_weight, balance_weight, args):
    matrix, bounds, affinity = item["matrix"], item["bounds"], item["affinity"]
    started = time.perf_counter()
    greedy = greedy_boundary_aware_tree(
        matrix, bounds, affinity, contrast, ClusterNode,
        contrast_weight=contrast_weight,balance_weight=balance_weight)
    greedy_seconds = time.perf_counter() - started
    greedy_info = boundary_aware_tree_objective(
        greedy, affinity, contrast, contrast_weight=contrast_weight,
        balance_weight=balance_weight)
    dp, dp_info = optimal_boundary_aware_tree(
        matrix, bounds, affinity, contrast, ClusterNode,
        contrast_weight=contrast_weight,balance_weight=balance_weight,
        tie_break=args.tie_break,
        max_bins=args.max_bins)
    assert_valid_ordered_binary_tree(greedy, bounds)
    assert_valid_ordered_binary_tree(dp, bounds)
    if dp_info.total_objective < greedy_info.total_objective - 1e-8:
        raise AssertionError("exact DP boundary-aware objective is below greedy")
    return [
        ("greedy", greedy, greedy_info, greedy_seconds),
        ("dp", dp, dp_info, dp_info.elapsed_seconds),
    ]


def candidate_rows(items, mode, scorer, lambdas, balance_weights, args, fold_fields):
    rows = []
    diagnostics = []
    for item in items:
        contrast = contrast_for_item(mode, scorer, item)
        for contrast_weight in lambdas:
            for balance_weight in balance_weights:
              for search, tree, info, seconds in build_trees(
                      item, contrast, contrast_weight, balance_weight, args):
                common = {
                    **fold_fields, "piece": item["piece"], "work": item["work"],
                    "model": mode, "search": search,
                    "contrast_weight": float(contrast_weight),
                    "balance_weight": float(balance_weight),
                }
                for threshold in args.thresholds:
                    predicted = collect_salient_splits(
                        tree, contrast, threshold=threshold)
                    score = boundary_scores(
                        predicted, item["reference"], args.selection_tolerance)
                    rows.append({
                        **common, "selection_mode": "threshold",
                        "candidate_value": float(threshold),
                        **{key: score[key] for key in
                           ["tp", "fp", "fn", "precision", "recall", "f1",
                            "predicted_boundary_count", "gt_boundary_count"]},
                    })
                for budget in args.boundary_budgets:
                    predicted = collect_salient_splits(tree, contrast, budget=budget)
                    score = boundary_scores(
                        predicted, item["reference"], args.selection_tolerance)
                    rows.append({
                        **common, "selection_mode": "fixed_budget",
                        "candidate_value": int(budget),
                        **{key: score[key] for key in
                           ["tp", "fp", "fn", "precision", "recall", "f1",
                            "predicted_boundary_count", "gt_boundary_count"]},
                    })
                diagnostics.append({
                    **common, "objective": info.total_objective,
                    "normalized_affinity_revenue": info.normalized_affinity_revenue,
                    "boundary_reward": info.boundary_reward,
                    "balance_penalty": info.balance_penalty,
                    "runtime_seconds": seconds,
                    "evaluated_splits": info.evaluated_splits,
                    "root_split": info.root_split,
                    "tie_count": info.tie_count,
                    **tree_shape_diagnostics(tree),
                })
    return rows, diagnostics


def select_candidate(frame, mode, selection_mode):
    subset = frame[(frame.model == mode) & (frame.selection_mode == selection_mode)]
    if subset.empty:
        raise ValueError(f"no validation candidates for {mode}/{selection_mode}")
    work = (subset.groupby(
        ["work", "search", "contrast_weight", "balance_weight",
         "candidate_value"], as_index=False)
        [["precision", "recall", "f1"]].mean())
    candidates = (work.groupby(
        ["contrast_weight", "balance_weight", "candidate_value"], as_index=False)
        [["precision", "recall", "f1"]].mean())
    candidate_ascending = selection_mode == "fixed_budget"
    candidates = candidates.sort_values(
        ["f1", "recall", "precision", "candidate_value",
         "balance_weight", "contrast_weight"],
        ascending=[False, False, False, candidate_ascending, True, True],
        kind="stable")
    return candidates.iloc[0].to_dict(), candidates


def model_parameter_rows(models, outer_work, stage, inner_fold=None):
    rows = []
    for mode, scorer in models.items():
        for index, name in enumerate(scorer.scaler.names):
            rows.append({
                "outer_work": outer_work, "stage": stage, "inner_fold": inner_fold,
                "model": mode, "feature": name,
                "weight": float(scorer.weights[index]),
                "training_median": float(scorer.scaler.medians[index]),
                "training_mad": float(scorer.scaler.mads[index]),
                "training_scale": float(scorer.scaler.scales[index]),
                "intercept": float(scorer.intercept), "gain": float(scorer.gain),
            })
    return rows


def evaluate_outer(items, models, selections, args, outer_work):
    rows = []
    diagnostics = []
    salience_output = []
    tree_cache = {}
    diagnostic_keys = set()
    for mode in MODEL_MODES:
        scorer = models.get(mode)
        contrast_by_piece = {
            item["piece"]: contrast_for_item(mode, scorer, item) for item in items
        }
        for selection_mode in ("threshold", "fixed_budget"):
            choice = selections[(mode, selection_mode)]
            contrast_weight = float(choice["contrast_weight"])
            balance_weight = float(choice["balance_weight"])
            candidate = float(choice["candidate_value"])
            for item in items:
                contrast = contrast_by_piece[item["piece"]]
                cache_key = (item["piece"], mode, contrast_weight, balance_weight)
                if cache_key not in tree_cache:
                    tree_cache[cache_key] = build_trees(
                        item, contrast, contrast_weight, balance_weight, args)
                for search, tree, info, seconds in tree_cache[cache_key]:
                    diagnostic_key = cache_key + (search,)
                    if diagnostic_key not in diagnostic_keys:
                        diagnostic_keys.add(diagnostic_key)
                        diagnostics.append({
                            "outer_work": outer_work, "piece": item["piece"],
                            "work": item["work"], "model": mode, "search": search,
                            "contrast_weight": contrast_weight,
                            "balance_weight": balance_weight,
                            "objective": info.total_objective,
                            "normalized_affinity_revenue": info.normalized_affinity_revenue,
                            "boundary_reward": info.boundary_reward,
                            "balance_penalty": info.balance_penalty,
                            "runtime_seconds": seconds,
                            "evaluated_splits": info.evaluated_splits,
                            "root_split": info.root_split, "tie_count": info.tie_count,
                            **tree_shape_diagnostics(tree),
                        })
                        for boundary in boundary_salience(tree, contrast):
                            salience_output.append({
                                "outer_work": outer_work, "piece": item["piece"],
                                "work": item["work"], "model": mode, "search": search,
                                "contrast_weight": contrast_weight,
                                "balance_weight": balance_weight, **boundary,
                            })
                    if selection_mode == "threshold":
                        predicted = collect_salient_splits(
                            tree, contrast, threshold=candidate)
                    else:
                        predicted = collect_salient_splits(
                            tree, contrast, budget=int(candidate))
                    for tolerance in args.tolerances:
                        score = boundary_scores(predicted, item["reference"], tolerance)
                        rows.append({
                            "outer_work": outer_work, "piece": item["piece"],
                            "work": item["work"], "model": mode, "search": search,
                            "selection_mode": selection_mode,
                            "selected_value": candidate,
                            "contrast_weight": contrast_weight,
                            "balance_weight": balance_weight,
                            "tolerance_qb": float(tolerance), "oracle": False,
                            **{key: score[key] for key in
                               ["tp", "fp", "fn", "precision", "recall", "f1",
                                "predicted_boundary_count", "gt_boundary_count",
                                "mean_match_error_qb"]},
                        })

        # Oracle count is diagnostic only and uses the fixed-budget-selected tree.
        choice = selections[(mode, "fixed_budget")]
        contrast_weight = float(choice["contrast_weight"])
        balance_weight = float(choice["balance_weight"])
        for item in items:
            contrast = contrast_by_piece[item["piece"]]
            cache_key = (item["piece"], mode, contrast_weight, balance_weight)
            if cache_key not in tree_cache:
                tree_cache[cache_key] = build_trees(
                    item, contrast, contrast_weight, balance_weight, args)
            for search, tree, _, _ in tree_cache[cache_key]:
                predicted = collect_salient_splits(
                    tree, contrast, budget=len(item["reference"]))
                for tolerance in args.tolerances:
                    score = boundary_scores(predicted, item["reference"], tolerance)
                    rows.append({
                        "outer_work": outer_work, "piece": item["piece"],
                        "work": item["work"], "model": mode, "search": search,
                        "selection_mode": "oracle_budget",
                        "selected_value": len(item["reference"]),
                        "contrast_weight": contrast_weight,
                        "balance_weight": balance_weight,
                        "tolerance_qb": float(tolerance), "oracle": True,
                        **{key: score[key] for key in
                           ["tp", "fp", "fn", "precision", "recall", "f1",
                            "predicted_boundary_count", "gt_boundary_count",
                            "mean_match_error_qb"]},
                    })
    return rows, diagnostics, salience_output


def plot_outputs(results, candidates, comparison, output_dir):
    selected = results[~results.oracle].copy()
    if not selected.empty:
        tolerance = selected.groupby(
            ["model", "search", "selection_mode", "tolerance_qb"],
            as_index=False).f1.mean()
        figure, axis = plt.subplots(figsize=(8, 5))
        for keys, group in tolerance.groupby(["model", "search", "selection_mode"]):
            axis.plot(group.tolerance_qb, group.f1, marker="o",
                      label="/".join(map(str, keys)))
        axis.set(xlabel="Tolerance (quarterbeats)", ylabel="Mean movement F1",
                 title="Held-out Boundary F1 vs tolerance")
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / "f1_vs_tolerance.png", dpi=160)
        plt.close(figure)

        pr = selected.groupby(
            ["model", "search", "selection_mode", "tolerance_qb"],
            as_index=False)[["precision", "recall"]].mean()
        figure, axis = plt.subplots(figsize=(7, 5))
        for keys, group in pr.groupby(["model", "search", "selection_mode"]):
            axis.plot(group.recall, group.precision, marker="o", linestyle="none",
                      label="/".join(map(str, keys)))
        axis.set(xlabel="Recall", ylabel="Precision",
                 title="Held-out tolerance operating points")
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / "precision_recall_operating_points.png", dpi=160)
        plt.close(figure)

    for selection_mode, filename, xlabel in (
        ("threshold", "f1_vs_threshold.png", "Salience threshold"),
        ("fixed_budget", "f1_vs_budget.png", "Boundary budget"),
    ):
        subset = candidates[candidates.selection_mode == selection_mode].copy()
        if subset.empty:
            continue
        # Hyperparameter curves are inner-validation diagnostics. For each
        # candidate value retain the best lambda within each outer fold.
        subset = (subset.sort_values("f1", ascending=False, kind="stable")
                  .drop_duplicates(["outer_work", "model", "candidate_value"]))
        curve = subset.groupby(["model", "candidate_value"], as_index=False).f1.mean()
        figure, axis = plt.subplots(figsize=(7, 4))
        for model, group in curve.groupby("model"):
            axis.plot(group.candidate_value, group.f1, marker="o", label=model)
        axis.set(xlabel=xlabel, ylabel="Inner-validation macro F1",
                 title=f"Selection sensitivity: {xlabel}")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    if not comparison.empty and "objective_gap" in comparison:
        figure, axis = plt.subplots(figsize=(7, 4))
        groups = [group.objective_gap.dropna().to_numpy()
                  for _, group in comparison.groupby("model")]
        labels = [name for name, _ in comparison.groupby("model")]
        if groups:
            axis.boxplot(groups, labels=labels)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set(ylabel="DP - Greedy objective",
                     title="Exact-search objective advantage")
            axis.tick_params(axis="x", rotation=20)
            figure.tight_layout()
            figure.savefig(output_dir / "objective_gap.png", dpi=160)
        plt.close(figure)


def main():
    args = parse_args()
    if args.quick:
        args.lambdas = [0.0, 0.5]
        args.balance_weights = [0.0, 1.0]
        args.thresholds = [0.3, 0.5, 0.7]
        args.boundary_budgets = [5, 10]
        args.tolerances = [args.selection_tolerance]
        args.inner_folds = 2
        args.epochs = min(args.epochs, 25)
    if any(not 0 <= value <= 1 for value in args.lambdas + args.thresholds):
        raise SystemExit("lambdas and thresholds must lie in [0,1]")
    if any(value < 0 for value in args.balance_weights):
        raise SystemExit("balance weights must be non-negative")
    if args.selection_tolerance not in args.tolerances:
        args.tolerances = sorted(set(args.tolerances + [args.selection_tolerance]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data_root, args.piece)
    if args.quick and not args.piece:
        grouped = {}
        for pair in pairs:
            grouped.setdefault(work_id(pair[0]), []).append(pair)
        selected_works = sorted(grouped)[:6]
        pairs = [sorted(grouped[work])[0] for work in selected_works]
    if not pairs:
        raise SystemExit("No paired ABC notes/harmonies files found")
    cache, statuses, affinity_rows = load_corpus(pairs, args)
    pd.DataFrame(statuses).to_csv(args.output_dir / "run_status.csv", index=False)
    pd.DataFrame(affinity_rows).to_csv(
        args.output_dir / "affinity_metadata.csv", index=False)
    if not cache:
        raise SystemExit("All pieces failed; inspect run_status.csv")

    all_works = sorted({item["work"] for item in cache.values()})
    requested_outer = set(args.outer_work or all_works)
    outer_works = [work for work in all_works if work in requested_outer]
    if args.quick and not args.outer_work:
        outer_works = outer_works[:2]
    missing_outer = sorted(requested_outer - set(all_works))
    if missing_outer:
        raise SystemExit(f"Unknown outer works: {missing_outer}")
    if len(all_works) < 4:
        raise SystemExit("At least four complete works are required for nested evaluation")

    assignment_rows = []
    inner_rows_all = []
    inner_diagnostics_all = []
    candidate_summary_all = []
    selection_rows = []
    parameter_rows = []
    training_audit_rows = []
    history_rows = []
    outer_rows = []
    outer_diagnostics = []
    salience_rows = []

    for outer_number, outer_work in enumerate(outer_works, 1):
        train_works = [work for work in all_works if work != outer_work]
        folds = stable_inner_folds(
            train_works, args.inner_folds, args.seed, outer_work)
        fold_frames = []
        for fold_index, validation_works in enumerate(folds):
            inner_train_works = [work for work in train_works
                                 if work not in set(validation_works)]
            inner_train = items_for_works(cache, inner_train_works)
            inner_validation = items_for_works(cache, validation_works)
            models, audit, history = fit_contrast_models(inner_train, args)
            parameter_rows.extend(model_parameter_rows(
                models, outer_work, "inner", fold_index))
            training_audit_rows.extend({
                "outer_work": outer_work, "stage": "inner",
                "inner_fold": fold_index, **row,
            } for row in audit)
            history_rows.extend({
                "outer_work": outer_work, "stage": "inner",
                "inner_fold": fold_index, **row,
            } for row in history)
            for work in inner_train_works:
                assignment_rows.append({
                    "outer_work": outer_work, "inner_fold": fold_index,
                    "work": work, "split": "inner_train"})
            for work in validation_works:
                assignment_rows.append({
                    "outer_work": outer_work, "inner_fold": fold_index,
                    "work": work, "split": "inner_validation"})
            for mode in MODEL_MODES:
                scorer = models.get(mode)
                lambdas = [0.0] if mode == "similarity_only" else args.lambdas
                balance_weights = args.balance_weights
                rows, diagnostics = candidate_rows(
                    inner_validation, mode, scorer, lambdas, balance_weights, args,
                    {"outer_work": outer_work, "inner_fold": fold_index})
                fold_frames.extend(rows)
                inner_diagnostics_all.extend(diagnostics)
        inner_frame = pd.DataFrame(fold_frames)
        inner_rows_all.extend(fold_frames)
        selections = {}
        for mode in MODEL_MODES:
            for selection_mode in ("threshold", "fixed_budget"):
                chosen, candidates = select_candidate(
                    inner_frame, mode, selection_mode)
                selections[(mode, selection_mode)] = chosen
                candidates.insert(0, "selection_mode", selection_mode)
                candidates.insert(0, "model", mode)
                candidates.insert(0, "outer_work", outer_work)
                candidates["chosen"] = (
                    np.isclose(candidates.contrast_weight, chosen["contrast_weight"])
                    & np.isclose(candidates.balance_weight, chosen["balance_weight"])
                    & np.isclose(candidates.candidate_value, chosen["candidate_value"]))
                candidate_summary_all.extend(candidates.to_dict("records"))
                selection_rows.append({
                    "outer_work": outer_work, "model": mode,
                    "selection_mode": selection_mode,
                    "selected_contrast_weight": chosen["contrast_weight"],
                    "selected_balance_weight": chosen["balance_weight"],
                    "selected_value": chosen["candidate_value"],
                    "inner_f1": chosen["f1"], "inner_recall": chosen["recall"],
                    "inner_precision": chosen["precision"],
                })

        outer_train = items_for_works(cache, train_works)
        refit_models, audit, history = fit_contrast_models(outer_train, args)
        parameter_rows.extend(model_parameter_rows(
            refit_models, outer_work, "outer_refit"))
        training_audit_rows.extend({
            "outer_work": outer_work, "stage": "outer_refit",
            "inner_fold": None, **row,
        } for row in audit)
        history_rows.extend({
            "outer_work": outer_work, "stage": "outer_refit",
            "inner_fold": None, **row,
        } for row in history)
        assignment_rows.extend([
            {"outer_work": outer_work, "inner_fold": None,
             "work": work, "split": "outer_train"} for work in train_works
        ])
        assignment_rows.append({
            "outer_work": outer_work, "inner_fold": None,
            "work": outer_work, "split": "outer_test"})
        tested = items_for_works(cache, [outer_work])
        rows, diagnostics, salience = evaluate_outer(
            tested, refit_models, selections, args, outer_work)
        outer_rows.extend(rows)
        outer_diagnostics.extend(diagnostics)
        salience_rows.extend(salience)
        print(f"[outer {outer_number}/{len(outer_works)}] {outer_work}: success",
              flush=True)

    inner_frame = pd.DataFrame(inner_rows_all)
    candidate_frame = pd.DataFrame(candidate_summary_all)
    results = pd.DataFrame(outer_rows)
    diagnostics = pd.DataFrame(outer_diagnostics)
    pd.DataFrame(assignment_rows).drop_duplicates().to_csv(
        args.output_dir / "fold_assignments.csv", index=False)
    inner_frame.to_csv(args.output_dir / "inner_validation_per_piece.csv", index=False)
    pd.DataFrame(inner_diagnostics_all).to_csv(
        args.output_dir / "inner_tree_diagnostics.csv", index=False)
    candidate_frame.to_csv(args.output_dir / "inner_candidate_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        args.output_dir / "outer_selections.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        args.output_dir / "learned_contrast_parameters.csv", index=False)
    pd.DataFrame(training_audit_rows).to_csv(
        args.output_dir / "training_example_audit.csv", index=False)
    pd.DataFrame(history_rows).to_csv(
        args.output_dir / "training_history.csv", index=False)
    results.to_csv(args.output_dir / "held_out_per_piece.csv", index=False)
    diagnostics.to_csv(args.output_dir / "tree_diagnostics.csv", index=False)
    pd.DataFrame(salience_rows).to_csv(
        args.output_dir / "boundary_salience.csv", index=False)

    group_columns = [
        "model", "search", "selection_mode", "tolerance_qb", "oracle"]
    work_results = (results.groupby(["work"] + group_columns, as_index=False)
                    [["tp", "fp", "fn", "precision", "recall", "f1"]].mean())
    work_results.to_csv(args.output_dir / "held_out_per_work.csv", index=False)
    bootstrap_summary(
        work_results, group_columns, samples=2000, seed=args.seed).to_csv(
            args.output_dir / "work_macro_summary.csv", index=False)
    micro_summary(results, group_columns).to_csv(
        args.output_dir / "micro_summary.csv", index=False)

    comparison = diagnostics.pivot_table(
        index=["outer_work", "piece", "work", "model",
               "contrast_weight", "balance_weight"],
        columns="search", values="objective", aggfunc="first").reset_index()
    if {"greedy", "dp"} <= set(comparison):
        comparison["objective_gap"] = comparison.dp - comparison.greedy
        comparison["relative_objective_gap"] = (
            comparison.objective_gap / comparison.dp.abs().replace(0, np.nan))
        if (comparison.objective_gap < -1e-8).any():
            raise AssertionError("negative exact DP objective gap in saved results")
    comparison.to_csv(args.output_dir / "objective_comparison.csv", index=False)

    primary = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)
    ].copy()
    primary["method"] = primary.model + "_" + primary.search
    paired_permutation_tests(
        primary[["work", "method", "f1"]], samples=10000,
        seed=args.seed, unit_column="work").to_csv(
            args.output_dir / "primary_paired_tests.csv", index=False)
    plot_outputs(results, candidate_frame, comparison, args.output_dir)

    metadata = {
        "primary_metric": "work-level macro DCML local-key Boundary F1",
        "primary_selection": "nested-validation salience threshold",
        "fair_comparison": "nested-validation fixed boundary budget",
        "objective": (
            "(1-lambda)*normalized ordered affinity revenue + "
            "lambda*mean span-weighted local boundary contrast - "
            "beta*mean squared child-size imbalance"),
        "dp_optimality": (
            "Exact only for fixed ordered leaves, fixed affinity/contrast and "
            "the stated additive boundary-aware objective."),
        "greedy": (
            "Adjacent bottom-up heuristic maximising "
            "(1-lambda)*mean affinity-lambda*boundary contrast-"
            "beta*squared child-size imbalance."),
        "contrast_features": (
            "Cross-context dissimilarity minus half the two within-context "
            "dispersions for Euclidean, circle-of-fifths and key-profile distances."),
        "contrast_calibration": "Training-work-only median/MAD zero-anchored sigmoid.",
        "supervision": (
            "DCML local-key boundaries train the scorer only; outer-test labels "
            "are read only by evaluation."),
        "oracle_warning": "oracle_budget rows are diagnostics and not model results",
        "bin_size_qb": args.bin_size,
        "affinity_method": args.affinity_method,
        "affinity_context_radius": args.affinity_context_radius,
        "contexts": args.contexts, "lambdas": args.lambdas,
        "balance_weights": args.balance_weights,
        "thresholds": args.thresholds,
        "boundary_budgets": args.boundary_budgets,
        "tolerances_qb": args.tolerances,
        "selection_tolerance_qb": args.selection_tolerance,
        "outer_works": outer_works, "inner_folds": args.inner_folds,
        "random_seed": args.seed, "test_used_for_selection": False,
        "successful_movements": int(sum(
            row["status"] == "success" for row in statuses)),
        "skipped_movements": int(sum(
            row["status"] == "skipped" for row in statuses)),
        "failed_movements": int(sum(
            row["status"] == "failed" for row in statuses)),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHELD-OUT WORK SUMMARY")
    summary = work_results[
        (~work_results.oracle)
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)
    ].groupby(["model", "search", "selection_mode"], as_index=False).agg(
        works=("work", "nunique"), precision=("precision", "mean"),
        recall=("recall", "mean"), f1=("f1", "mean"), std_f1=("f1", "std"))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
