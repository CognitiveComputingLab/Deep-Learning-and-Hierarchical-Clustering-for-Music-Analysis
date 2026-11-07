#!/usr/bin/env python
"""Nested work-blocked evaluation of boundary-aware Greedy and exact DP."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    ProbabilityCalibrator,
    RobustContrastScaler,
    boundary_contrast_features,
    fit_probability_calibrator,
    fit_supervised_scorer,
    make_supervised_examples,
    tolerance_aware_boundary_labels,
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
SELECTION_METRICS = ("f1", "f0.5", "precision_at_recall_floor")


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
    parser.add_argument(
        "--min-separations-qb", nargs="+", type=float, default=[0, 8, 16, 24],
        help="validation grid for salience-descending boundary NMS; 0 disables NMS")
    parser.add_argument(
        "--local-max-radii-qb", nargs="+", type=float, default=[0, 8],
        help="validation grid for local salience peak selection; 0 keeps all candidates")
    parser.add_argument("--boundary-budgets", nargs="+", type=int,
                        default=[3, 5, 8, 10, 12, 15, 20])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[2, 4, 8, 12, 24])
    parser.add_argument("--selection-tolerance", type=float, default=8.0)
    parser.add_argument(
        "--selection-metric", choices=SELECTION_METRICS, default="f1",
        help="inner-validation operating-point objective; F1 remains the default")
    parser.add_argument(
        "--recall-floor", type=float,
        help=("absolute validation recall constraint for precision_at_recall_floor; "
              "when omitted, use F1-selected recall minus --max-recall-drop"))
    parser.add_argument(
        "--max-recall-drop", type=float, default=0.05,
        help="maximum recall loss from the F1-selected validation candidate")
    parser.add_argument(
        "--threshold-calibration", choices=["shared", "both"], default="shared",
        help=("shared uses one threshold for Greedy/DP; both additionally reports "
              "search-specific thresholds with the same selected tree objective"))
    parser.add_argument(
        "--ignore-tolerance-qb", type=float, default=None,
        help=("exclude near-annotation candidates from supervised negatives; "
              "default uses --selection-tolerance, while 0 is legacy"))
    parser.add_argument(
        "--probability-calibration", choices=["none", "platt"], default="platt",
        help="training-work-only calibration; none preserves legacy scores")
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


def f_beta(precision, recall, beta=0.5):
    """Return a safe element-wise F-beta score."""
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    beta_squared = float(beta) ** 2
    denominator = beta_squared * precision + recall
    return np.divide(
        (1.0 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )


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


@dataclass(frozen=True)
class ProbabilityCalibratedScorer:
    """Apply a training-only probability calibrator to a base scorer."""

    base: BoundaryContrastScorer
    calibrator: ProbabilityCalibrator

    @property
    def scaler(self):
        return self.base.scaler

    @property
    def weights(self):
        return self.base.weights

    @property
    def mode(self):
        return self.base.mode

    @property
    def intercept(self):
        return self.base.intercept

    @property
    def gain(self):
        return self.base.gain

    def score(self, features):
        return self.calibrator.transform(self.base.score(features))

    def as_dict(self):
        payload = self.base.as_dict()
        payload["probability_calibration"] = self.calibrator.as_dict()
        return payload


def scorer_training_items(items):
    return [{
        "work": item["work"], "piece": item["piece"], "features": item["features"],
        "bounds": item["bounds"], "reference": item["reference"],
    } for item in items]


def natural_calibration_examples(items, scorer, ignore_tolerance_qb):
    """Return all non-ignored candidates with work-balanced natural priors."""
    records = []
    for item in sorted(items, key=lambda value: (
            str(value["work"]), str(value["piece"]))):
        features = item["features"]
        scores = np.asarray(scorer.score(features), dtype=float)
        labels, ignored = tolerance_aware_boundary_labels(
            item["bounds"], item["reference"], len(scores) + 1,
            ignore_tolerance_qb=ignore_tolerance_qb)
        for split_index, (score, label, is_ignored) in enumerate(
                zip(scores, labels, ignored), 1):
            records.append({
                "work": str(item["work"]), "piece": str(item["piece"]),
                "split_index": split_index, "score": float(score),
                "label": int(label), "ignored": bool(is_ignored),
            })
    counts = {}
    for row in records:
        if not row["ignored"]:
            counts[row["work"]] = counts.get(row["work"], 0) + 1
    if not counts:
        raise ValueError("no non-ignored candidates available for calibration")
    work_mass = 1.0 / len(counts)
    scores, labels, weights, audit = [], [], [], []
    for row in records:
        eligible = not row["ignored"]
        weight = work_mass / counts[row["work"]] if eligible else 0.0
        audit.append({
            "work": row["work"], "piece": row["piece"],
            "split_index": row["split_index"],
            "probability_calibration": "platt",
            "calibration_eligible": eligible,
            "calibration_sample_weight": float(weight),
            "uncalibrated_probability": row["score"],
        })
        if eligible:
            scores.append(row["score"])
            labels.append(row["label"])
            weights.append(weight)
    return (np.asarray(scores, dtype=float), np.asarray(labels, dtype=float),
            np.asarray(weights, dtype=float), audit)


def fit_contrast_models(items, args):
    scaler = RobustContrastScaler.fit([item["features"] for item in items])
    unsupervised = BoundaryContrastScorer.unsupervised(scaler)
    calibrated, labels, sample_weights, audit = make_supervised_examples(
        scorer_training_items(items), scaler,
        ignore_tolerance_qb=args.ignore_tolerance_qb)
    supervised, history = fit_supervised_scorer(
        scaler, calibrated, labels, sample_weights, epochs=args.epochs,
        learning_rate=args.learning_rate, regularization=args.regularization)
    calibration_mode = getattr(args, "probability_calibration", "none")
    if calibration_mode == "platt":
        scores, natural_labels, natural_weights, calibration_audit = (
            natural_calibration_examples(
                items, supervised, args.ignore_tolerance_qb))
        calibrator = fit_probability_calibrator(
            scores, natural_labels, natural_weights)
        supervised = ProbabilityCalibratedScorer(supervised, calibrator)
        calibration_by_candidate = {
            (row["work"], row["piece"], row["split_index"]): row
            for row in calibration_audit
        }
        for row in audit:
            row.update(calibration_by_candidate[
                (row["work"], row["piece"], row["split_index"])])
    elif calibration_mode == "none":
        for row in audit:
            row.update({
                "probability_calibration": "none",
                "calibration_eligible": False,
                "calibration_sample_weight": 0.0,
                "uncalibrated_probability": np.nan,
            })
    else:
        raise ValueError(
            "probability_calibration must be either 'platt' or 'none'")
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
                for min_separation in args.min_separations_qb:
                    for local_max_radius in args.local_max_radii_qb:
                        for threshold in args.thresholds:
                            predicted = collect_salient_splits(
                                tree, contrast, threshold=threshold,
                                min_separation_qb=min_separation,
                                local_max_radius_qb=local_max_radius)
                            score = boundary_scores(
                                predicted, item["reference"], args.selection_tolerance)
                            rows.append({
                                **common, "selection_mode": "threshold",
                                "candidate_value": float(threshold),
                                "min_separation_qb": float(min_separation),
                                "local_max_radius_qb": float(local_max_radius),
                                "f0.5": float(f_beta(score["precision"], score["recall"])),
                                **{key: score[key] for key in
                                   ["tp", "fp", "fn", "precision", "recall", "f1",
                                    "predicted_boundary_count", "gt_boundary_count"]},
                            })
                for budget in args.boundary_budgets:
                    # Fixed-budget rows deliberately disable peak/NMS filtering:
                    # otherwise two searches need not return the same boundary count.
                    predicted = collect_salient_splits(tree, contrast, budget=budget)
                    score = boundary_scores(
                        predicted, item["reference"], args.selection_tolerance)
                    rows.append({
                        **common, "selection_mode": "fixed_budget",
                        "candidate_value": int(budget),
                        "min_separation_qb": 0.0,
                        "local_max_radius_qb": 0.0,
                        "f0.5": float(f_beta(score["precision"], score["recall"])),
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


def _sort_candidates(candidates, metric, selection_mode):
    """Deterministically order validation candidates for one objective."""
    candidate_ascending = selection_mode == "fixed_budget"
    trailing = [
        "candidate_value", "min_separation_qb", "local_max_radius_qb",
        "balance_weight", "contrast_weight",
    ]
    if metric == "f1":
        columns = ["f1", "recall", "precision"] + trailing
        ascending = [False, False, False, candidate_ascending, True, True, True, True]
    elif metric == "f0.5":
        columns = ["f0.5", "precision", "recall", "f1"] + trailing
        ascending = [False, False, False, False,
                     candidate_ascending, True, True, True, True]
    elif metric == "precision_at_recall_floor":
        columns = ["precision", "recall", "f0.5", "f1"] + trailing
        ascending = [False, False, False, False,
                     candidate_ascending, True, True, True, True]
    else:
        raise ValueError(f"unknown selection metric: {metric}")
    return candidates.sort_values(columns, ascending=ascending, kind="stable")


def select_candidate(
    frame,
    mode,
    selection_mode,
    *,
    selection_metric="f1",
    recall_floor=None,
    max_recall_drop=0.05,
    search=None,
    fixed_tree_parameters=None,
):
    """Choose one nested-validation operating point without test-label access.

    ``precision_at_recall_floor`` uses an explicit absolute floor when supplied.
    Otherwise its floor is the recall of the ordinary F1-selected candidate
    minus ``max_recall_drop``. Search-specific calls may fix the tree objective
    chosen by the shared comparison and calibrate only extraction parameters.
    """
    subset = frame[(frame.model == mode) & (frame.selection_mode == selection_mode)]
    if search is not None:
        subset = subset[subset.search == search]
    if fixed_tree_parameters:
        for column in ("contrast_weight", "balance_weight"):
            if column in fixed_tree_parameters:
                subset = subset[np.isclose(
                    subset[column], float(fixed_tree_parameters[column]))]
    if subset.empty:
        raise ValueError(f"no validation candidates for {mode}/{selection_mode}")
    work = (subset.groupby(
        ["work", "search", "contrast_weight", "balance_weight",
         "candidate_value", "min_separation_qb", "local_max_radius_qb"],
        as_index=False)
        [["precision", "recall", "f1", "f0.5"]].mean())
    candidates = (work.groupby(
        ["contrast_weight", "balance_weight", "candidate_value",
         "min_separation_qb", "local_max_radius_qb"], as_index=False)
        [["precision", "recall", "f1", "f0.5"]].mean())

    candidates["selection_metric"] = selection_metric
    candidates["recall_floor"] = np.nan
    candidates["eligible_for_selection"] = True
    if selection_metric == "precision_at_recall_floor":
        if not 0 <= max_recall_drop <= 1:
            raise ValueError("max_recall_drop must lie in [0,1]")
        f1_reference = _sort_candidates(candidates, "f1", selection_mode).iloc[0]
        effective_floor = (
            float(recall_floor) if recall_floor is not None
            else max(0.0, float(f1_reference["recall"]) - max_recall_drop)
        )
        if not 0 <= effective_floor <= 1:
            raise ValueError("recall_floor must lie in [0,1]")
        candidates["recall_floor"] = effective_floor
        candidates["eligible_for_selection"] = (
            candidates.recall >= effective_floor - 1e-12)
        eligible = candidates[candidates.eligible_for_selection].copy()
        if eligible.empty:
            raise ValueError(
                f"recall floor {effective_floor:.4f} is infeasible for "
                f"{mode}/{selection_mode}")
        ordered = _sort_candidates(
            eligible, "precision_at_recall_floor", selection_mode)
    else:
        ordered = _sort_candidates(candidates, selection_metric, selection_mode)
    score_column = "f0.5" if selection_metric == "f0.5" else (
        "precision" if selection_metric == "precision_at_recall_floor" else "f1")
    candidates["selection_score"] = candidates[score_column]
    ordered = ordered.copy()
    ordered["selection_score"] = ordered[score_column]
    ordered["selection_metric"] = selection_metric
    if selection_metric == "precision_at_recall_floor":
        ordered["recall_floor"] = effective_floor
    return ordered.iloc[0].to_dict(), candidates


def model_parameter_rows(models, outer_work, stage, inner_fold=None):
    rows = []
    for mode, scorer in models.items():
        calibrator = getattr(scorer, "calibrator", None)
        for index, name in enumerate(scorer.scaler.names):
            rows.append({
                "outer_work": outer_work, "stage": stage, "inner_fold": inner_fold,
                "model": mode, "feature": name,
                "weight": float(scorer.weights[index]),
                "training_median": float(scorer.scaler.medians[index]),
                "training_mad": float(scorer.scaler.mads[index]),
                "training_scale": float(scorer.scaler.scales[index]),
                "intercept": float(scorer.intercept), "gain": float(scorer.gain),
                "probability_calibration": (
                    "platt" if calibrator is not None else "none"),
                "calibration_intercept": (
                    float(calibrator.intercept) if calibrator else np.nan),
                "calibration_slope": (
                    float(calibrator.slope) if calibrator else np.nan),
                "calibration_score_location": (
                    float(calibrator.score_location) if calibrator else np.nan),
                "calibration_score_scale": (
                    float(calibrator.score_scale) if calibrator else np.nan),
                "calibration_fit_scope": (
                    calibrator.fit_scope if calibrator else None),
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
        selected_configs = [
            (key, choice) for key, choice in selections.items()
            if key[0] == mode and key[1] in {"threshold", "fixed_budget"}
        ]
        for key, choice in selected_configs:
            _, selection_mode, calibration_scope, calibrated_search = key
            contrast_weight = float(choice["contrast_weight"])
            balance_weight = float(choice["balance_weight"])
            candidate = float(choice["candidate_value"])
            min_separation = float(choice.get("min_separation_qb", 0.0))
            local_max_radius = float(choice.get("local_max_radius_qb", 0.0))
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
                    if calibrated_search is not None and search != calibrated_search:
                        continue
                    if selection_mode == "threshold":
                        predicted = collect_salient_splits(
                            tree, contrast, threshold=candidate,
                            min_separation_qb=min_separation,
                            local_max_radius_qb=local_max_radius)
                    else:
                        predicted = collect_salient_splits(
                            tree, contrast, budget=int(candidate))
                    for tolerance in args.tolerances:
                        score = boundary_scores(predicted, item["reference"], tolerance)
                        rows.append({
                            "outer_work": outer_work, "piece": item["piece"],
                            "work": item["work"], "model": mode, "search": search,
                            "selection_mode": selection_mode,
                            "calibration_scope": calibration_scope,
                            "selected_value": candidate,
                            "min_separation_qb": min_separation,
                            "local_max_radius_qb": local_max_radius,
                            "selection_metric": args.selection_metric,
                            "contrast_weight": contrast_weight,
                            "balance_weight": balance_weight,
                            "tolerance_qb": float(tolerance), "oracle": False,
                            "f0.5": float(f_beta(score["precision"], score["recall"])),
                            **{key: score[key] for key in
                               ["tp", "fp", "fn", "precision", "recall", "f1",
                                "predicted_boundary_count", "gt_boundary_count",
                                "mean_match_error_qb"]},
                        })

        # Oracle count is diagnostic only and uses the fixed-budget-selected tree.
        choice = selections[(mode, "fixed_budget", "shared", None)]
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
                        "calibration_scope": "oracle",
                        "selected_value": len(item["reference"]),
                        "min_separation_qb": 0.0,
                        "local_max_radius_qb": 0.0,
                        "selection_metric": "oracle",
                        "contrast_weight": contrast_weight,
                        "balance_weight": balance_weight,
                        "tolerance_qb": float(tolerance), "oracle": True,
                        "f0.5": float(f_beta(score["precision"], score["recall"])),
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
            ["model", "search", "selection_mode", "calibration_scope",
             "tolerance_qb"],
            as_index=False).f1.mean()
        figure, axis = plt.subplots(figsize=(8, 5))
        for keys, group in tolerance.groupby(
                ["model", "search", "selection_mode", "calibration_scope"]):
            axis.plot(group.tolerance_qb, group.f1, marker="o",
                      label="/".join(map(str, keys)))
        axis.set(xlabel="Tolerance (quarterbeats)", ylabel="Mean movement F1",
                 title="Held-out Boundary F1 vs tolerance")
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / "f1_vs_tolerance.png", dpi=160)
        plt.close(figure)

        pr = selected.groupby(
            ["model", "search", "selection_mode", "calibration_scope",
             "tolerance_qb"],
            as_index=False)[["precision", "recall"]].mean()
        figure, axis = plt.subplots(figsize=(7, 5))
        for keys, group in pr.groupby(
                ["model", "search", "selection_mode", "calibration_scope"]):
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
                  .drop_duplicates([
                      "outer_work", "model", "calibration_scope",
                      "calibrated_search", "candidate_value"]))
        curve = subset.groupby(
            ["model", "calibration_scope", "calibrated_search", "candidate_value"],
            as_index=False, dropna=False).f1.mean()
        figure, axis = plt.subplots(figsize=(7, 4))
        for keys, group in curve.groupby(
                ["model", "calibration_scope", "calibrated_search"],
                dropna=False):
            axis.plot(group.candidate_value, group.f1, marker="o",
                      label="/".join(map(str, keys)))
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
        args.min_separations_qb = [0.0, 16.0]
        args.local_max_radii_qb = [0.0, 8.0]
        args.boundary_budgets = [5, 10]
        args.tolerances = [args.selection_tolerance]
        args.inner_folds = 2
        args.epochs = min(args.epochs, 25)
    if any(not 0 <= value <= 1 for value in args.lambdas + args.thresholds):
        raise SystemExit("lambdas and thresholds must lie in [0,1]")
    if any(value < 0 for value in args.balance_weights):
        raise SystemExit("balance weights must be non-negative")
    if any(value < 0 or not np.isfinite(value)
           for value in args.min_separations_qb + args.local_max_radii_qb):
        raise SystemExit("NMS separations and local-max radii must be non-negative")
    if args.recall_floor is not None and not 0 <= args.recall_floor <= 1:
        raise SystemExit("recall floor must lie in [0,1]")
    if not 0 <= args.max_recall_drop <= 1:
        raise SystemExit("max recall drop must lie in [0,1]")
    if (not np.isfinite(args.selection_tolerance)
            or args.selection_tolerance < 0):
        raise SystemExit("selection tolerance must be finite and non-negative")
    if args.ignore_tolerance_qb is None:
        args.ignore_tolerance_qb = float(args.selection_tolerance)
    elif (not np.isfinite(args.ignore_tolerance_qb)
          or args.ignore_tolerance_qb < 0):
        raise SystemExit("ignore tolerance must be finite and non-negative")
    if args.ignore_tolerance_qb > args.selection_tolerance:
        raise SystemExit("ignore tolerance must not exceed selection tolerance")
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
                shared_choice, shared_candidates = select_candidate(
                    inner_frame, mode, selection_mode,
                    selection_metric=args.selection_metric,
                    recall_floor=args.recall_floor,
                    max_recall_drop=args.max_recall_drop)
                selection_variants = [
                    ("shared", None, shared_choice, shared_candidates)
                ]
                if (selection_mode == "threshold"
                        and args.threshold_calibration == "both"):
                    fixed_tree = {
                        "contrast_weight": shared_choice["contrast_weight"],
                        "balance_weight": shared_choice["balance_weight"],
                    }
                    for search in ("greedy", "dp"):
                        choice, candidates = select_candidate(
                            inner_frame, mode, selection_mode,
                            selection_metric=args.selection_metric,
                            recall_floor=args.recall_floor,
                            max_recall_drop=args.max_recall_drop,
                            search=search,
                            fixed_tree_parameters=fixed_tree)
                        selection_variants.append(
                            ("search_specific", search, choice, candidates))

                for scope, calibrated_search, chosen, candidates in selection_variants:
                    selections[(mode, selection_mode, scope, calibrated_search)] = chosen
                    candidates = candidates.copy()
                    candidates.insert(0, "calibrated_search", calibrated_search)
                    candidates.insert(0, "calibration_scope", scope)
                    candidates.insert(0, "selection_mode", selection_mode)
                    candidates.insert(0, "model", mode)
                    candidates.insert(0, "outer_work", outer_work)
                    candidates["chosen"] = (
                        np.isclose(candidates.contrast_weight,
                                   chosen["contrast_weight"])
                        & np.isclose(candidates.balance_weight,
                                     chosen["balance_weight"])
                        & np.isclose(candidates.candidate_value,
                                     chosen["candidate_value"])
                        & np.isclose(candidates.min_separation_qb,
                                     chosen["min_separation_qb"])
                        & np.isclose(candidates.local_max_radius_qb,
                                     chosen["local_max_radius_qb"]))
                    candidate_summary_all.extend(candidates.to_dict("records"))
                    selection_rows.append({
                        "outer_work": outer_work, "model": mode,
                        "selection_mode": selection_mode,
                        "calibration_scope": scope,
                        "calibrated_search": calibrated_search,
                        "selection_metric": args.selection_metric,
                        "selection_score": chosen["selection_score"],
                        "recall_floor": chosen.get("recall_floor", np.nan),
                        "selected_contrast_weight": chosen["contrast_weight"],
                        "selected_balance_weight": chosen["balance_weight"],
                        "selected_value": chosen["candidate_value"],
                        "selected_min_separation_qb": chosen["min_separation_qb"],
                        "selected_local_max_radius_qb": chosen["local_max_radius_qb"],
                        "inner_f0.5": chosen["f0.5"],
                        "inner_f1": chosen["f1"],
                        "inner_recall": chosen["recall"],
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
        "model", "search", "selection_mode", "calibration_scope",
        "tolerance_qb", "oracle"]
    work_results = (results.groupby(["work"] + group_columns, as_index=False)
                    [["tp", "fp", "fn", "precision", "recall", "f1",
                      "f0.5"]].mean())
    work_results.to_csv(args.output_dir / "held_out_per_work.csv", index=False)
    bootstrap_summary(
        work_results, group_columns, samples=2000, seed=args.seed).to_csv(
            args.output_dir / "work_macro_summary.csv", index=False)
    bootstrap_summary(
        work_results, group_columns, value="f0.5", samples=2000,
        seed=args.seed).to_csv(
            args.output_dir / "work_macro_f0_5_summary.csv", index=False)
    micro = micro_summary(results, group_columns)
    micro["f0.5"] = f_beta(micro.precision, micro.recall)
    micro.to_csv(args.output_dir / "micro_summary.csv", index=False)

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
        & work_results.calibration_scope.eq("shared")
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)
    ].copy()
    primary["method"] = (
        primary.model + "_" + primary.search + "_" + primary.calibration_scope)
    paired_permutation_tests(
        primary[["work", "method", "f1"]], samples=10000,
        seed=args.seed, unit_column="work").to_csv(
            args.output_dir / "primary_paired_tests.csv", index=False)
    exploratory = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & work_results.calibration_scope.eq("search_specific")
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)
    ].copy()
    if not exploratory.empty:
        exploratory["method"] = exploratory.model + "_" + exploratory.search
        paired_permutation_tests(
            exploratory[["work", "method", "f1"]], samples=10000,
            seed=args.seed, unit_column="work").to_csv(
                args.output_dir / "exploratory_search_specific_paired_tests.csv",
                index=False)
    plot_outputs(results, candidate_frame, comparison, args.output_dir)

    metadata = {
        "primary_metric": "work-level macro DCML local-key Boundary F1",
        "primary_calibration_scope": "shared",
        "selection_metric": args.selection_metric,
        "primary_selection": (
            "nested-validation salience threshold; selection metric is configurable, "
            "but held-out F1 remains the primary reported metric"),
        "threshold_calibration": args.threshold_calibration,
        "precision_selection_semantics": (
            "precision_at_recall_floor uses --recall-floor when supplied; otherwise "
            "the floor is validation F1-selected recall minus --max-recall-drop"),
        "recall_floor": args.recall_floor,
        "max_recall_drop": args.max_recall_drop,
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
        "min_separations_qb": args.min_separations_qb,
        "local_max_radii_qb": args.local_max_radii_qb,
        "fixed_budget_peak_nms": "disabled to preserve equal predicted counts",
        "boundary_budgets": args.boundary_budgets,
        "tolerances_qb": args.tolerances,
        "selection_tolerance_qb": args.selection_tolerance,
        "supervised_ignore_tolerance_qb": args.ignore_tolerance_qb,
        "probability_calibration": args.probability_calibration,
        "probability_calibration_fit_scope": (
            "current training works only: inner_train for each fold and "
            "outer_train for final refit; never validation or outer-test labels"),
        "probability_calibration_sampling": (
            "all non-ignored candidates, equal total weight per work, natural "
            "positive prevalence retained within each work"),
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
    ].groupby(
        ["model", "search", "selection_mode", "calibration_scope"],
        as_index=False).agg(
        works=("work", "nunique"), precision=("precision", "mean"),
        recall=("recall", "mean"), f0_5=("f0.5", "mean"),
        f1=("f1", "mean"), std_f1=("f1", "std"))
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
