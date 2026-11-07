#!/usr/bin/env python
"""Evaluate Greedy/exact-DP trees against aligned explicit form levels."""

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

from dp_clustering import additive_tree_cost, optimal_adjacent_binary_tree
from hierarchical_evaluation import equal_budget_level_scores, tree_measure
from shared_annotation_format import SharedPiece
from shared_evaluation import boundary_reference


DEFAULT_METHODS = ["euclidean", "circle_of_fifths", "key_profile", "tonic_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-root", type=Path,
                        default=Path("data/shared_annotations_v1_aligned"))
    parser.add_argument("--dataset", choices=["beethoven", "mozart", "both"],
                        default="mozart")
    parser.add_argument("--piece", action="append")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--reference-levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--searches", nargs="+", choices=["greedy", "dp"],
                        default=["greedy", "dp"])
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tree-window-qb", type=float, default=None,
                        help="Optional local T-measure window; default uses the full piece")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/shared_hierarchy_stage"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _paths(args: argparse.Namespace) -> list[Path]:
    datasets = []
    if args.dataset in {"beethoven", "both"}:
        datasets.append("Taking_Form_Beethoven")
    if args.dataset in {"mozart", "both"}:
        datasets.append("Taking_Form_Mozart")
    paths = [path for dataset in datasets
             for path in sorted((args.aligned_root / dataset).glob("*.json"))]
    if args.piece:
        wanted = set(args.piece)
        paths = [path for path in paths if path.stem in wanted]
    if args.quick:
        paths = paths[:1]
    return paths


def _segments(piece: SharedPiece, level: int):
    boundaries = boundary_reference(piece, layers=["form"], max_level=level)
    boundaries = [value for value in boundaries if 0 < value < piece.duration_qb]
    edges = [0.0, *sorted(set(boundaries)), float(piece.duration_qb)]
    return [(edges[index], edges[index + 1], f"level{level}_segment{index}")
            for index in range(len(edges) - 1)]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = _paths(args)
    if not paths:
        raise SystemExit("No aligned shared hierarchy JSON files found")
    if args.quick:
        args.methods = ["euclidean"]
        args.reference_levels = [1]
        args.depths = [4]

    from greedy_clustering import ClusterNode, distance_functions, greedy_cluster, load_pc_bins
    from greedy_evaluation import bootstrap_summary, boundary_scores, collect_top_splits

    rows: list[dict] = []
    tree_rows: list[dict] = []
    budget_rows: list[dict] = []
    statuses: list[dict] = []
    for number, path in enumerate(paths, 1):
        piece = SharedPiece.read_json(path)
        try:
            if not piece.score_path or not Path(piece.score_path).exists():
                raise FileNotFoundError(piece.score_path or "missing score_path")
            matrix, bounds = load_pc_bins(Path(piece.score_path), args.bin_size)
            distances, estimated_key = distance_functions(matrix.sum(axis=0))
            missing = set(args.methods) - set(distances)
            if missing:
                raise KeyError(f"Unknown methods: {sorted(missing)}")
            for method in args.methods:
                distance = distances[method]
                trees = {}
                objectives = {}
                runtimes = {}
                diagnostics = None
                if "greedy" in args.searches:
                    started = time.perf_counter()
                    trees["greedy"] = greedy_cluster(matrix, bounds, distance)
                    runtimes["greedy"] = time.perf_counter() - started
                    objectives["greedy"] = additive_tree_cost(trees["greedy"], distance)
                if "dp" in args.searches:
                    trees["dp"], diagnostics = optimal_adjacent_binary_tree(
                        matrix, bounds, distance, ClusterNode, max_bins=args.max_bins)
                    runtimes["dp"] = diagnostics.elapsed_seconds
                    objectives["dp"] = diagnostics.total_cost
                if {"greedy", "dp"} <= set(trees) and (
                        objectives["dp"] > objectives["greedy"] + 1e-8):
                    raise AssertionError(
                        f"DP objective exceeds Greedy: {objectives['dp']} > {objectives['greedy']}")
                for search, tree in trees.items():
                    base = {
                        "dataset": piece.dataset, "piece": piece.piece_id,
                        "work": piece.work_id, "search": search, "method": method,
                        "bin_size_qb": args.bin_size,
                        "objective_cost": objectives[search],
                        "runtime_seconds": runtimes[search], "n_bins": len(matrix),
                    }
                    tree_rows.append({
                        **base, **tree_measure(
                            tree, piece, tree_id="form",
                            window_qb=args.tree_window_qb)})
                    for level in args.reference_levels:
                        budget_rows.append({
                            **base, "reference_max_level": level,
                            "tolerance_qb": args.tolerance,
                            **equal_budget_level_scores(
                                tree, piece, max_level=level,
                                tolerance_qb=args.tolerance)})
                for level in args.reference_levels:
                    segments = _segments(piece, level)
                    for depth in args.depths:
                        for search, tree in trees.items():
                            reference = [end for _, end, _ in segments[:-1]]
                            predicted = collect_top_splits(tree, depth)
                            score = boundary_scores(predicted, reference, args.tolerance)
                            row = {
                                "piece": piece.piece_id, "search": search,
                                "method": method, "bin_size_qb": args.bin_size,
                                "tolerance_qb": args.tolerance, "depth": depth,
                                **score,
                                "objective_cost": objectives[search],
                                "runtime_seconds": runtimes[search],
                                "evaluated_splits": (diagnostics.evaluated_splits
                                                     if search == "dp" and diagnostics else 0),
                                "tie_count": (diagnostics.tie_count
                                              if search == "dp" and diagnostics else 0),
                                "dataset": piece.dataset, "work": piece.work_id,
                                "reference_tree": "form",
                                "reference_max_level": level,
                                "reference_semantics": "cumulative explicit form boundaries",
                                "n_bins": len(matrix)}
                            rows.append(row)
            statuses.append({
                "dataset": piece.dataset, "piece": piece.piece_id,
                "work": piece.work_id, "status": "success", "message": "",
                **{f"key_{key}": value for key, value in estimated_key.items()},
            })
            print(f"[{number}/{len(paths)}] {piece.piece_id}: success")
        except Exception as error:
            status = "skipped" if "exceeds max_bins=" in str(error) else "failed"
            statuses.append({
                "dataset": piece.dataset, "piece": piece.piece_id,
                "work": piece.work_id, "status": status, "message": repr(error)})
            print(f"[{number}/{len(paths)}] {piece.piece_id}: {status}: {error}", file=sys.stderr)

    frame = pd.DataFrame(rows)
    pd.DataFrame(statuses).to_csv(args.output_dir / "run_status.csv", index=False)
    if frame.empty:
        raise SystemExit("No hierarchy evaluation succeeded")
    frame.to_csv(args.output_dir / "per_piece_results.csv", index=False)
    tree_frame = pd.DataFrame(tree_rows)
    budget_frame = pd.DataFrame(budget_rows)
    tree_frame.to_csv(args.output_dir / "tree_measure_per_piece.csv", index=False)
    budget_frame.to_csv(args.output_dir / "equal_budget_per_piece.csv", index=False)

    tree_metrics = ["t_precision_reduced", "t_recall_reduced",
                    "t_measure_reduced", "t_precision_full", "t_recall_full",
                    "t_measure_full", "lca_spearman", "lca_normalized_mae"]
    tree_work = (tree_frame.groupby(
        ["dataset", "work", "search", "method"], as_index=False)[tree_metrics]
        .mean())
    tree_summary = (tree_work.groupby(
        ["dataset", "search", "method"], as_index=False)
        .agg(works=("work", "nunique"),
             **{metric: (metric, "mean") for metric in tree_metrics}))
    tree_work.to_csv(args.output_dir / "tree_measure_per_work.csv", index=False)
    tree_summary.to_csv(args.output_dir / "tree_measure_summary.csv", index=False)
    tree_bootstrap = []
    for offset, metric in enumerate(tree_metrics):
        item = bootstrap_summary(
            tree_work, ["dataset", "search", "method"], value=metric,
            samples=2000, seed=args.seed + offset)
        item.insert(3, "metric", metric)
        tree_bootstrap.append(item)
    pd.concat(tree_bootstrap, ignore_index=True).to_csv(
        args.output_dir / "tree_measure_bootstrap.csv", index=False)

    budget_work = (budget_frame.groupby(
        ["dataset", "work", "search", "method", "reference_max_level"],
        as_index=False)
        .agg(precision=("precision", "mean"), recall=("recall", "mean"),
             f1=("f1", "mean"), tp=("tp", "sum"), fp=("fp", "sum"),
             fn=("fn", "sum")))
    budget_summary = (budget_work.groupby(
        ["dataset", "search", "method", "reference_max_level"], as_index=False)
        .agg(works=("work", "nunique"), precision=("precision", "mean"),
             recall=("recall", "mean"), f1=("f1", "mean")))
    budget_work.to_csv(args.output_dir / "equal_budget_per_work.csv", index=False)
    budget_summary.to_csv(args.output_dir / "equal_budget_summary.csv", index=False)
    bootstrap_summary(
        budget_work,
        ["dataset", "search", "method", "reference_max_level"],
        value="f1", samples=2000, seed=args.seed
    ).to_csv(args.output_dir / "equal_budget_bootstrap.csv", index=False)

    test_rows = []
    rng = np.random.default_rng(args.seed)
    for (dataset, method), group in tree_work.groupby(["dataset", "method"]):
        for metric in ("t_measure_reduced", "t_measure_full"):
            pivot = group.pivot_table(
                index="work", columns="search", values=metric, aggfunc="mean")
            if not {"greedy", "dp"} <= set(pivot.columns):
                continue
            paired = pivot[["dp", "greedy"]].dropna()
            differences = (paired["dp"] - paired["greedy"]).to_numpy(float)
            observed = abs(float(np.mean(differences)))
            signs = rng.choice([-1.0, 1.0], size=(10000, len(differences)))
            null = np.abs(np.mean(signs * differences, axis=1))
            test_rows.append({
                "dataset": dataset, "method": method, "metric": metric,
                "n_works": len(differences),
                "mean_dp_minus_greedy": float(np.mean(differences)),
                "p_value": float((1 + np.sum(null >= observed)) / 10001)})
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        ordered = tests.sort_values("p_value")
        adjusted = []
        running = 0.0
        for rank, value in enumerate(ordered.p_value):
            running = max(running, min(1.0, (len(ordered) - rank) * value))
            adjusted.append(running)
        tests["p_holm"] = np.nan
        tests.loc[ordered.index, "p_holm"] = adjusted
    tests.to_csv(args.output_dir / "dp_vs_greedy_tree_tests.csv", index=False)
    work = (frame.groupby(
        ["dataset", "work", "search", "method", "reference_max_level", "depth"],
        as_index=False)
        .agg(precision=("precision", "mean"), recall=("recall", "mean"),
             f1=("f1", "mean"), objective_cost=("objective_cost", "mean"),
             runtime_seconds=("runtime_seconds", "mean")))
    work.to_csv(args.output_dir / "per_work_results.csv", index=False)
    summary = (work.groupby(
        ["dataset", "search", "method", "reference_max_level", "depth"], as_index=False)
        .agg(works=("work", "nunique"), precision=("precision", "mean"),
             recall=("recall", "mean"), f1=("f1", "mean"),
             objective_cost=("objective_cost", "mean"),
             runtime_seconds=("runtime_seconds", "mean")))
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    selected_rows = []
    for (dataset, method, level), group in work.groupby(
            ["dataset", "method", "reference_max_level"]):
        for held_out in sorted(group.work.unique()):
            train = group[group.work != held_out]
            scores = (train.groupby("depth", as_index=False)
                      .agg(f1=("f1", "mean"), recall=("recall", "mean"))
                      .sort_values(["f1", "recall", "depth"],
                                   ascending=[False, False, True]))
            if scores.empty:
                continue
            selected_depth = int(scores.iloc[0].depth)
            test = group[(group.work == held_out)
                         & (group.depth == selected_depth)].copy()
            test["selected_depth"] = selected_depth
            test["selection_train_f1"] = float(scores.iloc[0].f1)
            test["held_out_work"] = held_out
            selected_rows.append(test)
    selected = (pd.concat(selected_rows, ignore_index=True)
                if selected_rows else pd.DataFrame())
    selected.to_csv(args.output_dir / "loowcv_selected_per_work.csv", index=False)
    if not selected.empty:
        selected_summary = (selected.groupby(
            ["dataset", "search", "method", "reference_max_level"], as_index=False)
            .agg(works=("work", "nunique"), precision=("precision", "mean"),
                 recall=("recall", "mean"), f1=("f1", "mean"),
                 objective_cost=("objective_cost", "mean"),
                 runtime_seconds=("runtime_seconds", "mean")))
        selected_summary.to_csv(args.output_dir / "loowcv_summary.csv", index=False)
    metadata = {
        "reference": (
            "Explicit Taking Form hierarchy boundaries accumulated through each "
            "reported source level. This is not DCML local-key ground truth."),
        "primary_metric": (
            "Full/reduced T-measure over input-bin samples; equal-budget Boundary "
            "F1 is the interpretable secondary metric."),
        "tree_measure": (
            "McFee, Nieto, and Bello (ISMIR 2015). LCA rank agreement is "
            "computed directly on ordered input-bin centres, so unequal tree "
            "depths are allowed and the genuine hierarchy is not flattened."),
        "primary_selection": (
            "Leave-one-work-out depth selection on training works, averaging "
            "Greedy and DP; the selected depth is shared by both searches."),
        "ted": (
            "Not computed here: the source is a genuine multi-level hierarchy, "
            "so flat-segmentation TED would be methodologically misleading."),
        "optimality_scope": "fixed ordered leaves and additive merge-distance objective",
        "configuration": vars(args),
    }
    metadata["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["configuration"].items()}
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
