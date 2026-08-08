#!/usr/bin/env python
"""Evaluate greedy and globally optimal DP trees under identical distances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT.parent / "src"))
sys.path.insert(0, "src")

from dp_clustering import (
    additive_tree_cost,
    assert_valid_ordered_binary_tree,
    optimal_adjacent_binary_tree,
)
from dp_stage_evaluation import evaluate_tree, print_result_table


DEFAULT_METHODS = [
    "euclidean",
    "circle_of_fifths",
    "key_profile",
    "tonic_weighted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dp_stage"))
    parser.add_argument("--piece", action="append", help="Repeat to select pieces")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--searches",
        nargs="+",
        choices=["greedy", "dp"],
        default=["greedy", "dp"],
    )
    parser.add_argument("--balance-lambda", type=float, default=0.0)
    parser.add_argument(
        "--length-weight",
        choices=["none", "log", "linear"],
        default="none",
        help="Use none for the strict greedy-versus-DP comparison.",
    )
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def discover_pairs(root: Path, selected: list[str] | None) -> list[tuple[str, Path, Path]]:
    notes = {
        path.name.replace(".notes.tsv", ""): path
        for path in (root / "notes").glob("*.notes.tsv")
    }
    harmonies = {
        path.name.replace(".harmonies.tsv", ""): path
        for path in (root / "harmonies").glob("*.harmonies.tsv")
    }
    stems = sorted(set(notes) & set(harmonies))
    if selected:
        allowed = set(selected)
        stems = [stem for stem in stems if stem in allowed]
    return [(stem, notes[stem], harmonies[stem]) for stem in stems]


def get_length_weight(name: str):
    if name == "none":
        return None
    if name == "log":
        return lambda length: float(np.log1p(length))
    if name == "linear":
        return lambda length: float(length)
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from greedy_clustering import (
        ClusterNode,
        distance_functions,
        greedy_cluster,
        load_pc_bins,
    )
    from greedy_evaluation import dcml_localkey_segments

    pairs = discover_pairs(args.data_root, args.piece)
    if args.quick:
        quick_works=sorted({piece.rsplit('_',1)[0] for piece,_,_ in pairs})[:2]
        pairs=[pair for pair in pairs if pair[0].rsplit('_',1)[0] in quick_works]
        args.methods = ['euclidean']
    if not pairs:
        raise SystemExit("No paired notes/harmonies files were found.")

    rows: list[dict] = []
    statuses: list[dict] = []
    length_weight = get_length_weight(args.length_weight)

    for number, (piece, notes_path, harmonies_path) in enumerate(pairs, 1):
        try:
            segments, total = dcml_localkey_segments(harmonies_path)
            matrix, bounds = load_pc_bins(notes_path, args.bin_size)
            distances, estimated_key = distance_functions(
                matrix.sum(axis=0),
                include_ablation=False,
            )

            missing = [name for name in args.methods if name not in distances]
            if missing:
                raise KeyError(
                    f"Methods absent from distance_functions(): {missing}. "
                    f"Available: {sorted(distances)}"
                )

            piece_results = []
            for method in args.methods:
                distance_fn = distances[method]

                if "greedy" in args.searches:
                    started = time.perf_counter()
                    greedy_tree = greedy_cluster(matrix, bounds, distance_fn)
                    greedy_seconds = time.perf_counter() - started
                    greedy_objective = additive_tree_cost(
                        greedy_tree,
                        distance_fn,
                        length_weight=length_weight,
                        balance_lambda=args.balance_lambda,
                    )
                    result = evaluate_tree(
                        piece=piece,
                        search="greedy",
                        method=method,
                        tree=greedy_tree,
                        segments=segments,
                        total=total,
                        bin_size_qb=args.bin_size,
                        tolerance_qb=args.tolerance,
                        depth=args.depth,
                        objective_cost=greedy_objective,
                        runtime_seconds=greedy_seconds,
                    )
                    piece_results.append(result)

                if "dp" in args.searches:
                    dp_tree, diagnostics = optimal_adjacent_binary_tree(
                        matrix,
                        bounds,
                        distance_fn,
                        ClusterNode,
                        length_weight=length_weight,
                        balance_lambda=args.balance_lambda,
                        max_bins=args.max_bins,
                    )
                    assert_valid_ordered_binary_tree(dp_tree, bounds)
                    result = evaluate_tree(
                        piece=piece,
                        search="dp",
                        method=method,
                        tree=dp_tree,
                        segments=segments,
                        total=total,
                        bin_size_qb=args.bin_size,
                        tolerance_qb=args.tolerance,
                        depth=args.depth,
                        objective_cost=diagnostics.total_cost,
                        runtime_seconds=diagnostics.elapsed_seconds,
                        evaluated_splits=diagnostics.evaluated_splits,
                        tie_count=diagnostics.tie_count,
                    )
                    piece_results.append(result)

                # Strong implementation check for the unregularised objective.
                if {"greedy", "dp"} <= set(args.searches):
                    g = next(
                        item for item in piece_results
                        if item.method == method and item.search == "greedy"
                    )
                    d = next(
                        item for item in piece_results
                        if item.method == method and item.search == "dp"
                    )
                    if d.objective_cost > g.objective_cost + 1e-8:
                        raise AssertionError(
                            f"DP objective exceeds greedy objective for {piece}/{method}: "
                            f"{d.objective_cost} > {g.objective_cost}"
                        )

            print_result_table(
                piece_results,
                title=(
                    f"[{number}/{len(pairs)}] {piece} | "
                    f"bins={len(matrix)}, bin={args.bin_size:g} qb, "
                    f"tol={args.tolerance:g} qb, depth={args.depth}"
                ),
            )
            for result in piece_results:
                row=result.as_dict()
                row['n_bins']=len(matrix)
                rows.append(row)
            statuses.append(
                {
                    "piece": piece,
                    "status": "success",
                    "message": "",
                    **{f"key_{k}": v for k, v in estimated_key.items()},
                }
            )
        except Exception as error:
            status='skipped' if 'exceeds max_bins=' in str(error) else 'failed'
            statuses.append(
                {
                    "piece": piece,
                    "status": status,
                    "message": repr(error),
                }
            )
            print(f"[{number}/{len(pairs)}] {piece}: {status.upper()}: {error}", file=sys.stderr)

    frame = pd.DataFrame(rows)
    status_frame = pd.DataFrame(statuses)
    status_frame.to_csv(args.output_dir / "run_status.csv", index=False)
    if frame.empty:
        raise SystemExit("All pieces failed. Inspect run_status.csv.")

    frame.to_csv(args.output_dir / "per_piece_results.csv", index=False)

    work_frame=(
        frame.groupby(['work','search','method'],as_index=False)
        .agg(precision=('precision','mean'),recall=('recall','mean'),
             f1=('f1','mean'),normalized_pruned_ted=('normalized_pruned_ted','mean'),
             objective_cost=('objective_cost','mean'),runtime_seconds=('runtime_seconds','mean'))
    )
    work_frame.to_csv(args.output_dir / 'per_work_results.csv',index=False)
    summary = (
        work_frame.groupby(["search", "method"], as_index=False)
        .agg(
            works=("work", "nunique"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            normalized_pruned_ted=("normalized_pruned_ted", "mean"),
            objective_cost=("objective_cost", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(["f1", "search"], ascending=[False, True])
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    comparisons=[]
    for method,group in frame.groupby('method'):
        pivot=group.pivot_table(index='piece',columns='search',values='objective_cost',aggfunc='mean')
        if {'greedy','dp'} <= set(pivot):
            paired=pivot[['greedy','dp']].dropna().reset_index()
            paired['method']=method
            paired['objective_gap']=paired.greedy-paired.dp
            paired['relative_objective_gap']=paired.objective_gap/paired.greedy.abs().replace(0,np.nan)
            comparisons.append(paired)
    comparison=pd.concat(comparisons,ignore_index=True) if comparisons else pd.DataFrame()
    comparison.to_csv(args.output_dir / 'objective_comparison.csv',index=False)

    from greedy_evaluation import paired_permutation_tests
    test_rows=[]
    for method,group in work_frame.groupby('method'):
        temporary=group[['work','search','f1']].rename(columns={'search':'method'})
        tested=paired_permutation_tests(temporary,samples=10000,seed=0,unit_column='work')
        if not tested.empty:
            tested.insert(0,'distance',method)
            test_rows.append(tested)
    tests=pd.concat(test_rows,ignore_index=True) if test_rows else pd.DataFrame()
    if not tests.empty:
        ordered=tests.sort_values('p_value')
        running=0.0; adjusted=[]
        for rank,p_value in enumerate(ordered.p_value):
            running=max(running,min(1.0,(len(ordered)-rank)*p_value))
            adjusted.append(running)
        tests['p_holm']=np.nan
        tests.loc[ordered.index,'p_holm']=adjusted
    tests.to_csv(args.output_dir / 'paired_dp_vs_greedy.csv',index=False)

    print("\nCORPUS SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    metadata = {
        "objective": (
            "Sum over internal nodes of the selected distance between aggregate "
            "left- and right-child interval representations."
        ),
        "global_optimality_scope": (
            "DP is globally optimal only under the stated additive objective."
        ),
        "bin_size_qb": args.bin_size,
        "tolerance_qb": args.tolerance,
        "depth": args.depth,
        "length_weight": args.length_weight,
        "balance_lambda": args.balance_lambda,
        "methods": args.methods,
        "searches": args.searches,
        "successful_movements": int((status_frame.status == "success").sum()),
        "failed_movements": int((status_frame.status == "failed").sum()),
        "skipped_movements": int((status_frame.status == "skipped").sum()),
        "successful_works": int(frame.work.nunique()),
        "ted_interpretation": (
            "Auxiliary structural discrepancy between the predicted hierarchy "
            "and a flat tree induced by DCML local-key segments."
        ),
        "distance_metadata": {
            "normalization": "L1 normalization of every aggregate pitch-class vector.",
            "euclidean": "Euclidean distance between normalized 12-D pitch-class vectors.",
            "circle_of_fifths": (
                "Euclidean distance between two-dimensional weighted centroids on "
                "the circle of fifths; Tonnetz-inspired, not a complete Tonnetz."
            ),
            "key_profile": (
                "Euclidean distance between 24 correlations with rotated "
                "Krumhansl-Kessler major/minor profiles."
            ),
            "tonic_weighted": (
                "Weighted Euclidean distance using the rotated Krumhansl-Kessler "
                "profile selected from the piece pitch-class distribution only."
            ),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
