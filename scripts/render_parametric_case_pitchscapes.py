#!/usr/bin/env python
"""Render held-out Greedy/DP case studies for the trained parametric distance.

Cases are selected from the two formal cross-corpus experiments.  Selection
requires DP gains in reduced/full T-measure and LCA Spearman correlation, then
ranks by reduced T-measure gain.  Expert form boundaries are visual references
only and never enter tree construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_contrast import (
    BoundaryContrastScorer,
    RobustContrastScaler,
    boundary_contrast_features,
)
from dp_pitchscape import collect_tree_geometry
from greedy_clustering import ClusterNode, distance_specs, load_pc_bins
from greedy_evaluation import tree_shape_diagnostics
from hierarchical_evaluation import tree_measure
from hierarchical_parametric import make_parametric_distance
from ordered_affinity import (
    boundary_aware_tree_objective,
    greedy_boundary_aware_tree,
    optimal_boundary_aware_tree,
    pairwise_affinity,
)
from shared_annotation_format import SharedPiece


FORMAL_RESULTS = (
    "parametric_hierarchy_beethoven_to_mozart",
    "parametric_hierarchy_mozart_to_beethoven",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--aligned-root", type=Path,
                        default=ROOT / "data/shared_annotations_v1_aligned")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/paper_parametric_cases")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--minimum-full-gain", type=float, default=0.08)
    parser.add_argument("--minimum-lca-gain", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--samples", type=int, default=180)
    parser.add_argument("--dpi", type=int, default=170)
    return parser.parse_args()


def paired_case_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Return one paired metric row per held-out piece."""
    data = pd.concat(frames, ignore_index=True)
    values = [
        "objective", "t_measure_reduced", "t_measure_full",
        "lca_spearman", "lca_normalized_mae",
    ]
    pivot = data.pivot_table(
        index=["source", "dataset", "piece", "work"],
        columns="search", values=values, aggfunc="first")
    pivot.columns = [f"{metric}_{search}" for metric, search in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ("objective", "t_measure_reduced", "t_measure_full",
                   "lca_spearman"):
        pivot[f"delta_{metric}"] = (
            pivot[f"{metric}_dp"] - pivot[f"{metric}_greedy"])
    pivot["delta_lca_normalized_mae"] = (
        pivot["lca_normalized_mae_greedy"]
        - pivot["lca_normalized_mae_dp"])
    return pivot


def select_cases(table: pd.DataFrame, *, top_k: int,
                 minimum_full_gain: float,
                 minimum_lca_gain: float) -> pd.DataFrame:
    """Select substantial, consistent DP improvements deterministically."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    selected = table.loc[
        (table.delta_objective >= -1e-8)
        & (table.delta_t_measure_reduced > 0)
        & (table.delta_t_measure_full >= minimum_full_gain)
        & (table.delta_lca_spearman >= minimum_lca_gain)
    ].copy()
    selected = selected.sort_values(
        ["delta_t_measure_reduced", "delta_t_measure_full",
         "delta_lca_spearman", "piece"],
        ascending=[False, False, False, True])
    if len(selected) < top_k:
        raise RuntimeError(
            f"only {len(selected)} cases satisfy the declared selection rule; "
            f"requested {top_k}")
    return selected.head(top_k).reset_index(drop=True)


def _load_scorer(parameters: dict) -> BoundaryContrastScorer:
    feature = parameters["feature_weights"]
    saved = feature["scaler"]
    names = tuple(saved["names"])
    scaler = RobustContrastScaler(
        names=names,
        medians=np.asarray(saved["medians"], dtype=float),
        mads=np.asarray(saved["mads"], dtype=float),
        scales=np.asarray(saved["scales"], dtype=float),
    )
    weights = np.asarray([feature["weights"][name] for name in names], dtype=float)
    return BoundaryContrastScorer(
        scaler=scaler, weights=weights, mode=feature["mode"],
        intercept=float(feature["intercept"]), gain=float(feature["gain"]))


def _piece_index(aligned_root: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(aligned_root.glob("*/*.json")):
        piece = SharedPiece.read_json(path)
        if piece.piece_id in result:
            raise ValueError(f"duplicate aligned piece id: {piece.piece_id}")
        result[piece.piece_id] = path
    return result


def _top_level_boundaries(piece: SharedPiece) -> list[float]:
    values = set()
    for item in piece.annotations:
        if (item.kind == "hierarchical_segment" and item.tree_id == "form"
                and item.level == 1):
            for value in (item.start_qb, item.end_qb):
                if value is not None and 0 < float(value) < float(piece.duration_qb):
                    values.add(float(value))
    return sorted(values)


def _plot_tree(axis, tree, scape, *, title: str, max_depth: int,
               samples: int, reference_boundaries: list[float]):
    import pitchscapes.plotting as plotting

    plotting.key_scape_plot(scape=scape, n_samples=samples, ax=axis)
    points, edges = collect_tree_geometry(tree, max_depth)
    total = float(tree.end - tree.start)
    for parent, child in edges:
        axis.plot(
            [(parent[0] - tree.start) / total, (child[0] - tree.start) / total],
            [parent[1] / total, child[1] / total],
            color="white", linewidth=1.05, alpha=0.82, zorder=5)
    for center, width, _ in points:
        axis.plot(
            (center - tree.start) / total, width / total, "o",
            color="black", markersize=4.4, markeredgecolor="white",
            markeredgewidth=0.7, zorder=10)
    for index, boundary in enumerate(reference_boundaries):
        axis.plot(
            (boundary - tree.start) / total, 0.018, marker="^",
            color="#FFD700", markeredgecolor="black", markeredgewidth=0.35,
            markersize=6, linestyle="None", zorder=12,
            label="expert level-1 boundary" if index == 0 else None)
    axis.set_title(title, fontsize=11)
    if reference_boundaries:
        axis.legend(loc="lower center", framealpha=0.82, fontsize=8)
    return len(points)


def render_case(row: pd.Series, args, piece_path: Path) -> dict:
    parameters_path = args.results_root / row.source / "model_parameters.json"
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    piece = SharedPiece.read_json(piece_path)
    notes = Path(piece.score_path)
    if not notes.is_absolute():
        notes = ROOT / notes
    matrix, bounds = load_pc_bins(notes, parameters["configuration"]["bin_size"])
    bounds = np.asarray(bounds, dtype=float)
    edges = np.r_[bounds[0, 0], bounds[:, 1]]
    specifications, estimated_key = distance_specs(matrix.sum(axis=0))
    names = tuple(parameters["distance_names"])
    contexts = tuple(parameters["configuration"]["contexts"])
    features = boundary_contrast_features(
        matrix, specifications, contexts=contexts, distances=names)
    scorer = _load_scorer(parameters)
    contrast = scorer.score(features)
    weights = [parameters["distance_weights"][name] for name in names]
    scales = [parameters["representation_scales"][name] for name in names]
    distance = make_parametric_distance(
        {"distance_specs": specifications}, names, weights, scales)
    affinity, affinity_metadata = pairwise_affinity(
        matrix, distance, scale=float(parameters["kernel_scale"]),
        context_radius=int(parameters["configuration"]["affinity_context_radius"]))
    contrast_weight = float(parameters["selected_contrast_weight"])
    balance_weight = float(parameters["selected_balance_weight"])
    greedy = greedy_boundary_aware_tree(
        matrix, bounds, affinity, contrast, ClusterNode,
        contrast_weight=contrast_weight, balance_weight=balance_weight)
    dp, diagnostics = optimal_boundary_aware_tree(
        matrix, bounds, affinity, contrast, ClusterNode,
        contrast_weight=contrast_weight, balance_weight=balance_weight,
        max_bins=int(parameters["configuration"]["max_bins"]))
    objectives = {
        search: boundary_aware_tree_objective(
            tree, affinity, contrast, contrast_weight=contrast_weight,
            balance_weight=balance_weight).total_objective
        for search, tree in (("greedy", greedy), ("dp", dp))
    }
    if objectives["dp"] < objectives["greedy"] - 1e-8:
        raise AssertionError("DP objective is below Greedy")
    computed = {
        "greedy": tree_measure(greedy, piece),
        "dp": tree_measure(dp, piece),
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pitchscapes.scapes import PitchScape

    # Keep interval mass so Pitch Scapes' default pseudocount does not swamp
    # an already L1-normalised window and make key colours artificially pale.
    scape = PitchScape(values=matrix, times=edges, normalise=False)
    references = _top_level_boundaries(piece)
    figure, axes = plt.subplots(1, 2, figsize=(19, 7.2))
    plotted = {}
    for axis, search, tree in zip(axes, ("greedy", "dp"), (greedy, dp)):
        metric = computed[search]
        title = (
            f"{search.upper()} | reduced T={metric['t_measure_reduced']:.3f}, "
            f"full T={metric['t_measure_full']:.3f}, "
            f"LCA rho={metric['lca_spearman']:.3f}")
        plotted[search] = _plot_tree(
            axis, tree, scape, title=title, max_depth=args.max_depth,
            samples=args.samples, reference_boundaries=references)
    figure.suptitle(
        f"{piece.piece_id}: held-out trained parametric distance — "
        f"Greedy vs exact DP", fontsize=15)
    figure.text(
        0.5, 0.018,
        f"trained on {parameters['training_dataset']}; "
        f"DP minus Greedy reduced T={row.delta_t_measure_reduced:+.3f}, "
        f"full T={row.delta_t_measure_full:+.3f}, "
        f"LCA rho={row.delta_lca_spearman:+.3f}",
        ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    safe = "".join(character if character.isalnum() else "_"
                   for character in piece.piece_id).strip("_")
    output = args.output_dir / f"{safe}_parametric_greedy_vs_dp.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    individual_outputs = {}
    for search, tree in (("greedy", greedy), ("dp", dp)):
        metric = computed[search]
        single, axis = plt.subplots(figsize=(14, 8))
        _plot_tree(
            axis, tree, scape,
            title=(
                f"{piece.piece_id}: {search.upper()} with held-out trained "
                f"parametric distance\n"
                f"reduced T={metric['t_measure_reduced']:.3f}, "
                f"full T={metric['t_measure_full']:.3f}, "
                f"LCA rho={metric['lca_spearman']:.3f}"),
            max_depth=args.max_depth, samples=args.samples,
            reference_boundaries=references)
        single.tight_layout()
        single_output = (
            args.output_dir / f"{safe}_parametric_{search}.png")
        single.savefig(single_output, dpi=args.dpi, bbox_inches="tight")
        plt.close(single)
        individual_outputs[search] = str(single_output)
    metadata = {
        "piece": piece.piece_id,
        "work": piece.work_id,
        "dataset": piece.dataset,
        "held_out_result_source": row.source,
        "training_dataset": parameters["training_dataset"],
        "test_dataset": parameters["held_out_test_dataset"],
        "test_labels_used_for_training_or_selection": False,
        "expert_boundaries_used_for_tree": False,
        "expert_boundaries_used_for_plot_and_evaluation": True,
        "selection_rule": {
            "rank": "descending DP-minus-Greedy reduced T-measure",
            "minimum_full_t_gain": args.minimum_full_gain,
            "minimum_lca_spearman_gain": args.minimum_lca_gain,
            "requires_positive_reduced_t_gain": True,
            "requires_nonnegative_objective_gap": True,
        },
        "recorded_metrics": {
            key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in row.items()
            if key.startswith(("t_measure_", "lca_", "objective_", "delta_"))
        },
        "recomputed_metrics": computed,
        "objective": {
            "name": parameters["optimality_scope"],
            "greedy": float(objectives["greedy"]),
            "dp": float(objectives["dp"]),
            "dp_minus_greedy": float(objectives["dp"] - objectives["greedy"]),
            "dp_root_split": int(diagnostics.root_split),
        },
        "model": {
            "distance_names": list(names),
            "distance_weights": parameters["distance_weights"],
            "representation_scales": parameters["representation_scales"],
            "kernel_scale": parameters["kernel_scale"],
            "contrast_weight": contrast_weight,
            "balance_weight": balance_weight,
            "bin_size_qb": parameters["configuration"]["bin_size"],
            "estimated_key_from_piece_features": estimated_key,
            "affinity_metadata": affinity_metadata,
        },
        "tree_shape": {
            "greedy": tree_shape_diagnostics(greedy),
            "dp": tree_shape_diagnostics(dp),
        },
        "visualization": {
            "max_depth": args.max_depth,
            "plotted_nodes": plotted,
            "expert_level_1_boundaries_qb": references,
            "pitch_scape_source": "same 8-qb pitch-class bins used by clustering",
            "individual_outputs": individual_outputs,
        },
        "output": str(output),
    }
    for search, individual_path in individual_outputs.items():
        single_metadata = {
            **metadata,
            "search": search,
            "output": individual_path,
            "paired_comparison_output": str(output),
            "reported_metrics": computed[search],
        }
        Path(individual_path).with_suffix(".json").write_text(
            json.dumps(single_metadata, indent=2), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    return metadata


def main():
    args = parse_args()
    frames = []
    for source in FORMAL_RESULTS:
        path = args.results_root / source / "held_out_per_piece.csv"
        frame = pd.read_csv(path)
        frames.append(frame.assign(source=source))
    table = paired_case_table(frames)
    selected = select_cases(
        table, top_k=args.top_k,
        minimum_full_gain=args.minimum_full_gain,
        minimum_lca_gain=args.minimum_lca_gain)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output_dir / "case_selection.csv", index=False)
    index = _piece_index(args.aligned_root)
    records = []
    for _, row in selected.iterrows():
        if row.piece not in index:
            raise FileNotFoundError(f"aligned annotation not found: {row.piece}")
        records.append(render_case(row, args, index[row.piece]))
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    lines = [
        "# Parametric Greedy–DP case studies",
        "",
        "These held-out examples were selected deterministically: DP must improve "
        "the optimized objective, reduced/full T-measure and LCA Spearman "
        "correlation; full-T gain must be at least "
        f"{args.minimum_full_gain:g} and LCA gain at least "
        f"{args.minimum_lca_gain:g}. Cases are ranked by reduced T-measure gain.",
        "",
        "They illustrate tree differences consistent with Greedy being unable "
        "to revise earlier merges. They do not by themselves prove that every "
        "difference was caused by one identifiable early merge and do not "
        "replace corpus-level statistics.",
        "",
        "| Piece | Training corpus | Reduced T gain | Full T gain | LCA rho gain |",
        "|---|---|---:|---:|---:|",
    ]
    for record, (_, row) in zip(records, selected.iterrows()):
        lines.append(
            f"| {record['piece']} | {record['training_dataset']} | "
            f"{row.delta_t_measure_reduced:+.4f} | "
            f"{row.delta_t_measure_full:+.4f} | "
            f"{row.delta_lca_spearman:+.4f} |")
    (args.output_dir / "CASE_STUDIES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
