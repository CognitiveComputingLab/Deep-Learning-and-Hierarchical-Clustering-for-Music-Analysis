"""Render a formal three-panel held-out Pitch-Scape case comparison.

The three separate outputs are handcrafted key-profile + exact DP, frozen
neural distance + Greedy, and frozen neural distance + exact DP.  All trees
use the same held-out piece, 8-qb bins, training-only contrast scorers and the
outer-fold validation-selected objective/operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dp_pitchscape import collect_tree_geometry
from evaluate_boundary_aware_stage import (
    build_trees, discover_pairs, fit_contrast_models, items_for_works,
    load_corpus,
)
from evaluate_neural_distance_stage import family_items
from greedy_evaluation import collect_salient_splits, tree_shape_diagnostics
from neural_distance import NeuralPitchClassDistance

KEY_FAMILY = "handcrafted_key_profile"
NEURAL_FAMILY = "neural_circular_siamese"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/dissertation_main/neural")
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/ABC")
    parser.add_argument("--outer-work", required=True)
    parser.add_argument("--piece", required=True)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/paper_pitchscapes/cases")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def configuration(results_dir: Path):
    metadata = json.loads(
        (results_dir / "metadata.json").read_text(encoding="utf-8"))
    value = metadata.get("scientific_configuration")
    if value is not None:
        return value
    marker = results_dir / "shards" / "n01op18-1" / "complete.json"
    return json.loads(marker.read_text(encoding="utf-8"))["configuration"]


def selection(results_dir: Path, outer_work: str, family: str):
    frame = pd.read_csv(results_dir / "outer_selections.csv")
    rows = frame[
        frame.outer_work.eq(outer_work) & frame.family.eq(family)
        & frame.selection_mode.eq("threshold")]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one threshold selection for {outer_work}/{family}")
    return rows.iloc[0]


def stored_rows(results_dir: Path, outer_work: str, piece: str,
                family: str, seed: int, search: str):
    diagnostics = pd.read_csv(results_dir / "tree_diagnostics.csv")
    drow = diagnostics[
        diagnostics.outer_work.eq(outer_work)
        & diagnostics.piece.eq(piece) & diagnostics.family.eq(family)
        & diagnostics.selection_mode.eq("threshold")
        & diagnostics.training_seed.eq(seed) & diagnostics.search.eq(search)]
    metrics = pd.read_csv(results_dir / "held_out_per_piece.csv")
    mrow = metrics[
        metrics.outer_work.eq(outer_work) & metrics.piece.eq(piece)
        & metrics.family.eq(family) & metrics.selection_mode.eq("threshold")
        & metrics.training_seed.eq(seed) & metrics.search.eq(search)
        & np.isclose(metrics.tolerance_qb, 8.0) & ~metrics.oracle]
    if len(drow) != 1 or len(mrow) != 1:
        raise RuntimeError(f"Stored formal row missing for {piece}/{family}/{search}")
    return drow.iloc[0], mrow.iloc[0]


def make_scape(item):
    from pitchscapes.scapes import PitchScape
    bounds = np.asarray(item["bounds"], dtype=float)
    edges = np.r_[bounds[0, 0], bounds[:, 1]]
    return PitchScape(values=item["matrix"], times=edges, normalise=False)


def render(args, item, tree, info, contrast, choice, family, search,
           drow, mrow, checkpoint=None, scale=np.nan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pitchscapes.plotting as plotting

    error = abs(float(info.total_objective) - float(drow.objective))
    if error > 1e-8:
        raise AssertionError(f"Objective reconstruction error {error:.3g}")
    predicted = collect_salient_splits(
        tree, contrast, threshold=float(choice.selected_value),
        min_separation_qb=float(choice.selected_min_separation_qb),
        local_max_radius_qb=float(choice.selected_local_max_radius_qb))
    if len(predicted) != int(mrow.predicted_boundary_count):
        raise AssertionError("Predicted-boundary count differs from formal result")

    points, edges = collect_tree_geometry(tree, args.max_depth)
    total = float(tree.end - tree.start)
    figure, axis = plt.subplots(figsize=(14, 8))
    plotting.key_scape_plot(scape=make_scape(item), n_samples=args.samples, ax=axis)
    for parent, child in edges:
        axis.plot(
            [(parent[0] - tree.start) / total, (child[0] - tree.start) / total],
            [parent[1] / total, child[1] / total], color="white",
            linewidth=1.35, alpha=0.9, zorder=5)
    for center, width, _ in points:
        axis.plot(
            (center - tree.start) / total, width / total, "o", color="black",
            markersize=5, markeredgecolor="white", markeredgewidth=0.8,
            zorder=10)
    for boundary in predicted:
        axis.plot(
            (float(boundary) - tree.start) / total, 0.008, marker="^",
            color="#FFD700", markeredgecolor="black", markersize=7,
            linestyle="none", zorder=12)
    family_label = "Key-profile" if family == KEY_FAMILY else "Neural"
    search_label = "Greedy" if search == "greedy" else "Exact DP"
    axis.set_title(
        f"{args.piece} - {family_label} + {search_label} "
        f"(held-out, validation-selected)")
    token = "keyprofile" if family == KEY_FAMILY else "neural"
    output = args.output_dir / f"{args.piece}_formal_{token}_{search}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    record = {
        "piece": args.piece, "outer_work": args.outer_work,
        "family": family, "search": search, "seed": args.seed,
        "training_scope": f"all works except {args.outer_work}",
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "training_affinity_scale": (
            None if not np.isfinite(scale) else float(scale)),
        "checkpoint_epoch": int(choice.selected_checkpoint_epoch),
        "contrast_weight": float(choice.selected_contrast_weight),
        "balance_weight": float(choice.selected_balance_weight),
        "threshold": float(choice.selected_value),
        "min_separation_qb": float(choice.selected_min_separation_qb),
        "local_max_radius_qb": float(choice.selected_local_max_radius_qb),
        "objective": float(info.total_objective),
        "objective_reconstruction_error": error,
        "precision": float(mrow.precision), "recall": float(mrow.recall),
        "f1": float(mrow.f1),
        "predicted_boundary_count": len(predicted),
        "predicted_boundaries_qb": [float(value) for value in predicted],
        "tree_shape": tree_shape_diagnostics(tree),
        "bin_size_qb": 8.0, "max_depth_displayed": args.max_depth,
        "gold_coloured_triangles": "predicted boundaries, not ground truth",
        "test_labels_used_for_tree_or_selection": False,
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    return record


def build_family(args, config_args, base_train, base_test, family):
    checkpoint = None
    scale = np.nan
    if family == KEY_FAMILY:
        current_train, current_test = base_train, base_test
    else:
        checkpoint = (args.results_dir / "shards" / args.outer_work /
                      "checkpoints" /
                      f"{args.outer_work}_{NEURAL_FAMILY}_seed{args.seed}.npz")
        distance = NeuralPitchClassDistance.load(checkpoint)
        current_train, current_test, _, scale = family_items(
            base_train, base_test, family, distance, config_args)
    models, _, _ = fit_contrast_models(current_train, config_args)
    scorer = models["supervised_contrast"]
    choice = selection(args.results_dir, args.outer_work, family)
    item = current_test[0]
    contrast = scorer.score(item["features"])
    trees = build_trees(
        item, contrast, float(choice.selected_contrast_weight),
        float(choice.selected_balance_weight), config_args)
    return item, contrast, trees, choice, checkpoint, scale


def main():
    args = parse_args()
    config = configuration(args.results_dir)
    if config.get("quick") or not np.isclose(float(config["bin_size"]), 8.0):
        raise RuntimeError("Expected the formal non-quick 8-qb experiment")
    config_args = SimpleNamespace(**config)
    config_args.learning_rate = float(config["contrast_learning_rate"])
    config_args.learning_rate_contrast = config_args.learning_rate
    cache, statuses, _ = load_corpus(
        discover_pairs(args.data_root, config.get("piece")), config_args)
    failures = [row for row in statuses if row["status"] != "success"]
    if failures:
        raise RuntimeError(f"Corpus reconstruction failed: {failures}")
    if args.piece not in cache or cache[args.piece]["work"] != args.outer_work:
        raise RuntimeError("Piece/outer-work mismatch")
    works = sorted({item["work"] for item in cache.values()})
    base_train = items_for_works(
        cache, [work for work in works if work != args.outer_work])
    base_test = [cache[args.piece]]

    records = []
    for family in (KEY_FAMILY, NEURAL_FAMILY):
        item, contrast, trees, choice, checkpoint, scale = build_family(
            args, config_args, base_train, base_test, family)
        selected_searches = ("dp",) if family == KEY_FAMILY else ("greedy", "dp")
        for search, tree, info, _ in trees:
            if search not in selected_searches:
                continue
            drow, mrow = stored_rows(
                args.results_dir, args.outer_work, args.piece,
                family, args.seed, search)
            records.append(render(
                args, item, tree, info, contrast, choice, family, search,
                drow, mrow, checkpoint, scale))
    manifest = args.output_dir / f"{args.piece}_formal_case_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
