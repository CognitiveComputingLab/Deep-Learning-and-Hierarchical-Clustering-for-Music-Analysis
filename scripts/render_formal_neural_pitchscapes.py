"""Render leakage-free Advanced Greedy/DP Pitch Scapes for one held-out work.

Unlike the legacy qualitative renderer, this command reconstructs the formal
boundary-aware objective.  It loads the outer-refit neural checkpoint, refits
the contrast scorer on the other works only, applies the validation-selected
operating point, and verifies the reconstructed objective against the stored
held-out diagnostics before saving figures.
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
    build_trees,
    discover_pairs,
    fit_contrast_models,
    items_for_works,
    load_corpus,
    work_id,
)
from evaluate_neural_distance_stage import family_items
from greedy_evaluation import collect_salient_splits, tree_shape_diagnostics
from neural_distance import NeuralPitchClassDistance


FAMILY = "neural_circular_siamese"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=ROOT / "results/dissertation_main/neural")
    parser.add_argument("--data-root", type=Path, default=ROOT / "external/ABC")
    parser.add_argument("--outer-work", default="n11op95")
    parser.add_argument("--piece", default="n11op95_01")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--midi", type=Path, default=ROOT / "n11op95_01.mid")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/paper_pitchscapes")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def scientific_configuration(results_dir: Path):
    metadata_path = results_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    configuration = metadata.get("scientific_configuration")
    if configuration is not None:
        return configuration
    marker_paths = sorted((results_dir / "shards").glob("*/complete.json"))
    if not marker_paths:
        raise FileNotFoundError("No metadata configuration or shard markers found")
    return json.loads(marker_paths[0].read_text(encoding="utf-8"))["configuration"]


def selected_row(results_dir: Path, outer_work: str):
    frame = pd.read_csv(results_dir / "outer_selections.csv")
    rows = frame[
        frame.outer_work.eq(outer_work)
        & frame.family.eq(FAMILY)
        & frame.selection_mode.eq("threshold")]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one formal threshold selection for {outer_work}, found {len(rows)}")
    return rows.iloc[0]


def checkpoint_path(results_dir: Path, outer_work: str, seed: int):
    path = (results_dir / "shards" / outer_work / "checkpoints" /
            f"{outer_work}_{FAMILY}_seed{seed}.npz")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def stored_diagnostic(results_dir: Path, outer_work: str, piece: str,
                      seed: int, search: str):
    frame = pd.read_csv(results_dir / "tree_diagnostics.csv")
    rows = frame[
        frame.outer_work.eq(outer_work)
        & frame.piece.eq(piece)
        & frame.family.eq(FAMILY)
        & frame.selection_mode.eq("threshold")
        & frame.training_seed.eq(seed)
        & frame.search.eq(search)]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one stored diagnostic for {piece}/{search}, found {len(rows)}")
    return rows.iloc[0]


def stored_metrics(results_dir: Path, outer_work: str, piece: str,
                   seed: int, search: str, tolerance: float):
    frame = pd.read_csv(results_dir / "held_out_per_piece.csv")
    rows = frame[
        frame.outer_work.eq(outer_work)
        & frame.piece.eq(piece)
        & frame.family.eq(FAMILY)
        & frame.selection_mode.eq("threshold")
        & frame.training_seed.eq(seed)
        & frame.search.eq(search)
        & np.isclose(frame.tolerance_qb, tolerance)
        & ~frame.oracle]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one stored metric row for {piece}/{search}, found {len(rows)}")
    return rows.iloc[0]


def render(args, tree, info, contrast, choice, search, checkpoint, scale,
           stored_info, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pitchscapes.plotting as plotting
    import pitchscapes.reader as reader

    points, edges = collect_tree_geometry(tree, args.max_depth)
    total = float(tree.end - tree.start)
    predicted = collect_salient_splits(
        tree, contrast,
        threshold=float(choice.selected_value),
        min_separation_qb=float(choice.selected_min_separation_qb),
        local_max_radius_qb=float(choice.selected_local_max_radius_qb))

    figure, axis = plt.subplots(figsize=(14, 8))
    plotting.key_scape_plot(
        scape=reader.get_pitch_scape(str(args.midi)),
        n_samples=args.samples, ax=axis)
    for parent, child in edges:
        axis.plot(
            [(parent[0] - tree.start) / total, (child[0] - tree.start) / total],
            [parent[1] / total, child[1] / total],
            color="white", linewidth=1.35, alpha=0.9, zorder=5)
    for center, width, _ in points:
        axis.plot(
            (center - tree.start) / total, width / total, "o",
            color="black", markersize=5, markeredgecolor="white",
            markeredgewidth=0.8, zorder=10)
    for boundary in predicted:
        axis.plot(
            (float(boundary) - tree.start) / total, 0.008, marker="^",
            color="#FFD700", markeredgecolor="black", markersize=7,
            linestyle="none", zorder=12)
    search_label = "Greedy" if search == "greedy" else "Exact DP"
    axis.set_title(
        f"{args.piece} - Advanced neural distance + {search_label} "
        f"(validation-selected boundary-aware objective)")
    output = args.output_dir / f"{args.piece}_advanced_neural_{search}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)

    objective_error = abs(float(info.total_objective) - float(stored_info.objective))
    if objective_error > 1e-8:
        raise AssertionError(
            f"Reconstructed {search} objective differs from formal result by "
            f"{objective_error:.3g}")
    if len(predicted) != int(metrics.predicted_boundary_count):
        raise AssertionError(
            f"Reconstructed {search} predicts {len(predicted)} boundaries; "
            f"formal result stores {int(metrics.predicted_boundary_count)}")
    metadata = {
        "piece": args.piece,
        "outer_work": args.outer_work,
        "family": FAMILY,
        "search": search,
        "training_seed": args.seed,
        "checkpoint": str(checkpoint.resolve()),
        "training_scope": f"all works except held-out {args.outer_work}",
        "training_affinity_scale": float(scale),
        "selection_mode": "nested-validation shared threshold",
        "selected_checkpoint_epoch": int(choice.selected_checkpoint_epoch),
        "contrast_weight": float(choice.selected_contrast_weight),
        "balance_weight": float(choice.selected_balance_weight),
        "threshold": float(choice.selected_value),
        "min_separation_qb": float(choice.selected_min_separation_qb),
        "local_max_radius_qb": float(choice.selected_local_max_radius_qb),
        "objective": float(info.total_objective),
        "stored_objective": float(stored_info.objective),
        "objective_reconstruction_error": objective_error,
        "root_split": int(info.root_split),
        "tree_shape": tree_shape_diagnostics(tree),
        "predicted_boundaries_qb": [float(value) for value in predicted],
        "predicted_boundary_count": len(predicted),
        "tolerance_qb": 8.0,
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "bin_size_qb": 8.0,
        "max_depth_displayed": args.max_depth,
        "dcml_used_for_tree_construction": False,
        "dcml_used_for_training_other_works": True,
        "gold_triangles": False,
        "gold_coloured_triangles": "predicted boundaries, not ground truth",
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    return metadata


def main():
    args = parse_args()
    if not args.midi.is_file():
        raise FileNotFoundError(args.midi)
    configuration = scientific_configuration(args.results_dir)
    if configuration.get("quick"):
        raise RuntimeError("Formal figures cannot be generated from a quick run")
    if not np.isclose(float(configuration["bin_size"]), 8.0):
        raise RuntimeError("Expected the formal 8-qb configuration")
    config_args = SimpleNamespace(**configuration)
    config_args.learning_rate = float(configuration["contrast_learning_rate"])
    config_args.learning_rate_contrast = config_args.learning_rate

    pairs = discover_pairs(args.data_root, configuration.get("piece"))
    cache, statuses, _ = load_corpus(pairs, config_args)
    failures = [row for row in statuses if row["status"] != "success"]
    if failures:
        raise RuntimeError(f"Corpus reconstruction failed: {failures}")
    if args.piece not in cache or cache[args.piece]["work"] != args.outer_work:
        raise RuntimeError("Selected piece does not belong to the held-out work")
    works = sorted({item["work"] for item in cache.values()})
    train_works = [work for work in works if work != args.outer_work]
    base_train = items_for_works(cache, train_works)
    base_test = [cache[args.piece]]

    checkpoint = checkpoint_path(args.results_dir, args.outer_work, args.seed)
    distance = NeuralPitchClassDistance.load(checkpoint)
    current_train, current_test, _, scale = family_items(
        base_train, base_test, FAMILY, distance, config_args)
    contrast_models, _, _ = fit_contrast_models(current_train, config_args)
    scorer = contrast_models["supervised_contrast"]
    choice = selected_row(args.results_dir, args.outer_work)
    item = current_test[0]
    contrast = scorer.score(item["features"])
    trees = build_trees(
        item, contrast,
        float(choice.selected_contrast_weight),
        float(choice.selected_balance_weight), config_args)

    records = []
    for search, tree, info, _ in trees:
        stored_info = stored_diagnostic(
            args.results_dir, args.outer_work, args.piece, args.seed, search)
        metrics = stored_metrics(
            args.results_dir, args.outer_work, args.piece,
            args.seed, search, tolerance=8.0)
        records.append(render(
            args, tree, info, contrast, choice, search, checkpoint, scale,
            stored_info, metrics))
    if records[1]["objective"] < records[0]["objective"] - 1e-8:
        raise AssertionError("Exact DP objective is below Greedy")
    manifest = args.output_dir / f"{args.piece}_advanced_neural_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {manifest}")


if __name__ == "__main__":
    main()
