#!/usr/bin/env python
"""Render reproducible representative Greedy and exact-DP pitch-scapes.

Neural figures require a frozen checkpoint.  They use the additive distance
objective for visualization and are explicitly labelled diagnostics unless a
scorer-aware Advanced result is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp_clustering import optimal_adjacent_binary_tree
from dp_pitchscape import collect_tree_geometry
from greedy_clustering import ClusterNode, distance_specs, greedy_cluster, load_pc_bins

DISTANCE_LABELS = {
    "key_profile": "key-profile distance",
    "circle_of_fifths": "circle-of-fifths embedding distance",
    "tonic_weighted": "tonic-relative weighted distance",
    "euclidean": "Euclidean distance",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["greedy", "dp", "neural", "all"], default="all")
    parser.add_argument("--distance", choices=sorted(DISTANCE_LABELS), default="key_profile")
    parser.add_argument("--midi", type=Path, default=ROOT / "n11op95_01.mid")
    parser.add_argument("--notes", type=Path,
                        default=ROOT / "external/ABC/notes/n11op95_01.notes.tsv")
    parser.add_argument("--neural-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_pitchscapes")
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def render_one(args, method: str, search: str) -> dict:
    if not args.midi.is_file():
        raise FileNotFoundError(f"MIDI file not found: {args.midi}")
    if not args.notes.is_file():
        raise FileNotFoundError(f"notes file not found: {args.notes}")
    matrix, bounds = load_pc_bins(args.notes, args.bin_size)
    if len(matrix) > args.max_bins:
        raise ValueError(f"{len(matrix)} bins exceeds max_bins={args.max_bins}")
    distances, estimated_key = distance_specs(matrix.sum(axis=0))
    if method == "neural":
        if args.neural_checkpoint is None or not args.neural_checkpoint.is_file():
            raise FileNotFoundError("--neural-checkpoint is required for neural figures")
        from neural_distance import NeuralPitchClassDistance
        distance = NeuralPitchClassDistance.load(args.neural_checkpoint)
        method_label = "neural frozen-distance"
        objective = "additive ordered child distance (diagnostic)"
    else:
        distance = distances[args.distance]
        method_label = f"{method} / {DISTANCE_LABELS[args.distance]}"
        objective = "additive ordered child distance"
    if search == "greedy":
        tree = greedy_cluster(matrix, bounds, distance)
        root_split = None
    else:
        tree, diagnostics = optimal_adjacent_binary_tree(
            matrix, bounds, distance, ClusterNode,
            balance_lambda=0.0, max_bins=args.max_bins)
        root_split = diagnostics.root_split

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pitchscapes.plotting as plotting
    import pitchscapes.reader as reader

    points, edges = collect_tree_geometry(tree, args.max_depth)
    total = float(tree.end - tree.start)
    if total <= 0:
        raise ValueError("tree duration must be positive")
    figure, axis = plt.subplots(figsize=(14, 8))
    plotting.key_scape_plot(scape=reader.get_pitch_scape(str(args.midi)),
                            n_samples=args.samples, ax=axis)
    for parent, child in edges:
        axis.plot([(parent[0] - tree.start) / total, (child[0] - tree.start) / total],
                  [parent[1] / total, child[1] / total], "w-", linewidth=1.2,
                  alpha=0.8, zorder=5)
    for center, width, _ in points:
        axis.plot((center - tree.start) / total, width / total, "o", color="black",
                  markersize=5, markeredgecolor="white", markeredgewidth=0.8, zorder=10)
    piece = args.notes.name.replace(".notes.tsv", "")
    axis.set_title(f"{piece} - {method_label} {search} tree through depth {args.max_depth}")
    output = args.output_dir / f"{piece}_{method}_{search}_{args.distance}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    metadata = {
        "piece": piece, "method": method, "search": search,
        "distance": args.distance, "distance_label": method_label,
        "objective": objective, "bin_size_qb": args.bin_size,
        "max_depth": args.max_depth, "leaf_count": len(matrix),
        "plotted_node_count": len(points), "root_split": root_split,
        "estimated_key": estimated_key if method != "neural" else None,
        "midi": str(args.midi), "notes": str(args.notes),
        "neural_checkpoint": str(args.neural_checkpoint) if method == "neural" else None,
        "output": str(output), "dcml_used_for_tree": False,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    return metadata


def main():
    args = parse_args()
    methods = [args.method] if args.method != "all" else ["greedy", "dp"]
    records = [render_one(args, method, search)
               for method in methods for search in ("greedy", "dp")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
