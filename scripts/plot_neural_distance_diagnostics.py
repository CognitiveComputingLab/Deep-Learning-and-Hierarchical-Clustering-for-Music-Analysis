#!/usr/bin/env python
"""Plot learned-distance separation and tree-shape diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distance_distributions(frame: pd.DataFrame, output: Path) -> None:
    required = {"family", "label", "distance"}
    if frame.empty or not required <= set(frame):
        raise ValueError("learned-distance diagnostics are empty or incomplete")
    families = sorted(frame.family.unique())
    figure, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 4),
                                squeeze=False)
    for axis, family in zip(axes[0], families):
        group = frame[frame.family == family]
        for label, name, colour in ((0, "non-boundary", "#377eb8"),
                                    (1, "boundary", "#e41a1c")):
            values = group.loc[group.label == label, "distance"].to_numpy(float)
            if len(values):
                axis.hist(values, bins=20, alpha=0.55, density=True,
                          label=name, color=colour)
        axis.set(xlabel="Learned interval distance", ylabel="Density", title=family)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_tree_shape(frame: pd.DataFrame, output: Path) -> None:
    columns = ["normalized_colless_index", "normalized_sackin_index",
               "normalized_max_depth"]
    required = {"family", "search", *columns}
    if frame.empty or not required <= set(frame):
        raise ValueError("tree diagnostics are empty or incomplete")
    shape = frame.groupby(["family", "search"], as_index=False)[columns].mean()
    labels = [f"{row.family}\n{row.search}" for row in shape.itertuples()]
    positions = np.arange(len(shape))
    figure, axis = plt.subplots(figsize=(10, 5))
    width = 0.25
    axis.bar(positions - width, shape.normalized_colless_index,
             width, label="normalised Colless")
    axis.bar(positions, shape.normalized_sackin_index,
             width, label="normalised Sackin")
    axis.bar(positions + width, shape.normalized_max_depth,
             width, label="normalised max depth")
    axis.set_xticks(positions, labels, rotation=15)
    axis.set(ylabel="Mean normalised tree-shape statistic",
             title="Tree-shape diagnostics")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path,
                        nargs="?", default=Path("results/neural_distance_stage"))
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    distances = pd.read_csv(result_dir / "learned_distance_diagnostics.csv")
    trees = pd.read_csv(result_dir / "tree_diagnostics.csv")
    plot_distance_distributions(
        distances, result_dir / "learned_distance_distributions.png")
    plot_tree_shape(trees, result_dir / "tree_shape.png")
    print(f"Saved diagnostic figures in {result_dir}")


if __name__ == "__main__":
    main()
