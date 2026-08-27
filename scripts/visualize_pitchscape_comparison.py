#!/usr/bin/env python
"""Render five pre-registered ABC works as Greedy/DP/DL Pitch Scapes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from greedy_evaluation import dcml_localkey_segments
from pitchscape_comparison import (
    METHOD_LABELS,
    SELECTED_PIECES,
    build_comparison_trees,
    draw_tree_on_pitchscape,
    load_formal_metric_checkpoint,
    make_pitchscape,
    select_resolution,
    tree_node_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("results/dissertation_main/deep"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dissertation_main/pitch_scapes"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--max-leaves", type=int, default=192)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--bin-sizes", nargs="+", type=float,
                        default=[8.0, 16.0, 32.0])
    parser.add_argument(
        "--pieces", nargs="+",
        choices=[piece.piece_id for piece in SELECTED_PIECES],
        help="Optional subset of the five pre-registered pieces")
    return parser.parse_args()


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _save_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default), encoding="utf-8")


def _piece_metadata(data_root: Path) -> dict[str, dict]:
    path = data_root / "metadata.tsv"
    if not path.is_file():
        raise FileNotFoundError(f"ABC metadata is missing: {path}")
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    return {str(row["piece"]): row.to_dict() for _, row in frame.iterrows()}


def _split_roles(checkpoint_dir: Path) -> dict[str, str]:
    path = checkpoint_dir / "data_split.csv"
    if not path.is_file():
        raise FileNotFoundError(f"formal data split is missing: {path}")
    frame = pd.read_csv(path)
    if not {"piece", "split"} <= set(frame.columns):
        raise ValueError(f"invalid formal data split: {path}")
    return {str(row.piece): str(row.split) for row in frame.itertuples()}


def _save_method_figure(
    output: Path,
    *,
    display_name: str,
    method: str,
    tree,
    scape,
    bin_size: float,
    checkpoint_seed: int,
    max_depth: int,
    samples: int,
    dpi: int,
) -> tuple[int, int]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(14, 8))
    counts = draw_tree_on_pitchscape(
        axis, scape, tree, method=method, max_depth=max_depth,
        n_samples=samples)
    suffix = f"; checkpoint seed {checkpoint_seed}" if method == "dl" else ""
    figure.suptitle(
        f"{display_name}\n{bin_size:g}-quarter-beat leaves{suffix}", fontsize=12)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return counts


def _save_comparison_figure(
    output: Path,
    *,
    display_name: str,
    trees: dict,
    scape,
    expert_boundaries: list[float],
    bin_size: float,
    max_depth: int,
    samples: int,
    dpi: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(24, 7), squeeze=False)
    for axis, method in zip(axes[0], ("greedy", "dp", "dl")):
        draw_tree_on_pitchscape(
            axis, scape, trees[method], method=method, max_depth=max_depth,
            n_samples=samples, expert_boundaries_qb=expert_boundaries)
    figure.suptitle(
        f"{display_name} — method comparison at {bin_size:g} quarter beats\n"
        "Gold dashed lines: ABC local-key reference boundaries",
        fontsize=13,
    )
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.max_depth < 0:
        raise SystemExit("--max-depth must be non-negative")
    if args.samples < 2 or args.dpi < 1:
        raise SystemExit("--samples must be >=2 and --dpi must be positive")
    data_root = args.data_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()

    # This is intentionally performed before any annotation is read.
    try:
        checkpoint = load_formal_metric_checkpoint(checkpoint_dir, args.device)
        roles = _split_roles(checkpoint_dir)
        metadata = _piece_metadata(data_root)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    selected_ids = set(args.pieces or [piece.piece_id for piece in SELECTED_PIECES])
    selected = [piece for piece in SELECTED_PIECES if piece.piece_id in selected_ids]
    if not selected:
        raise SystemExit("no pieces were selected")
    missing_roles = [piece.piece_id for piece in selected if piece.piece_id not in roles]
    if missing_roles:
        raise RuntimeError(
            "selected pieces are absent from the formal split: " + ", ".join(missing_roles))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    for piece in selected:
        notes_path = data_root / "notes" / f"{piece.piece_id}.notes.tsv"
        harmony_path = data_root / "harmonies" / f"{piece.piece_id}.harmonies.tsv"
        if not notes_path.is_file() or not harmony_path.is_file():
            raise FileNotFoundError(
                f"paired ABC files are missing for {piece.piece_id}")

        matrix, bounds, bin_size = select_resolution(
            notes_path, args.bin_sizes, args.max_leaves)
        trees, diagnostics = build_comparison_trees(
            matrix, bounds, checkpoint.distance, max_leaves=args.max_leaves)
        scape = make_pitchscape(notes_path)

        # Annotation access occurs only after all three inference trees exist.
        segments, _ = dcml_localkey_segments(harmony_path)
        expert_boundaries = [
            float(end) for _, end, _ in segments[:-1]
            if float(trees["dp"].start) < float(end) < float(trees["dp"].end)
        ]
        piece_dir = output_dir / piece.directory_name
        piece_dir.mkdir(parents=True, exist_ok=True)
        filenames = {
            "greedy": "greedy_key_profile.png",
            "dp": "dp_key_profile.png",
            "dl": "dl_harmonic_cnn_dp.png",
        }
        plotted = {}
        for method, filename in filenames.items():
            plotted[method] = _save_method_figure(
                piece_dir / filename,
                display_name=piece.display_name,
                method=method,
                tree=trees[method],
                scape=scape,
                bin_size=bin_size,
                checkpoint_seed=checkpoint.seed,
                max_depth=args.max_depth,
                samples=args.samples,
                dpi=args.dpi,
            )
        _save_comparison_figure(
            piece_dir / "comparison_with_expert_boundaries.png",
            display_name=piece.display_name,
            trees=trees,
            scape=scape,
            expert_boundaries=expert_boundaries,
            bin_size=bin_size,
            max_depth=args.max_depth,
            samples=args.samples,
            dpi=args.dpi,
        )

        node_rows = [
            row for method in ("greedy", "dp", "dl")
            for row in tree_node_rows(trees[method], method)
        ]
        pd.DataFrame(node_rows).to_csv(piece_dir / "tree_nodes.csv", index=False)
        source = metadata.get(piece.piece_id, {})
        piece_record = {
            "piece_id": piece.piece_id,
            "display_name": piece.display_name,
            "selection_rationale": piece.rationale,
            "formal_split": roles[piece.piece_id],
            "annotated_key": source.get("annotated_key", ""),
            "composed_start": source.get("composed_start", ""),
            "composed_end": source.get("composed_end", ""),
            "notes_path": str(notes_path.resolve()),
            "harmonies_path": str(harmony_path.resolve()),
            "bin_size_qb": bin_size,
            "leaf_count": len(matrix),
            "expert_boundary_count": len(expert_boundaries),
            "plotted_max_depth": args.max_depth,
            "plotted_counts": {
                method: {"nodes": counts[0], "edges": counts[1]}
                for method, counts in plotted.items()
            },
            "methods": METHOD_LABELS,
            "diagnostics": diagnostics,
            "checkpoint": {
                "seed": checkpoint.seed,
                "metric_validation_work_macro_ap": checkpoint.validation_ap,
                "path": str(checkpoint.path),
                "sha256": checkpoint.sha256,
                "encoder_config": checkpoint.encoder_config,
            },
        }
        _save_json(piece_dir / "metadata.json", piece_record)
        selection_rows.append({
            "piece": piece.piece_id,
            "display_name": piece.display_name,
            "directory": piece.directory_name,
            "formal_split": roles[piece.piece_id],
            "selection_rationale": piece.rationale,
            "bin_size_qb": bin_size,
            "leaf_count": len(matrix),
            "checkpoint_seed": checkpoint.seed,
        })
        print(f"[pitch-scape] {piece.display_name}: {piece_dir}")

    pd.DataFrame(selection_rows).to_csv(
        output_dir / "selection_manifest.csv", index=False)
    _save_json(output_dir / "run_config.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "Five musicologically pre-registered ABC examples; no model-result-based selection"),
        "methods": METHOD_LABELS,
        "primary_figures_include_annotations": False,
        "comparison_figure_annotation": "ABC local-key boundaries, post-inference only",
        "resolution_candidates_qb": args.bin_sizes,
        "max_leaves": args.max_leaves,
        "max_depth": args.max_depth,
        "pitchscape_source": "ABC notes TSV exact event timeline; no MIDI/MuseScore export",
        "checkpoint_seed_selection": "highest metric validation work-macro AP; lower seed breaks ties",
        "checkpoint_seed": checkpoint.seed,
        "checkpoint_sha256": checkpoint.sha256,
        "pieces": [piece.piece_id for piece in selected],
    })


if __name__ == "__main__":
    main()
