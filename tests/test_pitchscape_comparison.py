"""Tests for the pre-registered Greedy/DP/DL Pitch Scape figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))

from dp_clustering import assert_valid_ordered_binary_tree
from neural_clustering import BoundaryDistanceModel, NeuralEmbeddingDistance, PitchClassEncoder
from pitchscape_comparison import (
    SELECTED_PIECES,
    build_comparison_trees,
    load_formal_metric_checkpoint,
    make_pitchscape,
    notes_tsv_to_pitchscape_values,
    select_resolution,
    tree_node_rows,
)


def _write_notes(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_five_examples_are_pre_registered_without_duplicates():
    assert [piece.piece_id for piece in SELECTED_PIECES] == [
        "n01op18-1_01", "n07op59-1_01", "n11op95_01",
        "n14op131_01", "n16op135_04",
    ]
    assert len({piece.directory_name for piece in SELECTED_PIECES}) == 5


def test_notes_tsv_pitchscape_uses_exact_event_timeline(tmp_path):
    notes = tmp_path / "piece.notes.tsv"
    _write_notes(notes, [
        {"quarterbeats": "0", "duration_qb": 2.0, "midi": 60},
        {"quarterbeats": "1", "duration_qb": 2.0, "midi": 67},
        {"quarterbeats": "2", "duration_qb": 0.0, "midi": 72},
    ])
    values, times = notes_tsv_to_pitchscape_values(notes)
    assert np.allclose(times, [0, 1, 2, 3])
    assert np.array_equal(values[:, 0], [1, 1, 0])
    assert np.array_equal(values[:, 7], [0, 1, 1])
    scape = make_pitchscape(notes)
    assert np.isclose(scape.min_time, 0)
    assert np.isclose(scape.max_time, 3)


def test_resolution_uses_finest_candidate_below_leaf_cap(tmp_path):
    notes = tmp_path / "long.notes.tsv"
    _write_notes(notes, [
        {"quarterbeats": 0.0, "duration_qb": 20.0, "midi": 60},
    ])
    matrix, bounds, bin_size = select_resolution(
        notes, candidates=[4, 8, 16], max_leaves=3)
    assert bin_size == 8
    assert len(matrix) == len(bounds) == 3


def test_three_methods_share_leaves_and_make_valid_trees():
    matrix = np.asarray([
        [3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
        [2, 0, 1, 0, 1, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0],
        [0, 0, 3, 0, 0, 0, 0, 1, 0, 2, 0, 0],
    ], dtype=float)
    bounds = [(0, 8), (8, 16), (16, 24), (24, 32)]
    encoder = PitchClassEncoder(dropout=0.0).eval()
    trees, diagnostics = build_comparison_trees(
        matrix, bounds, NeuralEmbeddingDistance(encoder), max_leaves=10)
    assert set(trees) == {"greedy", "dp", "dl"}
    for method, tree in trees.items():
        assert_valid_ordered_binary_tree(tree, bounds)
        rows = tree_node_rows(tree, method)
        assert len(rows) == 2 * len(matrix) - 1
        assert sum(bool(row["is_leaf"]) for row in rows) == len(matrix)
        assert diagnostics[method]["leaf_count"] == len(matrix)


def _write_formal_checkpoint_dir(path: Path, *, quick: bool = False) -> None:
    path.mkdir()
    seeds = [20260827, 20260828, 20260829]
    (path / "experiment_state.json").write_text(json.dumps({
        "phase": "complete", "completed_seeds": seeds,
        "requested_seeds": seeds, "device": "cpu", "quick": quick,
    }), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({
        "quick": quick, "model_seeds": seeds,
    }), encoding="utf-8")
    scores = [0.50, 0.75, 0.60]
    for seed, score in zip(seeds, scores):
        model = BoundaryDistanceModel(PitchClassEncoder(dropout=0.0))
        torch.save({
            "seed": seed,
            "encoder_config": model.encoder.architecture_config(),
            "metric": model.state_dict(),
            "selection": {"metric": {"validation_work_macro_ap": score}},
        }, path / f"checkpoint_seed_{seed}.pt")


def test_formal_checkpoint_is_selected_by_validation_only(tmp_path):
    checkpoint_dir = tmp_path / "formal"
    _write_formal_checkpoint_dir(checkpoint_dir)
    selected = load_formal_metric_checkpoint(checkpoint_dir, "cpu")
    assert selected.seed == 20260828
    assert selected.validation_ap == pytest.approx(0.75)
    assert selected.encoder_config["name"] == "circular_harmonic_cnn"
    assert len(selected.sha256) == 64


def test_quick_checkpoint_is_rejected(tmp_path):
    checkpoint_dir = tmp_path / "smoke"
    _write_formal_checkpoint_dir(checkpoint_dir, quick=True)
    with pytest.raises(RuntimeError, match="quick/smoke"):
        load_formal_metric_checkpoint(checkpoint_dir, "cpu")
