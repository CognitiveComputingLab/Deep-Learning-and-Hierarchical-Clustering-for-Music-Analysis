from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boundary_contrast import BoundaryContrastScorer, RobustContrastScaler
from evaluate_neural_distance_stage import (
    SHARD_TABLES,
    configuration_fingerprint,
    estimate_training_affinity_scale,
    load_shard,
    materialize_learned_items,
    select_family_candidate,
    validate_shard,
    write_shard,
)
from evaluate_boundary_aware_stage import build_trees
from neural_distance import build_interval_training_set, train_distance_checkpoints


def _items():
    output = []
    for work, shift in (("w1", 0), ("w2", 2)):
        matrix = np.zeros((6, 12), dtype=float)
        matrix[:3, shift] = 1
        matrix[3:, (shift + 6) % 12] = 1
        output.append({
            "work": work, "piece": work + "_01", "matrix": matrix,
            "bounds": [(8 * index, 8 * (index + 1)) for index in range(6)],
            "reference": [24.0], "segments": [(0, 24, "I"), (24, 48, "V")],
            "total": 48.0,
        })
    return output


def test_checkpoint_selection_is_work_macro_and_fixes_epoch_for_budget():
    rows = []
    for work in ("w1", "w2"):
        for search in ("greedy", "dp"):
            for epoch, f1 in ((20, 0.7), (40, 0.6)):
                rows.append({
                    "work": work, "search": search, "family": "neural_circular_siamese",
                    "selection_mode": "threshold", "checkpoint_epoch": epoch,
                    "contrast_weight": 0.25, "balance_weight": 0.1,
                    "candidate_value": 0.5, "min_separation_qb": 8.0,
                    "local_max_radius_qb": 0.0, "precision": f1,
                    "recall": f1, "f1": f1, "f0.5": f1,
                })
    frame = pd.DataFrame(rows)
    choice, candidates = select_family_candidate(
        frame, "neural_circular_siamese", "threshold")
    assert choice["checkpoint_epoch"] == 20
    assert candidates.chosen.sum() == 1


def test_learned_terms_use_training_only_scale_and_dp_dominates_greedy():
    items = _items()
    training = build_interval_training_set(items, contexts=(1, 2), ignore_tolerance_qb=8)
    checkpoints, _ = train_distance_checkpoints(
        training, checkpoint_epochs=(2,), batch_size=8, seed=9)
    distance = checkpoints[0].distance
    scale = estimate_training_affinity_scale(items[:1], distance)
    assert np.isfinite(scale) and scale > 0
    args = SimpleNamespace(
        contexts=[1, 2], affinity_context_radius=0,
        max_bins=20, tie_break="midpoint")
    transformed, metadata = materialize_learned_items(items, distance, scale, args)
    assert all(row["scale_source"] == "provided_training_scale" for row in metadata)
    scaler = RobustContrastScaler.fit([item["features"] for item in transformed])
    scorer = BoundaryContrastScorer.unsupervised(scaler)
    item = transformed[0]
    contrast = scorer.score(item["features"])
    trees = build_trees(item, contrast, 0.5, 0.1, args)
    objective = {search: diagnostics.total_objective
                 for search, _, diagnostics, _ in trees}
    assert objective["dp"] >= objective["greedy"] - 1e-8


def test_outer_test_work_cannot_enter_training_audit():
    training = build_interval_training_set(
        _items()[:1], contexts=(1,), ignore_tolerance_qb=8)
    assert {row["work"] for row in training.audit} == {"w1"}
    assert "w2" not in training.works


def test_outer_shard_round_trip_and_corruption_detection(tmp_path):
    collections = {key: [] for key in SHARD_TABLES}
    for rows in collections.values():
        rows.append({"outer_work": "w1", "value": 1})
    starts = {key: 0 for key in collections}
    configuration = {"families": ["handcrafted_key_profile"], "seed": 7}
    fingerprint = configuration_fingerprint(configuration, "data", "code")

    marker = write_shard(
        tmp_path, "w1", collections, starts, fingerprint,
        configuration, "data", "code", expected_checkpoint_count=0)
    assert marker["checkpoint_count"] == 0
    assert marker["row_counts"]["held_out_per_piece.csv"] == 1

    restored = {key: [] for key in SHARD_TABLES}
    assert load_shard(tmp_path, "w1", restored, fingerprint)["status"] == "success"
    assert all(rows == [{"outer_work": "w1", "value": 1}]
               for rows in restored.values())
    with pytest.raises(RuntimeError, match="different data, code, or settings"):
        validate_shard(tmp_path, "w1", "another-fingerprint")

    table = tmp_path / "shards" / "w1" / "held_out_per_piece.csv"
    table.write_text("corrupt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or corrupt"):
        validate_shard(tmp_path, "w1", fingerprint)
