from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import greedy_evaluation
from greedy_clustering import ClusterNode
from greedy_evaluation import boundary_salience, collect_salient_splits


def _balanced_four_leaf_tree():
    leaves = [ClusterNode(index, index + 1, None) for index in range(4)]
    left = ClusterNode(0, 2, None, leaves[:2])
    right = ClusterNode(2, 4, None, leaves[2:])
    return ClusterNode(0, 4, None, [left, right])


def _row(boundary, index, salience, leaves=2):
    return {
        'boundary_qb': float(boundary),
        'boundary_index': int(index),
        'salience': float(salience),
        'lca_leaf_count': int(leaves),
    }


def test_disabled_peak_options_preserve_legacy_threshold_and_budget():
    root = _balanced_four_leaf_tree()
    contrast = [0.4, 0.8, 0.6]
    rows = boundary_salience(root, contrast)
    legacy_threshold = sorted(
        row['boundary_qb'] for row in rows if row['salience'] >= 0.3)
    legacy_budget = sorted(row['boundary_qb'] for row in rows[:2])
    assert collect_salient_splits(
        root, contrast, threshold=0.3) == legacy_threshold
    assert collect_salient_splits(
        root, contrast, budget=2) == legacy_budget


def test_nms_is_salience_descending_with_time_ascending_ties(monkeypatch):
    # Return rows out of rank order and give the later tied peak a larger
    # structural span. NMS must still retain the earlier tied peak.
    rows = [
        _row(25, 3, 0.8),
        _row(20, 2, 0.9, leaves=20),
        _row(40, 4, 0.7),
        _row(10, 1, 0.9, leaves=2),
    ]
    monkeypatch.setattr(greedy_evaluation, 'boundary_salience',
                        lambda root, contrast: rows)
    selected = collect_salient_splits(
        object(), threshold=0.0, min_separation_qb=15)
    assert selected == [10.0, 25.0, 40.0]


def test_budget_is_applied_after_nms(monkeypatch):
    rows = [_row(10, 1, 0.9), _row(12, 2, 0.8), _row(30, 3, 0.7)]
    monkeypatch.setattr(greedy_evaluation, 'boundary_salience',
                        lambda root, contrast: rows)
    assert collect_salient_splits(
        object(), budget=2, min_separation_qb=5) == [10.0, 30.0]


def test_local_maximum_radius_collapses_equal_peak_deterministically(monkeypatch):
    rows = [
        _row(10, 1, 0.2),
        _row(20, 2, 0.9),
        _row(30, 3, 0.9),
        _row(40, 4, 0.2),
        _row(60, 5, 0.7),
    ]
    monkeypatch.setattr(greedy_evaluation, 'boundary_salience',
                        lambda root, contrast: list(reversed(rows)))
    assert collect_salient_splits(
        object(), threshold=0.0, local_max_radius_qb=11) == [20.0, 60.0]


def test_threshold_is_inclusive_before_nms(monkeypatch):
    rows = [_row(10, 1, 0.5), _row(12, 2, 0.49), _row(30, 3, 0.5)]
    monkeypatch.setattr(greedy_evaluation, 'boundary_salience',
                        lambda root, contrast: rows)
    assert collect_salient_splits(
        object(), threshold=0.5, min_separation_qb=5) == [10.0, 30.0]
