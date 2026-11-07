from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greedy_clustering import ClusterNode
from hierarchical_evaluation import (
    equal_budget_level_scores, estimated_lca_depths, full_tree_measure,
    reference_lca_depths, sample_times_from_tree, tree_measure)
from shared_annotation_format import SharedAnnotation, SharedPiece


def _node(start, end, left=None, right=None):
    children = [] if left is None else [left, right]
    return ClusterNode(start, end, np.ones(12), children)


def _balanced_tree():
    leaves = [_node(i, i + 1) for i in range(4)]
    return _node(0, 4, _node(0, 2, leaves[0], leaves[1]),
                 _node(2, 4, leaves[2], leaves[3]))


def _piece():
    rows = [
        ("root", "ROOT", 0, 4, 0, None),
        ("a", "A", 0, 2, 1, "root"),
        ("b", "B", 2, 4, 1, "root"),
        ("a1", "a1", 0, 1, 2, "a"),
        ("a2", "a2", 1, 2, 2, "a"),
        ("b1", "b1", 2, 3, 2, "b"),
        ("b2", "b2", 3, 4, 2, "b"),
    ]
    annotations = [SharedAnnotation(
        annotation_id=name, kind="hierarchical_segment", layer="form",
        label=label, start_qb=start, end_qb=end, level=level,
        parent_id=parent, tree_id="form")
        for name, label, start, end, level, parent in rows]
    return SharedPiece(dataset="toy", piece_id="toy", work_id="toy",
                       coordinate_system="quarterbeat", duration_qb=4,
                       annotations=annotations)


def test_matching_tree_has_perfect_tree_measures():
    result = tree_measure(_balanced_tree(), _piece())
    assert result["t_measure_full"] == 1.0
    assert result["t_measure_reduced"] == 1.0
    assert result["lca_spearman"] == 1.0
    assert result["lca_normalized_mae"] == 0.0
    times = sample_times_from_tree(_balanced_tree())
    prepared = (times, reference_lca_depths(_piece(), times))
    fast = full_tree_measure(
        _balanced_tree(), _piece(), prepared_reference=prepared)
    assert fast["t_measure_full"] == result["t_measure_full"]


def test_wrong_tree_scores_below_matching_tree():
    leaves = [_node(i, i + 1) for i in range(4)]
    wrong = _node(0, 4, leaves[0],
                  _node(1, 4, leaves[1], _node(2, 4, leaves[2], leaves[3])))
    result = tree_measure(wrong, _piece())
    assert result["t_measure_full"] < 1.0
    assert result["lca_spearman"] < 1.0


def test_equal_budget_level_boundary_score_is_exact():
    result = equal_budget_level_scores(
        _balanced_tree(), _piece(), max_level=1, tolerance_qb=0)
    assert result["prediction_budget"] == 1
    assert result["predicted_boundary_count"] == 1
    assert result["f1"] == 1.0


def test_lca_matrix_uses_safe_integer_dtype():
    tree = _node(0, 2, _node(0, 1), _node(1, 2))
    matrix = estimated_lca_depths(tree)
    assert matrix.dtype == np.int32
    assert matrix.tolist() == [[2, 1], [1, 2]]
