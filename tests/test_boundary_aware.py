from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boundary_contrast import (
    BoundaryContrastScorer,
    ProbabilityCalibrator,
    RobustContrastScaler,
    boundary_contrast_features,
    fit_probability_calibrator,
    fit_supervised_scorer,
    make_supervised_examples,
    nearest_boundary_labels,
    tolerance_aware_boundary_labels,
)
from greedy_clustering import ClusterNode, distance_specs
from greedy_evaluation import boundary_salience, collect_salient_splits
from ordered_affinity import (
    boundary_aware_tree_objective,
    greedy_boundary_aware_tree,
    hierarchical_span_weight,
    optimal_affinity_tree,
    optimal_boundary_aware_tree,
)
from evaluate_boundary_aware_stage import stable_inner_folds


def test_local_contrast_peaks_at_clean_change_and_constant_is_zero():
    matrix = np.zeros((8, 12))
    matrix[:4, 0] = 1
    matrix[4:, 6] = 1
    specs, _ = distance_specs(matrix.sum(0))
    features = boundary_contrast_features(matrix, specs, contexts=[1, 2])
    assert features.values.shape == (7, 6)
    assert features.values[3].mean() == pytest.approx(
        features.values.mean(axis=1).max())

    constant = np.zeros((8, 12))
    constant[:, 0] = 1
    constant_features = boundary_contrast_features(constant, specs, contexts=[1, 2, 4])
    assert np.allclose(constant_features.values, 0)
    scaler = RobustContrastScaler.fit([constant_features])
    scorer = BoundaryContrastScorer.unsupervised(scaler)
    assert np.allclose(scorer.score(constant_features), 0)


def test_contrast_is_finite_at_edges_and_transposition_invariant():
    rng = np.random.default_rng(22)
    matrix = rng.random((9, 12))
    specs, _ = distance_specs(matrix.sum(0))
    original = boundary_contrast_features(matrix, specs, contexts=[1, 4])
    shifted_matrix = np.roll(matrix, 5, axis=1)
    shifted_specs, _ = distance_specs(shifted_matrix.sum(0))
    shifted = boundary_contrast_features(shifted_matrix, shifted_specs, contexts=[1, 4])
    assert np.isfinite(original.values).all()
    assert np.allclose(original.values, shifted.values, atol=1e-10)


def test_supervised_scorer_is_nonnegative_deterministic_and_work_balanced():
    matrix_a = np.zeros((6, 12))
    matrix_a[:3, 0] = 1
    matrix_a[3:, 7] = 1
    matrix_b = np.zeros((6, 12))
    matrix_b[:2, 2] = 1
    matrix_b[2:, 9] = 1
    items = []
    for work, piece, matrix, boundary in (
        ("w1", "w1_01", matrix_a, 3.0),
        ("w2", "w2_01", matrix_b, 2.0),
    ):
        specs, _ = distance_specs(matrix.sum(0))
        features = boundary_contrast_features(matrix, specs, contexts=[1, 2])
        items.append({
            "work": work, "piece": piece, "features": features,
            "bounds": np.arange(7), "reference": [boundary],
        })
    scaler = RobustContrastScaler.fit([item["features"] for item in items])
    calibrated, labels, sample_weights, audit = make_supervised_examples(items, scaler)
    assert set(labels) == {0, 1}
    assert sample_weights.sum() == pytest.approx(1)
    assert sum(row["sample_weight"] for row in audit) == pytest.approx(1)
    assert {row["work"] for row in audit} == {"w1", "w2"}
    assert sum(row["selected"] for row in audit) == len(labels)
    first, history_a = fit_supervised_scorer(
        scaler, calibrated, labels, sample_weights, epochs=50)
    second, history_b = fit_supervised_scorer(
        scaler, calibrated, labels, sample_weights, epochs=50)
    assert np.all(first.weights >= 0)
    assert first.weights.sum() == pytest.approx(1)
    assert np.allclose(first.weights, second.weights)
    assert history_a == history_b
    assert np.isfinite(first.score(items[0]["features"])).all()


def test_nearest_annotation_mapping_collapses_duplicates_stably():
    labels = nearest_boundary_labels(
        np.arange(6, dtype=float), [1.4, 1.49, 4.1], n_bins=5)
    assert labels.tolist() == [1, 0, 0, 1]


def test_tolerance_aware_training_ignores_near_gt_candidates():
    matrix = np.zeros((6, 12))
    matrix[:3, 0] = 1
    matrix[3:, 7] = 1
    specs, _ = distance_specs(matrix.sum(0))
    features = boundary_contrast_features(matrix, specs, contexts=[1, 2])
    item = {
        "work": "w1", "piece": "w1_01", "features": features,
        "bounds": np.arange(7, dtype=float), "reference": [3.1],
    }
    scaler = RobustContrastScaler.fit([features])
    nearest, ignored = tolerance_aware_boundary_labels(
        item["bounds"], item["reference"], 6, ignore_tolerance_qb=1.1)
    assert nearest.tolist() == [0, 0, 1, 0, 0]
    assert ignored.tolist() == [False, True, False, True, False]

    calibrated, labels, weights, audit = make_supervised_examples(
        [item], scaler, ignore_tolerance_qb=1.1)
    assert calibrated.shape[0] == len(labels) == len(weights) == 2
    assert labels.tolist() == [1.0, 0.0]
    by_split = {row["split_index"]: row for row in audit}
    assert by_split[3]["status"] == "selected_positive"
    assert by_split[3]["selected"] and not by_split[3]["ignored"]
    for split in (2, 4):
        row = by_split[split]
        assert row["status"] == "ignored_tolerance"
        assert row["ignored"] and not row["selected"]
        assert row["label"] is None and row["sample_weight"] == 0.0


def test_zero_ignore_tolerance_preserves_legacy_training_arrays():
    matrix = np.eye(6, 12)
    specs, _ = distance_specs(matrix.sum(0))
    features = boundary_contrast_features(matrix, specs, contexts=[1])
    item = {"work": "w", "piece": "w_01", "features": features,
            "bounds": np.arange(7, dtype=float), "reference": [3.0]}
    scaler = RobustContrastScaler.fit([features])
    implicit = make_supervised_examples([item], scaler)
    explicit = make_supervised_examples(
        [item], scaler, ignore_tolerance_qb=0.0)
    for left, right in zip(implicit[:3], explicit[:3]):
        assert np.array_equal(left, right)
    assert implicit[3] == explicit[3]
    assert not any(row["ignored"] for row in implicit[3])


def test_probability_calibration_is_deterministic_monotone_and_serialisable():
    scores = np.array([0.05, 0.15, 0.25, 0.35, 0.55, 0.65, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    first = fit_probability_calibrator(scores, labels)
    second = fit_probability_calibrator(scores, labels)
    assert isinstance(first, ProbabilityCalibrator)
    assert first == second
    calibrated = first.transform(scores)
    assert np.isfinite(calibrated).all()
    assert np.all((0 < calibrated) & (calibrated < 1))
    assert np.all(np.diff(calibrated) >= 0)
    assert np.allclose(first.score(scores), calibrated)
    assert np.allclose(first.predict_proba(scores), calibrated)
    payload = first.as_dict()
    assert payload["fit_scope"] == "training_works_all_nonignored_candidates"
    assert json.loads(json.dumps(payload))["slope"] == pytest.approx(first.slope)


def test_probability_calibration_rejects_missing_class_and_bad_scores():
    with pytest.raises(ValueError, match="both classes"):
        fit_probability_calibrator([0.1, 0.2], [0, 0])
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        fit_probability_calibrator([0.1, 1.2], [0, 1])


def brute_boundary_objective(affinity, contrast, weight, balance=0.0):
    n = len(affinity)
    denominator = ((n - 2) * affinity[np.triu_indices(n, 1)].sum()
                   if n > 2 else 0.0)

    @lru_cache(None)
    def solve(i, j):
        if j == i + 1:
            return 0.0
        values = []
        for k in range(i + 1, j):
            cross = affinity[i:k, k:j].sum()
            local_affinity = ((n - (j - i)) * cross / denominator
                              if denominator > 0 else 0.0)
            local_boundary = (
                hierarchical_span_weight(j - i, n)
                * contrast[k - 1] / max(1, n - 1))
            imbalance = abs((k - i) - (j - k)) / (j - i)
            local_balance = imbalance * imbalance / max(1, n - 1)
            values.append(
                solve(i, k) + solve(k, j)
                + (1 - weight) * local_affinity + weight * local_boundary
                - balance * local_balance)
        return max(values)

    return solve(0, n)


def test_exact_boundary_dp_matches_explicit_recurrence_for_small_trees():
    rng = np.random.default_rng(913)
    for n in range(2, 8):
        raw = rng.random((n, n))
        affinity = (raw + raw.T) / 2
        np.fill_diagonal(affinity, 1)
        contrast = rng.random(n - 1)
        matrix = rng.random((n, 12))
        for weight in (0.0, 0.35, 1.0):
            for balance in (0.0, 0.4):
                root, diagnostics = optimal_boundary_aware_tree(
                    matrix, np.arange(n + 1), affinity, contrast, ClusterNode,
                    contrast_weight=weight, balance_weight=balance)
                expected = brute_boundary_objective(
                    affinity, contrast, weight, balance)
                recomputed = boundary_aware_tree_objective(
                    root, affinity, contrast, contrast_weight=weight,
                    balance_weight=balance)
                assert diagnostics.total_objective == pytest.approx(expected)
                assert recomputed.total_objective == pytest.approx(expected)


def tree_signature(node):
    return (node.start, node.end, tuple(tree_signature(child) for child in node.children))


def test_lambda_zero_reproduces_similarity_dp_and_high_contrast_becomes_root():
    rng = np.random.default_rng(55)
    n = 7
    raw = rng.random((n, n))
    affinity = (raw + raw.T) / 2
    np.fill_diagonal(affinity, 1)
    matrix = rng.random((n, 12))
    contrast = np.zeros(n - 1)
    contrast[3] = 1
    legacy, legacy_info = optimal_affinity_tree(
        matrix, np.arange(n + 1), affinity, ClusterNode)
    zero, zero_info = optimal_boundary_aware_tree(
        matrix, np.arange(n + 1), affinity, contrast, ClusterNode,
        contrast_weight=0)
    assert tree_signature(zero) == tree_signature(legacy)
    assert zero_info.normalized_affinity_revenue == pytest.approx(
        legacy_info.normalized_revenue)

    boundary_only, info = optimal_boundary_aware_tree(
        matrix, np.arange(n + 1), affinity, contrast, ClusterNode,
        contrast_weight=1)
    assert info.root_split == 4
    assert boundary_salience(boundary_only, contrast)[0]["boundary_index"] == 4


def test_exact_objective_dominates_greedy_and_salience_selection_is_stable():
    rng = np.random.default_rng(431)
    n = 9
    raw = rng.random((n, n))
    affinity = (raw + raw.T) / 2
    np.fill_diagonal(affinity, 1)
    contrast = rng.random(n - 1)
    matrix = rng.random((n, 12))
    greedy = greedy_boundary_aware_tree(
        matrix, np.arange(n + 1), affinity, contrast, ClusterNode,
        contrast_weight=0.5, balance_weight=0.5)
    greedy_info = boundary_aware_tree_objective(
        greedy, affinity, contrast, contrast_weight=0.5,
        balance_weight=0.5)
    dp, dp_info = optimal_boundary_aware_tree(
        matrix, np.arange(n + 1), affinity, contrast, ClusterNode,
        contrast_weight=0.5, balance_weight=0.5)
    assert dp_info.total_objective >= greedy_info.total_objective - 1e-12
    assert len(collect_salient_splits(dp, contrast, budget=3)) == 3
    thresholded = collect_salient_splits(dp, contrast, threshold=0.5)
    assert thresholded == sorted(thresholded)


def test_unweighted_boundary_sum_is_tree_invariant_but_span_reward_is_not():
    contrast = np.array([0.1, 1.0, 0.2])
    leaves = [ClusterNode(i, i + 1, np.ones(12)) for i in range(4)]
    left = ClusterNode(0, 2, np.ones(12), leaves[:2])
    right = ClusterNode(2, 4, np.ones(12), leaves[2:])
    strong_root = ClusterNode(0, 4, np.ones(12), [left, right])
    weak_pair = ClusterNode(0, 2, np.ones(12), [leaves[0], leaves[1]])
    weak_left = ClusterNode(0, 3, np.ones(12), [weak_pair, leaves[2]])
    # Use a valid second tree with the strong boundary below the root.
    weak_root = ClusterNode(0, 4, np.ones(12), [weak_left, leaves[3]])
    def split_indices(root):
        rows = []
        cursor = [0]
        def walk(node):
            if not node.children:
                start = cursor[0]
                cursor[0] += 1
                return start, start + 1
            left_start, split = walk(node.children[0])
            _, end = walk(node.children[1])
            rows.append(split)
            return left_start, end
        walk(root)
        return sorted(rows)
    assert split_indices(strong_root) == split_indices(weak_root) == [1, 2, 3]
    assert contrast[[
        index - 1 for index in split_indices(strong_root)]].sum() == pytest.approx(
        contrast[[index - 1 for index in split_indices(weak_root)]].sum())
    affinity = np.eye(4)
    strong = boundary_aware_tree_objective(
        strong_root, affinity, contrast, contrast_weight=1)
    weak = boundary_aware_tree_objective(
        weak_root, affinity, contrast, contrast_weight=1)
    assert strong.boundary_reward > weak.boundary_reward


def test_grouped_inner_folds_are_deterministic_and_disjoint():
    works = [f"work_{index}" for index in range(10)]
    first = stable_inner_folds(works, 3, 91, "held_out")
    second = stable_inner_folds(works, 3, 91, "held_out")
    assert first == second
    flattened = [work for fold in first for work in fold]
    assert sorted(flattened) == sorted(works)
    assert len(flattened) == len(set(flattened))
