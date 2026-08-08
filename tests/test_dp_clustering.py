"""Unit tests for globally optimal interval DP."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dp_clustering import (
    additive_tree_cost,
    assert_valid_ordered_binary_tree,
    optimal_adjacent_binary_tree,
)


@dataclass
class Node:
    start: float
    end: float
    vector: np.ndarray
    children: list["Node"] = field(default_factory=list)


def euclidean(left, right):
    left = left / left.sum() if left.sum() else left
    right = right / right.sum() if right.sum() else right
    return float(np.linalg.norm(left - right))


def test_two_bins_have_single_possible_cost():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]])
    bounds = np.array([0.0, 1.0, 2.0])
    root, diagnostics = optimal_adjacent_binary_tree(
        matrix, bounds, euclidean, Node
    )
    expected = euclidean(matrix[0], matrix[1])
    assert np.isclose(diagnostics.total_cost, expected)
    assert_valid_ordered_binary_tree(root, bounds)


def test_three_bins_choose_lower_of_two_trees():
    a = np.array([1.0, 0.0])
    b = np.array([0.8, 0.2])
    c = np.array([0.0, 1.0])
    matrix = np.vstack([a, b, c])
    bounds = np.arange(4, dtype=float)

    left_first = euclidean(a, b) + euclidean(a + b, c)
    right_first = euclidean(b, c) + euclidean(a, b + c)

    _, diagnostics = optimal_adjacent_binary_tree(
        matrix, bounds, euclidean, Node
    )
    assert np.isclose(diagnostics.total_cost, min(left_first, right_first))


def brute_force_cost(matrix, distance):
    prefix = np.zeros((len(matrix) + 1, matrix.shape[1]))
    prefix[1:] = np.cumsum(matrix, axis=0)

    def vector(i, j):
        return prefix[j] - prefix[i]

    @lru_cache(None)
    def solve(i, j):
        if j == i + 1:
            return 0.0
        return min(
            solve(i, k) + solve(k, j) + distance(vector(i, k), vector(k, j))
            for k in range(i + 1, j)
        )

    return solve(0, len(matrix))


def test_dp_matches_brute_force_small_random_cases():
    rng = np.random.default_rng(7)
    for n in range(2, 7):
        for _ in range(10):
            matrix = rng.random((n, 4))
            bounds = np.arange(n + 1, dtype=float)
            _, diagnostics = optimal_adjacent_binary_tree(
                matrix, bounds, euclidean, Node
            )
            expected = brute_force_cost(matrix, euclidean)
            assert np.isclose(diagnostics.total_cost, expected)


def all_ordered_tree_costs(matrix, distance):
    prefix=np.vstack([np.zeros(matrix.shape[1]),np.cumsum(matrix,axis=0)])
    def enumerate_interval(i,j):
        if j==i+1:
            return [0.0]
        values=[]
        for k in range(i+1,j):
            local=distance(prefix[k]-prefix[i],prefix[j]-prefix[k])
            values.extend(left+right+local
                          for left in enumerate_interval(i,k)
                          for right in enumerate_interval(k,j))
        return values
    return enumerate_interval(0,len(matrix))


def test_dp_matches_explicit_enumeration_through_seven_leaves():
    rng=np.random.default_rng(71)
    for n in range(2,8):
        matrix=rng.random((n,4))
        _,diagnostics=optimal_adjacent_binary_tree(
            matrix,np.arange(n+1,dtype=float),euclidean,Node)
        assert np.isclose(diagnostics.total_cost,min(all_ordered_tree_costs(matrix,euclidean)))


def test_dp_never_worse_than_a_given_binary_tree():
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )
    bounds = np.arange(5, dtype=float)
    dp_root, diagnostics = optimal_adjacent_binary_tree(
        matrix, bounds, euclidean, Node
    )

    leaves = [
        Node(bounds[i], bounds[i + 1], matrix[i].copy())
        for i in range(4)
    ]
    left = Node(0.0, 2.0, matrix[:2].sum(axis=0), leaves[:2])
    right = Node(2.0, 4.0, matrix[2:].sum(axis=0), leaves[2:])
    candidate = Node(0.0, 4.0, matrix.sum(axis=0), [left, right])

    candidate_cost = additive_tree_cost(candidate, euclidean)
    assert diagnostics.total_cost <= candidate_cost + 1e-12
    assert_valid_ordered_binary_tree(dp_root, bounds)
