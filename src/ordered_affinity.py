"""Order-preserving hierarchical clustering from pairwise affinities.

This module provides a principled alternative to the legacy objective that
sums distances between aggregate child chromagrams.  Both search algorithms
consume the same non-negative leaf affinity matrix:

* ``greedy_adjacent_average_linkage`` is the required bottom-up, adjacent
  agglomerative heuristic;
* ``optimal_affinity_tree`` is an exact interval dynamic program.

The optimized objective is the similarity revenue dual to Dasgupta's
hierarchical clustering cost, restricted to temporally contiguous trees::

    revenue(T) = sum_{p < q} (n - |LCA_T(p, q)|) * similarity[p, q]

Large similarities are rewarded when their leaves meet deep in the tree.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal, Sequence

import numpy as np


TieBreak = Literal["earliest", "midpoint", "latest"]


@dataclass(frozen=True)
class AffinityDiagnostics:
    total_revenue: float
    normalized_revenue: float
    n_bins: int
    evaluated_splits: int
    elapsed_seconds: float
    root_split: int
    tie_count: int
    objective_name: str = "ordered_similarity_revenue"


@dataclass(frozen=True)
class BoundaryAwareDiagnostics:
    """Components of the shared boundary-aware Greedy/DP objective."""

    total_objective: float
    affinity_revenue: float
    normalized_affinity_revenue: float
    boundary_reward: float
    balance_penalty: float
    contrast_weight: float
    balance_weight: float
    n_bins: int
    evaluated_splits: int
    elapsed_seconds: float
    root_split: int
    tie_count: int
    objective_name: str = "normalized_affinity_plus_hierarchical_boundary_contrast"


def _validated_bounds(matrix: np.ndarray, bounds: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 1:
        raise ValueError("matrix must be a non-empty 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    if bounds.ndim == 2 and bounds.shape == (len(matrix), 2):
        if len(bounds) > 1 and not np.allclose(bounds[:-1, 1], bounds[1:, 0]):
            raise ValueError("interval bounds must be temporally contiguous")
        bounds = np.r_[bounds[0, 0], bounds[:, 1]]
    elif bounds.ndim != 1 or len(bounds) != len(matrix) + 1:
        raise ValueError("bounds must have shape (n_bins,2) or (n_bins+1,)")
    if not np.all(np.isfinite(bounds)) or np.any(np.diff(bounds) <= 0):
        raise ValueError("bounds must be finite and strictly increasing")
    return matrix, bounds


def validate_affinity(affinity: np.ndarray, *, atol: float = 1e-10) -> np.ndarray:
    affinity = np.asarray(affinity, dtype=float)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1]:
        raise ValueError("affinity must be a square matrix")
    if not np.all(np.isfinite(affinity)):
        raise ValueError("affinity contains non-finite values")
    if np.any(affinity < -atol):
        raise ValueError("affinity must be non-negative")
    if not np.allclose(affinity, affinity.T, atol=atol, rtol=0):
        raise ValueError("affinity must be symmetric")
    affinity = np.maximum(0.0, (affinity + affinity.T) / 2.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def pairwise_affinity(
    matrix: np.ndarray,
    distance_spec: Any,
    *,
    scale: float | None = None,
    context_radius: int = 0,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Convert a project ``DistanceSpec`` into a calibrated RBF affinity.

    ``scale`` is normally estimated on training data.  If omitted, the
    within-piece non-zero median is used for unsupervised handcrafted
    baselines and is explicitly recorded in the returned metadata.

    Context smoothing averages similarities along self-similarity-matrix
    diagonals.  It adds short sequential evidence without using annotations.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 1:
        raise ValueError("matrix must be a non-empty 2-D array")
    transform = getattr(distance_spec, "transform_batch", None)
    if transform is None:
        raise TypeError("distance_spec must provide transform_batch()")
    representation = np.asarray(transform(matrix), dtype=float)
    if representation.ndim != 2 or len(representation) != len(matrix):
        raise ValueError("transform_batch must return one vector per bin")
    differences = representation[:, None, :] - representation[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    positive = distances[np.triu_indices(len(matrix), 1)]
    positive = positive[np.isfinite(positive) & (positive > 1e-12)]
    estimated = scale is None
    if scale is None:
        scale = float(np.median(positive)) if len(positive) else 1.0
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("affinity scale must be positive and finite")
    affinity = np.exp(-0.5 * (distances / scale) ** 2)

    radius = int(context_radius)
    if radius < 0:
        raise ValueError("context_radius must be non-negative")
    if radius:
        total = np.zeros_like(affinity)
        count = np.zeros_like(affinity)
        n = len(affinity)
        for offset in range(-radius, radius + 1):
            if offset >= 0:
                source = affinity[offset:, offset:]
                total[: n - offset, : n - offset] += source
                count[: n - offset, : n - offset] += 1
            else:
                shift = -offset
                source = affinity[: n - shift, : n - shift]
                total[shift:, shift:] += source
                count[shift:, shift:] += 1
        affinity = np.divide(total, count, out=affinity.copy(), where=count > 0)
    affinity = validate_affinity(affinity)
    return affinity, {
        "distance": str(getattr(distance_spec, "name", type(distance_spec).__name__)),
        "kernel": "exp(-0.5 * (distance / scale)^2)",
        "scale": scale,
        "scale_source": "within_piece_nonzero_median" if estimated else "provided_training_scale",
        "context_radius": radius,
    }


def _prefix2d(values: np.ndarray) -> np.ndarray:
    prefix = np.zeros((len(values) + 1, len(values) + 1), dtype=float)
    prefix[1:, 1:] = np.cumsum(np.cumsum(values, axis=0), axis=1)
    return prefix


def _rectangle_sum(prefix: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    return float(prefix[r1, c1] - prefix[r0, c1] - prefix[r1, c0] + prefix[r0, c0])


def _make_node(node_type: type, start: float, end: float, feature: np.ndarray,
               children: list[Any], merge_order: int = -1) -> Any:
    attempts = (
        lambda: node_type(start, end, feature, children, merge_order),
        lambda: node_type(start, end, feature, children),
        lambda: node_type(start=start, end=end, feature=feature, children=children),
        lambda: node_type(start=start, end=end, vector=feature, children=children),
    )
    error: Exception | None = None
    for attempt in attempts:
        try:
            node = attempt()
            if not hasattr(node, "children"):
                node.children = children
            if not hasattr(node, "feature"):
                node.feature = feature
            node.merge_order = merge_order
            return node
        except (TypeError, AttributeError) as candidate:
            error = candidate
    raise TypeError("Could not construct the requested node type") from error


def _normalization_denominator(affinity: np.ndarray) -> float:
    n = len(affinity)
    if n <= 2:
        return 0.0
    return float((n - 2) * affinity[np.triu_indices(n, 1)].sum())


def optimal_affinity_tree(
    matrix: np.ndarray,
    bounds: Sequence[float],
    affinity: np.ndarray,
    node_type: type,
    *,
    tie_break: TieBreak = "midpoint",
    tie_tolerance: float = 1e-12,
    max_bins: int | None = None,
) -> tuple[Any, AffinityDiagnostics]:
    """Find the exact maximum-revenue contiguous ordered binary tree."""
    matrix, edges = _validated_bounds(matrix, bounds)
    affinity = validate_affinity(affinity)
    n = len(matrix)
    if affinity.shape != (n, n):
        raise ValueError("affinity size must match the number of bins")
    if max_bins is not None and n > max_bins:
        raise ValueError(f"{n} bins exceeds max_bins={max_bins}")
    if tie_break not in {"earliest", "midpoint", "latest"}:
        raise ValueError("unknown tie_break")

    started = time.perf_counter()
    prefix = _prefix2d(affinity)
    feature_prefix = np.vstack([np.zeros(matrix.shape[1]), np.cumsum(matrix, axis=0)])
    revenue = np.full((n + 1, n + 1), -np.inf, dtype=float)
    split = np.full((n + 1, n + 1), -1, dtype=int)
    for i in range(n):
        revenue[i, i + 1] = 0.0
    evaluated = 0
    tie_count = 0

    for length in range(2, n + 1):
        coefficient = n - length
        for i in range(0, n - length + 1):
            j = i + length
            ks = np.arange(i + 1, j)
            # Rectangle [i:k) x [k:j) for every candidate k.  Vectorising
            # this inner loop is important because the complete DP is cubic.
            cross = (prefix[ks, j] - prefix[i, j]
                     - prefix[ks, ks] + prefix[i, ks])
            candidates = revenue[i, ks] + revenue[ks, j] + coefficient * cross
            evaluated += len(ks)
            best = float(candidates.max())
            tied = np.flatnonzero(candidates >= best - tie_tolerance)
            if len(tied) > 1:
                tie_count += 1
            if tie_break == "earliest":
                chosen = int(tied[0])
            elif tie_break == "latest":
                chosen = int(tied[-1])
            else:
                midpoint = (i + j) / 2.0
                chosen = min((int(index) for index in tied),
                             key=lambda index: (abs(float(ks[index]) - midpoint), int(ks[index])))
            split[i, j] = int(ks[chosen])
            revenue[i, j] = float(candidates[chosen])

    merge_order = [0]

    def build(i: int, j: int) -> Any:
        feature = (feature_prefix[j] - feature_prefix[i]).copy()
        if j == i + 1:
            node = _make_node(node_type, edges[i], edges[j], feature, [])
            node.merge_revenue = 0.0
            node.subtree_objective = 0.0
            return node
        k = int(split[i, j])
        left, right = build(i, k), build(k, j)
        node = _make_node(node_type, edges[i], edges[j], feature, [left, right], merge_order[0])
        merge_order[0] += 1
        cross = _rectangle_sum(prefix, i, k, k, j)
        node.child_mean_affinity = cross / ((k - i) * (j - k))
        node.merge_revenue = float(revenue[i, j] - revenue[i, k] - revenue[k, j])
        node.subtree_objective = float(revenue[i, j])
        return node

    root = build(0, n)
    total = float(revenue[0, n]) if n > 1 else 0.0
    denominator = _normalization_denominator(affinity)
    diagnostics = AffinityDiagnostics(
        total_revenue=total,
        normalized_revenue=total / denominator if denominator > 0 else 0.0,
        n_bins=n,
        evaluated_splits=evaluated,
        elapsed_seconds=time.perf_counter() - started,
        root_split=int(split[0, n]) if n > 1 else -1,
        tie_count=tie_count,
    )
    return root, diagnostics


def greedy_adjacent_average_linkage(
    matrix: np.ndarray,
    bounds: Sequence[float],
    affinity: np.ndarray,
    node_type: type,
) -> Any:
    """Bottom-up adjacent average-linkage clustering on a fixed affinity."""
    matrix, edges = _validated_bounds(matrix, bounds)
    affinity = validate_affinity(affinity)
    n = len(matrix)
    if affinity.shape != (n, n):
        raise ValueError("affinity size must match the number of bins")
    prefix = _prefix2d(affinity)
    clusters: list[tuple[int, int, Any]] = [
        (i, i + 1, _make_node(node_type, edges[i], edges[i + 1], matrix[i].copy(), []))
        for i in range(n)
    ]
    merge_order = 0
    while len(clusters) > 1:
        scores = []
        for left, right in zip(clusters[:-1], clusters[1:]):
            cross = _rectangle_sum(prefix, left[0], left[1], right[0], right[1])
            scores.append(cross / ((left[1] - left[0]) * (right[1] - right[0])))
        index = int(np.argmax(scores))
        i, k, left_node = clusters[index]
        _, j, right_node = clusters[index + 1]
        feature = np.asarray(left_node.feature) + np.asarray(right_node.feature)
        node = _make_node(node_type, edges[i], edges[j], feature,
                          [left_node, right_node], merge_order)
        node.child_mean_affinity = float(scores[index])
        merge_order += 1
        clusters[index:index + 2] = [(i, j, node)]
    return clusters[0][2]


def affinity_tree_revenue(root: Any, affinity: np.ndarray) -> tuple[float, float]:
    """Recompute raw and normalized revenue for any compatible binary tree."""
    affinity = validate_affinity(affinity)
    n = len(affinity)
    prefix = _prefix2d(affinity)
    cursor = [0]

    def walk(node: Any) -> tuple[int, int, float]:
        children = list(getattr(node, "children", []) or [])
        if not children:
            start = cursor[0]
            cursor[0] += 1
            return start, start + 1, 0.0
        if len(children) != 2:
            raise ValueError("affinity_tree_revenue expects a binary tree")
        li, lk, left_value = walk(children[0])
        rk, rj, right_value = walk(children[1])
        if lk != rk:
            raise ValueError("tree leaves are not temporally contiguous")
        cross = _rectangle_sum(prefix, li, lk, rk, rj)
        value = left_value + right_value + (n - (rj - li)) * cross
        return li, rj, float(value)

    start, end, total = walk(root)
    if start != 0 or end != n or cursor[0] != n:
        raise ValueError("tree leaf count does not match affinity")
    denominator = _normalization_denominator(affinity)
    return total, total / denominator if denominator > 0 else 0.0


def hierarchical_span_weight(span_leaves: int, n_bins: int) -> float:
    """Map an LCA leaf span to [0, 1], with the complete piece equal to one."""
    span_leaves, n_bins = int(span_leaves), int(n_bins)
    if span_leaves < 2 or n_bins < 2 or span_leaves > n_bins:
        return 0.0
    return float(np.log(span_leaves) / np.log(n_bins))


def _validate_boundary_contrast(contrast: Sequence[float], n: int) -> np.ndarray:
    contrast = np.asarray(contrast, dtype=float)
    if contrast.shape != (max(0, n - 1),):
        raise ValueError("boundary contrast must have one value per internal leaf boundary")
    if not np.all(np.isfinite(contrast)):
        raise ValueError("boundary contrast contains non-finite values")
    if np.any(contrast < -1e-12) or np.any(contrast > 1.0 + 1e-12):
        raise ValueError("boundary contrast must lie in [0, 1]")
    return np.clip(contrast, 0.0, 1.0)


def _validate_contrast_weight(contrast_weight: float) -> float:
    contrast_weight = float(contrast_weight)
    if not np.isfinite(contrast_weight) or not 0.0 <= contrast_weight <= 1.0:
        raise ValueError("contrast_weight must lie in [0, 1]")
    return contrast_weight


def _validate_balance_weight(balance_weight: float) -> float:
    balance_weight = float(balance_weight)
    if not np.isfinite(balance_weight) or balance_weight < 0.0:
        raise ValueError("balance_weight must be finite and non-negative")
    return balance_weight


def _iter_leaves(node: Any):
    children = list(getattr(node, "children", []) or [])
    if not children:
        yield node
        return
    for child in children:
        yield from _iter_leaves(child)


def boundary_aware_tree_objective(
    root: Any,
    affinity: np.ndarray,
    boundary_contrast: Sequence[float],
    *,
    contrast_weight: float,
    balance_weight: float = 0.0,
) -> BoundaryAwareDiagnostics:
    """Recompute the complete shared objective for any ordered binary tree."""
    affinity = validate_affinity(affinity)
    n = len(affinity)
    contrast = _validate_boundary_contrast(boundary_contrast, n)
    weight = _validate_contrast_weight(contrast_weight)
    balance = _validate_balance_weight(balance_weight)
    prefix = _prefix2d(affinity)
    cursor = [0]

    def walk(node: Any) -> tuple[int, int, float, float, float]:
        children = list(getattr(node, "children", []) or [])
        if not children:
            start = cursor[0]
            cursor[0] += 1
            return start, start + 1, 0.0, 0.0, 0.0
        if len(children) != 2:
            raise ValueError("boundary-aware objective expects a binary tree")
        li, lk, left_affinity, left_boundary, left_balance = walk(children[0])
        rk, rj, right_affinity, right_boundary, right_balance = walk(children[1])
        if lk != rk:
            raise ValueError("tree leaves are not temporally contiguous")
        cross = _rectangle_sum(prefix, li, lk, rk, rj)
        revenue = left_affinity + right_affinity + (n - (rj - li)) * cross
        reward = (left_boundary + right_boundary
                  + hierarchical_span_weight(rj - li, n) * contrast[lk - 1] / max(1, n - 1))
        imbalance = abs((lk - li) - (rj - rk)) / (rj - li)
        penalty = left_balance + right_balance + imbalance * imbalance / max(1, n - 1)
        return li, rj, float(revenue), float(reward), float(penalty)

    start, end, revenue, reward, penalty = walk(root)
    if start != 0 or end != n or cursor[0] != n:
        raise ValueError("tree leaf count does not match affinity")
    denominator = _normalization_denominator(affinity)
    normalized = revenue / denominator if denominator > 0 else 0.0
    root_children = list(getattr(root, "children", []) or [])
    root_split = sum(1 for _ in _iter_leaves(root_children[0])) if root_children else -1
    return BoundaryAwareDiagnostics(
        total_objective=((1.0 - weight) * normalized + weight * reward
                         - balance * penalty),
        affinity_revenue=revenue,
        normalized_affinity_revenue=normalized,
        boundary_reward=reward,
        balance_penalty=penalty,
        contrast_weight=weight,
        balance_weight=balance,
        n_bins=n,
        evaluated_splits=0,
        elapsed_seconds=0.0,
        root_split=int(root_split),
        tie_count=0,
    )


def optimal_boundary_aware_tree(
    matrix: np.ndarray,
    bounds: Sequence[float],
    affinity: np.ndarray,
    boundary_contrast: Sequence[float],
    node_type: type,
    *,
    contrast_weight: float,
    balance_weight: float = 0.0,
    tie_break: TieBreak = "midpoint",
    tie_tolerance: float = 1e-12,
    max_bins: int | None = None,
) -> tuple[Any, BoundaryAwareDiagnostics]:
    """Exactly maximise normalized affinity revenue plus span-weighted contrast."""
    matrix, edges = _validated_bounds(matrix, bounds)
    affinity = validate_affinity(affinity)
    n = len(matrix)
    if affinity.shape != (n, n):
        raise ValueError("affinity size must match the number of bins")
    contrast = _validate_boundary_contrast(boundary_contrast, n)
    weight = _validate_contrast_weight(contrast_weight)
    balance = _validate_balance_weight(balance_weight)
    if max_bins is not None and n > max_bins:
        raise ValueError(f"{n} bins exceeds max_bins={max_bins}")
    if tie_break not in {"earliest", "midpoint", "latest"}:
        raise ValueError("unknown tie_break")

    started = time.perf_counter()
    prefix = _prefix2d(affinity)
    feature_prefix = np.vstack([np.zeros(matrix.shape[1]), np.cumsum(matrix, axis=0)])
    denominator = _normalization_denominator(affinity)
    objective = np.full((n + 1, n + 1), -np.inf, dtype=float)
    split = np.full((n + 1, n + 1), -1, dtype=int)
    for i in range(n):
        objective[i, i + 1] = 0.0
    evaluated = 0
    tie_count = 0

    for length in range(2, n + 1):
        coefficient = n - length
        span_factor = hierarchical_span_weight(length, n) / max(1, n - 1)
        for i in range(0, n - length + 1):
            j = i + length
            ks = np.arange(i + 1, j)
            cross = (prefix[ks, j] - prefix[i, j]
                     - prefix[ks, ks] + prefix[i, ks])
            affinity_local = (coefficient * cross / denominator
                              if denominator > 0 else np.zeros(len(ks)))
            boundary_local = span_factor * contrast[ks - 1]
            imbalance = np.abs((ks - i) - (j - ks)) / length
            balance_local = imbalance * imbalance / max(1, n - 1)
            local = ((1.0 - weight) * affinity_local + weight * boundary_local
                     - balance * balance_local)
            candidates = objective[i, ks] + objective[ks, j] + local
            evaluated += len(ks)
            best = float(candidates.max())
            tied = np.flatnonzero(candidates >= best - tie_tolerance)
            if len(tied) > 1:
                tie_count += 1
            if tie_break == "earliest":
                chosen = int(tied[0])
            elif tie_break == "latest":
                chosen = int(tied[-1])
            else:
                midpoint = (i + j) / 2.0
                chosen = min((int(index) for index in tied),
                             key=lambda index: (abs(float(ks[index]) - midpoint), int(ks[index])))
            split[i, j] = int(ks[chosen])
            objective[i, j] = float(candidates[chosen])

    merge_order = [0]

    def build(i: int, j: int) -> Any:
        feature = (feature_prefix[j] - feature_prefix[i]).copy()
        if j == i + 1:
            node = _make_node(node_type, edges[i], edges[j], feature, [])
            node.merge_revenue = 0.0
            node.boundary_contrast = 0.0
            node.hierarchical_boundary_reward = 0.0
            node.subtree_objective = 0.0
            return node
        k = int(split[i, j])
        left, right = build(i, k), build(k, j)
        node = _make_node(node_type, edges[i], edges[j], feature, [left, right], merge_order[0])
        merge_order[0] += 1
        cross = _rectangle_sum(prefix, i, k, k, j)
        node.child_mean_affinity = cross / ((k - i) * (j - k))
        node.merge_revenue = float((n - (j - i)) * cross)
        node.boundary_contrast = float(contrast[k - 1])
        node.hierarchical_boundary_reward = float(
            hierarchical_span_weight(j - i, n) * contrast[k - 1] / max(1, n - 1))
        imbalance = abs((k - i) - (j - k)) / (j - i)
        node.balance_penalty = float(imbalance * imbalance / max(1, n - 1))
        node.merge_objective = float(
            objective[i, j] - objective[i, k] - objective[k, j])
        node.subtree_objective = float(objective[i, j])
        return node

    root = build(0, n)
    components = boundary_aware_tree_objective(
        root, affinity, contrast, contrast_weight=weight,
        balance_weight=balance)
    diagnostics = BoundaryAwareDiagnostics(
        total_objective=float(objective[0, n]) if n > 1 else 0.0,
        affinity_revenue=components.affinity_revenue,
        normalized_affinity_revenue=components.normalized_affinity_revenue,
        boundary_reward=components.boundary_reward,
        balance_penalty=components.balance_penalty,
        contrast_weight=weight,
        balance_weight=balance,
        n_bins=n,
        evaluated_splits=evaluated,
        elapsed_seconds=time.perf_counter() - started,
        root_split=int(split[0, n]) if n > 1 else -1,
        tie_count=tie_count,
    )
    return root, diagnostics


def greedy_boundary_aware_tree(
    matrix: np.ndarray,
    bounds: Sequence[float],
    affinity: np.ndarray,
    boundary_contrast: Sequence[float],
    node_type: type,
    *,
    contrast_weight: float,
    balance_weight: float = 0.0,
) -> Any:
    """Adjacent bottom-up heuristic using high affinity and low contrast."""
    matrix, edges = _validated_bounds(matrix, bounds)
    affinity = validate_affinity(affinity)
    n = len(matrix)
    if affinity.shape != (n, n):
        raise ValueError("affinity size must match the number of bins")
    contrast = _validate_boundary_contrast(boundary_contrast, n)
    weight = _validate_contrast_weight(contrast_weight)
    balance = _validate_balance_weight(balance_weight)
    prefix = _prefix2d(affinity)
    clusters: list[tuple[int, int, Any]] = [
        (i, i + 1, _make_node(node_type, edges[i], edges[i + 1], matrix[i].copy(), []))
        for i in range(n)
    ]
    merge_order = 0
    while len(clusters) > 1:
        priorities: list[float] = []
        means: list[float] = []
        for left, right in zip(clusters[:-1], clusters[1:]):
            cross = _rectangle_sum(prefix, left[0], left[1], right[0], right[1])
            mean_affinity = cross / ((left[1] - left[0]) * (right[1] - right[0]))
            means.append(float(mean_affinity))
            parent_length = right[1] - left[0]
            imbalance = abs((left[1] - left[0]) - (right[1] - right[0])) / parent_length
            priorities.append((1.0 - weight) * mean_affinity
                              - weight * contrast[left[1] - 1]
                              - balance * imbalance * imbalance)
        index = int(np.argmax(priorities))
        i, k, left_node = clusters[index]
        _, j, right_node = clusters[index + 1]
        feature = np.asarray(left_node.feature) + np.asarray(right_node.feature)
        node = _make_node(node_type, edges[i], edges[j], feature,
                          [left_node, right_node], merge_order)
        node.child_mean_affinity = means[index]
        node.boundary_contrast = float(contrast[k - 1])
        imbalance = abs((k - i) - (j - k)) / (j - i)
        node.balance_penalty = float(imbalance * imbalance / max(1, n - 1))
        node.greedy_merge_priority = float(priorities[index])
        merge_order += 1
        clusters[index:index + 2] = [(i, j, node)]
    return clusters[0][2]
