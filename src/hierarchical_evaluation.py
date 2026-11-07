"""Tree-aware evaluation for ordered musical segment hierarchies.

The T-measure implementation follows McFee, Nieto, and Bello (ISMIR 2015),
but computes LCA rankings directly at the clustering leaves. The paper
explicitly permits beat/onset-aligned (rather than uniformly sampled) frames.
This avoids flattening a genuine Taking Form hierarchy and permits reference
and estimated trees to have unequal depth.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from shared_annotation_format import SharedPiece
from shared_evaluation import boundary_reference


def ordered_leaves(root) -> list:
    """Return leaves in temporal order and validate binary contiguity."""
    leaves = []

    def walk(node):
        children = list(getattr(node, "children", []) or [])
        if not children:
            leaves.append(node)
            return
        if len(children) != 2:
            raise ValueError("tree-aware evaluation requires a binary tree")
        if abs(float(children[0].end) - float(children[1].start)) > 1e-7:
            raise ValueError("tree children are not temporally contiguous")
        walk(children[0])
        walk(children[1])

    walk(root)
    if any(float(left.end) > float(right.start) + 1e-7
           for left, right in zip(leaves, leaves[1:])):
        raise ValueError("tree leaves are not temporally ordered")
    return leaves


def sample_times_from_tree(root) -> np.ndarray:
    """Use one sample at the centre of every clustering input bin."""
    leaves = ordered_leaves(root)
    return np.asarray([(float(node.start) + float(node.end)) / 2.0
                       for node in leaves], dtype=float)


def estimated_lca_depths(root, sample_times: np.ndarray | None = None) -> np.ndarray:
    """LCA depth matrix induced by an estimated ordered binary tree."""
    times = sample_times_from_tree(root) if sample_times is None else np.asarray(
        sample_times, dtype=float)
    matrix = np.zeros((len(times), len(times)), dtype=np.int32)

    def walk(node, depth):
        indices = np.flatnonzero(
            (times >= float(node.start) - 1e-8)
            & (times < float(node.end) + 1e-8))
        if len(indices):
            matrix[np.ix_(indices, indices)] = depth
        for child in list(getattr(node, "children", []) or []):
            walk(child, depth + 1)

    walk(root, 1)
    return matrix


def reference_lca_depths(piece: SharedPiece, sample_times: np.ndarray,
                         *, tree_id: str = "form",
                         max_level: int | None = None) -> np.ndarray:
    """LCA depth matrix from explicit shared-format parent/level claims."""
    times = np.asarray(sample_times, dtype=float)
    items = [item for item in piece.annotations
             if item.kind == "hierarchical_segment"
             and item.tree_id == tree_id
             and item.start_qb is not None and item.end_qb is not None
             and (max_level is None or item.level is None
                  or item.level <= max_level)]
    if not items:
        raise ValueError(f"{piece.piece_id} has no aligned {tree_id!r} hierarchy")
    levels = sorted({int(item.level or 0) for item in items})
    rank = {level: index + 1 for index, level in enumerate(levels)}
    matrix = np.zeros((len(times), len(times)), dtype=np.int32)
    for item in sorted(items, key=lambda value: (int(value.level or 0),
                                                  float(value.start_qb))):
        indices = np.flatnonzero(
            (times >= float(item.start_qb) - 1e-8)
            & (times < float(item.end_qb) + 1e-8))
        if len(indices):
            matrix[np.ix_(indices, indices)] = rank[int(item.level or 0)]
    return matrix


def _count_inversions(lower: np.ndarray, upper: np.ndarray) -> int:
    """Count pairs where an estimated lower rank is not below an upper rank."""
    low_values, low_counts = np.unique(lower, return_counts=True)
    high_values, high_counts = np.unique(upper, return_counts=True)
    inversions = 0
    i = j = 0
    while i < len(low_values) and j < len(high_values):
        if low_values[i] < high_values[j]:
            i += 1
        else:
            inversions += int(np.sum(low_counts[i:]) * high_counts[j])
            j += 1
    return inversions


def _compare_rankings(reference: np.ndarray, estimate: np.ndarray,
                      *, transitive: bool) -> tuple[int, float]:
    order = np.argsort(reference, kind="stable")
    ref = reference[order]
    est = estimate[order]
    levels, positions, counts = np.unique(
        ref, return_index=True, return_counts=True)
    slices = defaultdict(lambda: slice(0), {
        level: slice(int(start), int(start + count))
        for level, start, count in zip(levels, positions, counts)})
    frequencies = defaultdict(int, {
        level: int(count) for level, count in zip(levels, counts)})
    if transitive:
        pairs = list(combinations(levels, 2))
    else:
        pairs = [(level, level + 1) for level in levels]
    normalizer = float(sum(frequencies[a] * frequencies[b] for a, b in pairs))
    if normalizer == 0:
        return 0, 0.0
    inversions = sum(_count_inversions(est[slices[a]], est[slices[b]])
                     for a, b in pairs)
    return int(inversions), normalizer


def _gauc(reference: np.ndarray, estimate: np.ndarray, sample_times: np.ndarray,
          *, transitive: bool, window_qb: float | None) -> float:
    if reference.shape != estimate.shape or reference.shape[0] != len(sample_times):
        raise ValueError("reference, estimate, and sample times must align")
    scores = []
    for query in range(len(sample_times)):
        mask = np.ones(len(sample_times), dtype=bool)
        mask[query] = False
        if window_qb is not None:
            mask &= np.abs(sample_times - sample_times[query]) <= float(window_qb)
        inversions, normalizer = _compare_rankings(
            reference[query, mask], estimate[query, mask],
            transitive=transitive)
        if normalizer:
            scores.append(1.0 - inversions / normalizer)
    return float(np.mean(scores)) if scores else 0.0


def _f_measure(precision: float, recall: float) -> float:
    return (2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, sufficient for Spearman diagnostics."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def tree_measure(root, piece: SharedPiece, *, tree_id: str = "form",
                 max_level: int | None = None,
                 window_qb: float | None = None) -> dict[str, float | int]:
    """Compute reduced/full T-measures and complementary LCA diagnostics."""
    times = sample_times_from_tree(root)
    reference = reference_lca_depths(
        piece, times, tree_id=tree_id, max_level=max_level)
    estimate = estimated_lca_depths(root, times)
    result: dict[str, float | int] = {"tree_sample_count": len(times)}
    for name, transitive in (("reduced", False), ("full", True)):
        recall = _gauc(reference, estimate, times, transitive=transitive,
                       window_qb=window_qb)
        precision = _gauc(estimate, reference, times, transitive=transitive,
                          window_qb=window_qb)
        result[f"t_precision_{name}"] = precision
        result[f"t_recall_{name}"] = recall
        result[f"t_measure_{name}"] = _f_measure(precision, recall)

    upper = np.triu_indices(len(times), 1)
    ref_pairs = reference[upper].astype(float)
    est_pairs = estimate[upper].astype(float)
    ref_ranks = _rankdata(ref_pairs)
    est_ranks = _rankdata(est_pairs)
    if len(ref_pairs) and np.std(ref_ranks) > 0 and np.std(est_ranks) > 0:
        result["lca_spearman"] = float(np.corrcoef(ref_ranks, est_ranks)[0, 1])
    else:
        result["lca_spearman"] = float("nan")
    ref_scale = max(1.0, float(np.max(reference)))
    est_scale = max(1.0, float(np.max(estimate)))
    result["lca_normalized_mae"] = float(np.mean(
        np.abs(ref_pairs / ref_scale - est_pairs / est_scale)
    )) if len(ref_pairs) else 0.0
    return result


def full_tree_measure(
    root,
    piece: SharedPiece,
    *,
    tree_id: str = "form",
    max_level: int | None = None,
    window_qb: float | None = None,
    prepared_reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Fast full T-measure path for training-only hyperparameter selection.

    It omits reduced T-measure and auxiliary correlations. A prepared
    (sample_times, reference_lca) pair may be reused across candidate trees
    with the same ordered leaves.
    """
    if prepared_reference is None:
        times = sample_times_from_tree(root)
        reference = reference_lca_depths(
            piece, times, tree_id=tree_id, max_level=max_level)
    else:
        times = np.asarray(prepared_reference[0], dtype=float)
        reference = np.asarray(prepared_reference[1], dtype=np.int32)
    estimate = estimated_lca_depths(root, times)
    recall = _gauc(reference, estimate, times, transitive=True,
                   window_qb=window_qb)
    precision = _gauc(estimate, reference, times, transitive=True,
                      window_qb=window_qb)
    return {
        "tree_sample_count": len(times),
        "t_precision_full": precision,
        "t_recall_full": recall,
        "t_measure_full": _f_measure(precision, recall),
    }


def equal_budget_level_scores(root, piece: SharedPiece, *, max_level: int,
                              tolerance_qb: float) -> dict[str, float | int]:
    """Boundary F1 with prediction count fixed to that reference level."""
    from greedy_evaluation import boundary_scores, collect_prominent_splits

    reference = boundary_reference(piece, layers=["form"], max_level=max_level)
    reference = [value for value in reference
                 if 0 < value < float(piece.duration_qb)]
    predicted = collect_prominent_splits(root, len(reference))
    score = boundary_scores(predicted, reference, tolerance_qb)
    return {**score, "prediction_budget": len(reference),
            "predicted_boundary_count": len(predicted),
            "reference_boundary_count": len(reference)}
