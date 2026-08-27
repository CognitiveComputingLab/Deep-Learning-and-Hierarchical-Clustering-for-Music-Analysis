"""Metrics shared by Taking Form and Algomus zero-shot evaluation."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from annotated_piece import intervals_iou
from greedy_evaluation import boundary_scores
from neural_clustering import average_precision


def spans_from_boundaries(boundaries: Iterable[float], duration_qb: float
                          ) -> list[tuple[float, float]]:
    edges = [0.0] + sorted({float(value) for value in boundaries
                            if 0 < value < duration_qb}) + [float(duration_qb)]
    return list(zip(edges[:-1], edges[1:]))


def span_scores(predicted: Sequence[tuple[float, float]],
                reference: Sequence[tuple[float, float]],
                tolerance_qb: float) -> dict[str, float | int]:
    """Ordered one-to-one span matching, maximizing matches then timing fit."""
    predicted = sorted((float(a), float(b)) for a, b in predicted)
    reference = sorted((float(a), float(b)) for a, b in reference)
    m, n = len(predicted), len(reference)
    # Each state stores (matches, negative endpoint error).
    scores = [[(0, 0.0) for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            candidates = [scores[i - 1][j], scores[i][j - 1]]
            pa, pb = predicted[i - 1]
            ra, rb = reference[j - 1]
            if abs(pa - ra) <= tolerance_qb and abs(pb - rb) <= tolerance_qb:
                previous = scores[i - 1][j - 1]
                candidates.append((previous[0] + 1,
                                   previous[1] - abs(pa - ra) - abs(pb - rb)))
            scores[i][j] = max(candidates)
    tp = scores[m][n][0]
    precision = tp / m if m else 0.0
    recall = tp / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fp": m - tp, "fn": n - tp,
        "precision": precision, "recall": recall, "f1": f1,
    }


def boundary_ap_from_times(candidate_times: Sequence[float],
                           candidate_scores: Sequence[float],
                           references: Sequence[float],
                           tolerance_qb: float) -> float:
    """Threshold-free AP with one-to-one labels projected to candidate edges."""
    times = np.asarray(candidate_times, dtype=float)
    scores = np.asarray(candidate_scores, dtype=float)
    if times.shape != scores.shape:
        raise ValueError("candidate times and scores must match")
    labels = np.zeros(len(times), dtype=int)
    available = set(range(len(times)))
    for reference in sorted(references):
        eligible = [index for index in available
                    if abs(times[index] - reference) <= tolerance_qb]
        if eligible:
            chosen = min(eligible, key=lambda index: (
                abs(times[index] - reference), index))
            labels[chosen] = 1
            available.remove(chosen)
    return average_precision(labels, scores)


def best_tree_span_iou(predicted_spans: Sequence[tuple[float, float]],
                       reference_intervals: Sequence[tuple[float, float]]) -> float:
    """Mean best IoU of each reference interval with a predicted tree segment."""
    if not reference_intervals:
        return float("nan")
    values = []
    for reference in reference_intervals:
        values.append(max(
            (intervals_iou([span], [reference]) for span in predicted_spans),
            default=0.0))
    return float(np.mean(values))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
