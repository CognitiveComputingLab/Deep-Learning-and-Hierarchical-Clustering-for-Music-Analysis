"""Training-only simple parametric distance for explicit form hierarchies."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from boundary_contrast import (
    BoundaryContrastScorer,
    BoundaryFeatureMatrix,
    RobustContrastScaler,
    bounds_to_edges,
)
from parametric_distances import CalibratedRepresentationMixture
from shared_annotation_format import SharedPiece


def hierarchical_boundary_targets(piece: SharedPiece, bounds: Sequence[float],
                                  n_bins: int, *, tree_id: str = "form") -> np.ndarray:
    """Map explicit hierarchy boundaries to graded candidate-bin strengths.

    Level-one boundaries receive strength one. Finer boundaries decrease
    linearly but remain positive. Coincident boundaries retain the coarsest
    (strongest) claim.
    """
    edges = bounds_to_edges(bounds, n_bins)
    internal = edges[1:-1]
    targets = np.zeros(max(0, n_bins - 1), dtype=float)
    items = [item for item in piece.annotations
             if item.kind == "hierarchical_segment" and item.tree_id == tree_id
             and item.level is not None and item.level > 0
             and item.start_qb is not None and item.end_qb is not None]
    if not items:
        raise ValueError(f"{piece.piece_id} has no aligned hierarchy boundaries")
    maximum = max(int(item.level) for item in items)
    duration = float(piece.duration_qb)
    for item in items:
        strength = (maximum - int(item.level) + 1) / maximum
        for boundary in (float(item.start_qb), float(item.end_qb)):
            if len(internal) and 0 < boundary < duration:
                index = int(np.argmin(np.abs(internal - boundary)))
                targets[index] = max(targets[index], strength)
    return targets


def work_balanced_soft_examples(
    pieces: Sequence[Mapping[str, Any]],
    scaler: RobustContrastScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Use all candidates while giving every work and class equal loss mass."""
    calibrated = []
    targets = []
    works = []
    audit = []
    for item in pieces:
        features = item["features"]
        if not isinstance(features, BoundaryFeatureMatrix):
            raise TypeError("features must be BoundaryFeatureMatrix")
        values = scaler.transform(features)
        target = np.asarray(item["targets"], dtype=float)
        if target.shape != (len(values),) or np.any((target < 0) | (target > 1)):
            raise ValueError("targets must align with candidate boundaries")
        calibrated.append(values)
        targets.append(target)
        works.extend([str(item["work"])] * len(target))
        for index, value in enumerate(target, 1):
            audit.append({
                "work": str(item["work"]), "piece": str(item["piece"]),
                "split_index": index, "target_strength": float(value),
                "class": "boundary" if value > 0 else "non_boundary",
            })
    x = np.vstack(calibrated)
    y = np.concatenate(targets)
    works = np.asarray(works, dtype=object)
    weights = np.zeros(len(y), dtype=float)
    for work in sorted(set(works)):
        selected_work = works == work
        for positive in (False, True):
            selected = selected_work & ((y > 0) == positive)
            count = int(selected.sum())
            if count:
                weights[selected] = 0.5 / count
    if not np.any(y > 0) or not np.any(y == 0) or weights.sum() <= 0:
        raise ValueError("training needs boundary and non-boundary candidates")
    weights /= weights.sum()
    return x, y, weights, audit


def fit_hierarchical_parametric_scorer(
    scaler: RobustContrastScaler,
    calibrated: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
) -> tuple[BoundaryContrastScorer, list[dict[str, float]]]:
    """Fit non-negative distance/context weights to graded hierarchy targets."""
    x = np.asarray(calibrated, dtype=float)
    y = np.asarray(targets, dtype=float)
    mass = np.asarray(sample_weights, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(scaler.names):
        raise ValueError("calibrated features have the wrong shape")
    if y.shape != (len(x),) or mass.shape != y.shape:
        raise ValueError("targets/weights must align with features")
    if np.any((y < 0) | (y > 1)) or not np.all(np.isfinite(x)):
        raise ValueError("invalid soft-label training data")
    mass = mass / mass.sum()

    dimension = x.shape[1]
    parameters = np.r_[np.zeros(dimension), 0.0, 0.0]
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    history = []

    def sigmoid(value):
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))

    def softmax(value):
        shifted = value - np.max(value)
        result = np.exp(shifted)
        return result / result.sum()

    for epoch in range(1, epochs + 1):
        feature_weights = softmax(parameters[:dimension])
        intercept = float(parameters[dimension])
        raw_gain = float(parameters[dimension + 1])
        gain = float(np.logaddexp(0.0, raw_gain)) + 1e-6
        mean = x @ feature_weights
        probability = sigmoid(intercept + gain * (mean - 0.5))
        tiny = 1e-12
        loss = -float(np.sum(mass * (
            y * np.log(probability + tiny)
            + (1.0 - y) * np.log(1.0 - probability + tiny))))
        loss += regularization * float(np.sum(
            (feature_weights - 1.0 / dimension) ** 2))

        dlinear = mass * (probability - y)
        gradient_weights = x.T @ (dlinear * gain)
        gradient_weights += 2 * regularization * (
            feature_weights - 1.0 / dimension)
        gradient_logits = feature_weights * (
            gradient_weights - feature_weights @ gradient_weights)
        gradient_intercept = float(dlinear.sum())
        gradient_gain = float(np.sum(dlinear * (mean - 0.5))) * float(
            sigmoid(raw_gain))
        gradient = np.r_[gradient_logits, gradient_intercept, gradient_gain]
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        corrected_first = first / (1 - 0.9 ** epoch)
        corrected_second = second / (1 - 0.999 ** epoch)
        parameters -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            history.append({
                "epoch": epoch, "loss": loss,
                "weighted_mae": float(np.sum(mass * np.abs(probability - y))),
            })

    feature_weights = softmax(parameters[:dimension])
    scorer = BoundaryContrastScorer(
        scaler=scaler, weights=feature_weights, mode="supervised",
        intercept=float(parameters[dimension]),
        gain=float(np.logaddexp(0.0, parameters[dimension + 1])) + 1e-6)
    return scorer, history


def distance_weights_from_scorer(
    scorer: BoundaryContrastScorer,
    distance_names: Sequence[str],
) -> np.ndarray:
    """Marginalise learned context weights into one weight per distance."""
    totals = defaultdict(float)
    for name, weight in zip(scorer.scaler.names, scorer.weights):
        distance = name.rsplit("_w", 1)[0]
        totals[distance] += float(weight)
    values = np.asarray([totals[str(name)] for name in distance_names], dtype=float)
    if values.sum() <= 0:
        raise ValueError("scorer contains no requested distance weights")
    return values / values.sum()


def estimate_representation_scales(
    pieces: Sequence[Mapping[str, Any]],
    distance_names: Sequence[str],
) -> np.ndarray:
    """Training-only non-zero median adjacent-bin scales."""
    scales = []
    for name in distance_names:
        values = []
        for item in pieces:
            representation = np.asarray(
                item["distance_specs"][name].transform_batch(item["matrix"]),
                dtype=float)
            if len(representation) > 1:
                values.extend(np.linalg.norm(
                    np.diff(representation, axis=0), axis=1).tolist())
        positive = np.asarray(values, dtype=float)
        positive = positive[np.isfinite(positive) & (positive > 1e-12)]
        scales.append(float(np.median(positive)) if len(positive) else 1.0)
    return np.asarray(scales, dtype=float)


def make_parametric_distance(item: Mapping[str, Any],
                             distance_names: Sequence[str],
                             weights: Sequence[float],
                             scales: Sequence[float]):
    specifications = {
        name: item["distance_specs"][name] for name in distance_names}
    return CalibratedRepresentationMixture.from_specs(
        specifications, weights, scales)


def estimate_kernel_scale(
    pieces: Sequence[Mapping[str, Any]],
    distance_names: Sequence[str],
    weights: Sequence[float],
    scales: Sequence[float],
) -> float:
    values = []
    for item in pieces:
        distance = make_parametric_distance(
            item, distance_names, weights, scales)
        representation = distance.transform_batch(item["matrix"])
        if len(representation) > 1:
            values.extend(np.linalg.norm(
                np.diff(representation, axis=0), axis=1).tolist())
    positive = np.asarray(values, dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 1e-12)]
    return float(np.median(positive)) if len(positive) else 1.0
