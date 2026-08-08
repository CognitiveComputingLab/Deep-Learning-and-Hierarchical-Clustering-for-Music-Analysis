"""Local, interpretable contrast features for candidate music boundaries.

The functions in this module never read annotations while making a prediction.
Annotations are accepted only by :func:`make_supervised_examples`, which is a
training helper.  Robust feature calibration is deliberately represented by a
small serialisable object so that held-out pieces cannot silently re-estimate
their scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


DEFAULT_CONTRAST_DISTANCES = ("euclidean", "circle_of_fifths", "key_profile")
DEFAULT_CONTEXTS = (1, 2, 4)


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    shifted = values - float(np.max(values))
    exponentials = np.exp(shifted)
    return exponentials / float(exponentials.sum())


def _softplus(value: float) -> float:
    return float(np.logaddexp(0.0, float(value)))


def _validate_contexts(contexts: Sequence[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(int(value) for value in contexts)))
    if not result or any(value < 1 for value in result):
        raise ValueError("contexts must contain positive integers")
    return result


@dataclass(frozen=True)
class BoundaryFeatureMatrix:
    """Raw contrast values for the ``n-1`` candidate leaf boundaries."""

    values: np.ndarray
    names: tuple[str, ...]
    contexts: tuple[int, ...]
    distances: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError("feature values must have shape (n_boundaries, n_features)")
        if not np.all(np.isfinite(values)):
            raise ValueError("boundary features contain non-finite values")
        object.__setattr__(self, "values", values)


def boundary_contrast_features(
    matrix: np.ndarray,
    distance_specs: Mapping[str, Any],
    *,
    contexts: Sequence[int] = DEFAULT_CONTEXTS,
    distances: Sequence[str] = DEFAULT_CONTRAST_DISTANCES,
) -> BoundaryFeatureMatrix:
    """Compute cross-context dissimilarity minus within-context dispersion.

    Contexts are truncated at piece edges instead of dropping edge candidates.
    Within-context means exclude the diagonal.  A one-bin context therefore has
    zero internal dispersion, as specified by the boundary-aware experiment.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or len(matrix) < 1:
        raise ValueError("matrix must be a non-empty 2-D array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix contains non-finite values")
    contexts = _validate_contexts(contexts)
    distances = tuple(str(name) for name in distances)
    missing = [name for name in distances if name not in distance_specs]
    if missing:
        raise KeyError(f"missing distance specifications: {missing}")

    names = tuple(f"{name}_w{context}" for name in distances for context in contexts)
    result = np.zeros((max(0, len(matrix) - 1), len(names)), dtype=float)
    column = 0
    for name in distances:
        transform = getattr(distance_specs[name], "transform_batch", None)
        if transform is None:
            raise TypeError(f"distance specification {name!r} has no transform_batch()")
        representation = np.asarray(transform(matrix), dtype=float)
        if representation.ndim != 2 or len(representation) != len(matrix):
            raise ValueError("transform_batch must return one vector per bin")
        differences = representation[:, None, :] - representation[None, :, :]
        pairwise = np.linalg.norm(differences, axis=2)
        for context in contexts:
            for split in range(1, len(matrix)):
                left = np.arange(max(0, split - context), split)
                right = np.arange(split, min(len(matrix), split + context))
                cross = float(pairwise[np.ix_(left, right)].mean())

                def within(indices: np.ndarray) -> float:
                    if len(indices) < 2:
                        return 0.0
                    block = pairwise[np.ix_(indices, indices)]
                    return float(block[np.triu_indices(len(indices), 1)].mean())

                result[split - 1, column] = cross - 0.5 * (within(left) + within(right))
            column += 1
    return BoundaryFeatureMatrix(result, names, contexts, distances)


@dataclass(frozen=True)
class RobustContrastScaler:
    """Training-only median/MAD calibration with zero mapped exactly to zero."""

    names: tuple[str, ...]
    medians: np.ndarray
    mads: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, feature_matrices: Sequence[BoundaryFeatureMatrix]) -> "RobustContrastScaler":
        if not feature_matrices:
            raise ValueError("at least one training feature matrix is required")
        names = feature_matrices[0].names
        if any(item.names != names for item in feature_matrices):
            raise ValueError("all boundary feature matrices must use the same columns")
        nonempty = [item.values for item in feature_matrices if len(item.values)]
        if not nonempty:
            raise ValueError("training pieces contain no candidate boundaries")
        values = np.vstack(nonempty)
        medians = np.median(values, axis=0)
        mads = np.median(np.abs(values - medians), axis=0)
        positive_median = np.asarray([
            np.median(column[column > 0]) if np.any(column > 0) else 0.0
            for column in values.T
        ], dtype=float)
        scales = np.maximum.reduce((1.4826 * mads, positive_median, np.full_like(mads, 1e-8)))
        return cls(names, medians.astype(float), mads.astype(float), scales.astype(float))

    def transform(self, features: BoundaryFeatureMatrix | np.ndarray) -> np.ndarray:
        if isinstance(features, BoundaryFeatureMatrix):
            if features.names != self.names:
                raise ValueError("feature columns do not match the fitted scaler")
            values = features.values
        else:
            values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.names):
            raise ValueError("feature matrix has the wrong number of columns")
        raw = _sigmoid((values - self.medians) / self.scales)
        at_zero = _sigmoid((0.0 - self.medians) / self.scales)
        calibrated = (raw - at_zero) / np.maximum(1.0 - at_zero, 1e-12)
        return np.clip(calibrated, 0.0, 1.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "medians": self.medians.tolist(),
            "mads": self.mads.tolist(),
            "scales": self.scales.tolist(),
            "fit_scope": "training_works_only",
            "transform": "zero-anchored robust sigmoid",
        }


@dataclass(frozen=True)
class BoundaryContrastScorer:
    """Unsupervised mean or supervised non-negative logistic scorer."""

    scaler: RobustContrastScaler
    weights: np.ndarray
    mode: str = "unsupervised"
    intercept: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (len(self.scaler.names),) or np.any(weights < 0):
            raise ValueError("scorer weights must be a non-negative feature vector")
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            raise ValueError("scorer weights must be finite and have positive sum")
        if self.mode not in {"unsupervised", "supervised"}:
            raise ValueError("mode must be unsupervised or supervised")
        if not np.isfinite(self.intercept) or not np.isfinite(self.gain) or self.gain <= 0:
            raise ValueError("intercept/gain must be finite and gain must be positive")
        object.__setattr__(self, "weights", weights / weights.sum())

    @classmethod
    def unsupervised(cls, scaler: RobustContrastScaler) -> "BoundaryContrastScorer":
        return cls(scaler, np.ones(len(scaler.names), dtype=float), "unsupervised")

    def score(self, features: BoundaryFeatureMatrix | np.ndarray) -> np.ndarray:
        calibrated = self.scaler.transform(features)
        mean = calibrated @ self.weights
        if self.mode == "unsupervised":
            return np.clip(mean, 0.0, 1.0)
        return _sigmoid(self.intercept + self.gain * (mean - 0.5))

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "weights": {name: float(value) for name, value in zip(self.scaler.names, self.weights)},
            "intercept": float(self.intercept),
            "gain": float(self.gain),
            "scaler": self.scaler.as_dict(),
        }


@dataclass(frozen=True)
class SupervisedContrastExample:
    work: str
    piece: str
    split_index: int
    features: np.ndarray
    label: int


def bounds_to_edges(bounds: Sequence[float], n_bins: int | None = None) -> np.ndarray:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim == 2 and bounds.shape[1:] == (2,):
        if n_bins is not None and len(bounds) != n_bins:
            raise ValueError("bounds length does not match n_bins")
        if len(bounds) > 1 and not np.allclose(bounds[:-1, 1], bounds[1:, 0]):
            raise ValueError("interval bounds must be contiguous")
        edges = np.r_[bounds[0, 0], bounds[:, 1]]
    elif bounds.ndim == 1 and (n_bins is None or len(bounds) == n_bins + 1):
        edges = bounds
    else:
        raise ValueError("bounds must have shape (n,2) or (n+1,)")
    if len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("bounds must be finite and strictly increasing")
    return edges.astype(float)


def nearest_boundary_labels(bounds: Sequence[float], reference_qb: Sequence[float], n_bins: int) -> np.ndarray:
    """Map each annotation deterministically to its nearest internal bin edge."""
    edges = bounds_to_edges(bounds, n_bins)
    labels = np.zeros(max(0, n_bins - 1), dtype=int)
    internal = edges[1:-1]
    for boundary in sorted(set(float(value) for value in reference_qb)):
        if len(internal):
            labels[int(np.argmin(np.abs(internal - boundary)))] = 1
    return labels


def make_supervised_examples(
    pieces: Sequence[Mapping[str, Any]],
    scaler: RobustContrastScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return work-balanced positives and deterministic hard negatives.

    Each input mapping supplies ``work``, ``piece``, ``features``, ``bounds``
    and ``reference``.  For every work all positives are retained; the same
    number of negatives with the largest unsupervised contrast are selected.
    Sample weights make each class within each work contribute one half.
    """
    rows: list[SupervisedContrastExample] = []
    calibrated_by_piece: dict[str, np.ndarray] = {}
    for item in pieces:
        features = item["features"]
        if not isinstance(features, BoundaryFeatureMatrix):
            raise TypeError("features must be BoundaryFeatureMatrix instances")
        labels = nearest_boundary_labels(item["bounds"], item["reference"], len(features.values) + 1)
        calibrated_by_piece[str(item["piece"])] = scaler.transform(features)
        for split_index, (vector, label) in enumerate(zip(features.values, labels), 1):
            rows.append(SupervisedContrastExample(
                str(item["work"]), str(item["piece"]), split_index,
                np.asarray(vector, dtype=float), int(label)))

    selected: list[SupervisedContrastExample] = []
    for work in sorted({row.work for row in rows}):
        positives = [row for row in rows if row.work == work and row.label == 1]
        negatives = [row for row in rows if row.work == work and row.label == 0]
        count = min(len(positives), len(negatives))
        if not count:
            continue
        negatives = sorted(
            negatives,
            key=lambda row: (
                -float(calibrated_by_piece[row.piece][row.split_index - 1].mean()),
                row.piece,
                row.split_index,
            ),
        )[:count]
        selected.extend(sorted(positives, key=lambda row: (row.piece, row.split_index)))
        selected.extend(negatives)
    if not selected:
        raise ValueError("no balanced supervised boundary examples could be constructed")

    raw = np.vstack([row.features for row in selected])
    calibrated = scaler.transform(raw)
    labels = np.asarray([row.label for row in selected], dtype=float)
    sample_weights = np.zeros(len(selected), dtype=float)
    for work in sorted({row.work for row in selected}):
        for label in (0, 1):
            indices = [index for index, row in enumerate(selected) if row.work == work and row.label == label]
            if indices:
                sample_weights[indices] = 0.5 / len(indices)
    sample_weights /= sample_weights.sum()
    audit = [{
        "work": row.work, "piece": row.piece, "split_index": row.split_index,
        "label": row.label, "sample_weight": float(sample_weights[index]),
    } for index, row in enumerate(selected)]
    return calibrated, labels, sample_weights, audit


def fit_supervised_scorer(
    scaler: RobustContrastScaler,
    calibrated: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    epochs: int = 300,
    learning_rate: float = 0.05,
    regularization: float = 1e-3,
) -> tuple[BoundaryContrastScorer, list[dict[str, float]]]:
    """Fit a deterministic non-negative logistic scorer with NumPy Adam."""
    calibrated = np.asarray(calibrated, dtype=float)
    labels = np.asarray(labels, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)
    if calibrated.ndim != 2 or calibrated.shape[1] != len(scaler.names):
        raise ValueError("calibrated training features have the wrong shape")
    if labels.shape != (len(calibrated),) or sample_weights.shape != labels.shape:
        raise ValueError("labels/sample weights must align with training rows")
    if not set(np.unique(labels)).issubset({0.0, 1.0}) or len(np.unique(labels)) < 2:
        raise ValueError("supervised scorer requires both boundary classes")
    if epochs < 1 or learning_rate <= 0 or regularization < 0:
        raise ValueError("invalid optimiser settings")
    sample_weights = sample_weights / sample_weights.sum()

    d = calibrated.shape[1]
    parameters = np.r_[np.zeros(d), 0.0, 0.0]  # logits, intercept, raw gain
    first = np.zeros_like(parameters)
    second = np.zeros_like(parameters)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        logits = parameters[:d]
        intercept = float(parameters[d])
        raw_gain = float(parameters[d + 1])
        weights = _softmax(logits)
        gain = _softplus(raw_gain) + 1e-6
        mean = calibrated @ weights
        linear = intercept + gain * (mean - 0.5)
        probability = _sigmoid(linear)
        eps = 1e-12
        loss = -float(np.sum(sample_weights * (
            labels * np.log(probability + eps) + (1.0 - labels) * np.log(1.0 - probability + eps))))
        loss += regularization * float(np.sum((weights - 1.0 / d) ** 2))

        dlinear = sample_weights * (probability - labels)
        gradient_weights = calibrated.T @ (dlinear * gain)
        gradient_weights += 2.0 * regularization * (weights - 1.0 / d)
        gradient_logits = weights * (gradient_weights - float(weights @ gradient_weights))
        gradient_intercept = float(dlinear.sum())
        gradient_raw_gain = float(np.sum(dlinear * (mean - 0.5))) * float(_sigmoid(raw_gain))
        gradient = np.r_[gradient_logits, gradient_intercept, gradient_raw_gain]

        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        corrected_first = first / (1.0 - 0.9 ** epoch)
        corrected_second = second / (1.0 - 0.999 ** epoch)
        parameters -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            prediction = probability >= 0.5
            accuracy = float(np.sum(sample_weights * (prediction == labels)))
            history.append({"epoch": float(epoch), "loss": loss, "weighted_accuracy": accuracy})

    weights = _softmax(parameters[:d])
    scorer = BoundaryContrastScorer(
        scaler=scaler,
        weights=weights,
        mode="supervised",
        intercept=float(parameters[d]),
        gain=_softplus(float(parameters[d + 1])) + 1e-6,
    )
    return scorer, history
