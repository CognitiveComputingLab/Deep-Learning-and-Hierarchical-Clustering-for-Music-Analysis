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
        temporal_contrast = getattr(
            distance_specs[name], "boundary_contrast_batch", None)
        if temporal_contrast is not None:
            values = np.asarray(temporal_contrast(matrix, contexts), dtype=float)
            expected = (max(0, len(matrix) - 1), len(contexts))
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(
                    "boundary_contrast_batch must return finite (n-1, contexts) values")
            result[:, column:column + len(contexts)] = values
            column += len(contexts)
            continue
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
class ProbabilityCalibrator:
    """Monotone scalar logistic calibration for boundary probabilities."""

    intercept: float
    slope: float
    score_location: float
    score_scale: float
    epsilon: float = 1e-6
    fit_scope: str = "training_works_all_nonignored_candidates"

    def __post_init__(self) -> None:
        values = (self.intercept, self.slope, self.score_location,
                  self.score_scale, self.epsilon)
        if not np.all(np.isfinite(values)):
            raise ValueError("calibrator parameters must be finite")
        if self.slope < 0 or self.score_scale <= 0:
            raise ValueError("calibrator slope must be non-negative and scale positive")
        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must lie strictly between zero and one half")

    def transform(self, scores: Sequence[float] | np.ndarray) -> np.ndarray:
        """Calibrate scores without fitting state from the prediction set."""
        scores = np.asarray(scores, dtype=float)
        if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must be finite probabilities in [0, 1]")
        clipped = np.clip(scores, self.epsilon, 1.0 - self.epsilon)
        log_odds = np.log(clipped) - np.log1p(-clipped)
        feature = (log_odds - self.score_location) / self.score_scale
        return _sigmoid(self.intercept + self.slope * feature)

    score = transform
    predict_proba = transform

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "monotone scalar logistic calibration on score log-odds",
            "intercept": float(self.intercept), "slope": float(self.slope),
            "score_location": float(self.score_location),
            "score_scale": float(self.score_scale), "epsilon": float(self.epsilon),
            "fit_scope": self.fit_scope,
        }


def _fit_monotone_logistic(design, labels, weights, regularization,
                           max_iter, tolerance):
    prevalence = float(np.sum(weights * labels))
    parameters = np.array([np.log(prevalence) - np.log1p(-prevalence), 0.0])

    def objective(candidate):
        probability = _sigmoid(design @ candidate)
        tiny = 1e-12
        loss = -np.sum(weights * (labels * np.log(probability + tiny)
                       + (1.0 - labels) * np.log(1.0 - probability + tiny)))
        return float(loss + 0.5 * regularization * candidate[1] ** 2)

    current = objective(parameters)
    for _ in range(max_iter):
        probability = _sigmoid(design @ parameters)
        residual = weights * (probability - labels)
        gradient = design.T @ residual
        gradient[1] += regularization * parameters[1]
        curvature = weights * probability * (1.0 - probability)
        hessian = design.T @ (curvature[:, None] * design)
        hessian[1, 1] += regularization
        hessian += np.eye(2) * 1e-12
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        if float(np.linalg.norm(step, ord=np.inf)) <= tolerance:
            break
        multiplier, accepted, change = 1.0, False, np.inf
        while multiplier >= 2.0 ** -20:
            candidate = parameters - multiplier * step
            candidate[1] = max(0.0, float(candidate[1]))
            candidate_objective = objective(candidate)
            if candidate_objective <= current + 1e-14:
                change = abs(current - candidate_objective)
                parameters, current = candidate, candidate_objective
                accepted = True
                break
            multiplier *= 0.5
        if not accepted or change <= tolerance:
            break
    return parameters


def fit_probability_calibrator(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    sample_weights: Sequence[float] | np.ndarray | None = None,
    *,
    regularization: float = 1e-4,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    epsilon: float = 1e-6,
) -> ProbabilityCalibrator:
    """Fit deterministic Platt-style calibration on training candidates.

    Pass all eligible candidates from the current training works with their
    natural boundary prevalence, not the class-balanced hard-negative subset.
    Inner-validation and outer-test labels must never be supplied here.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if scores.ndim != 1 or labels.shape != scores.shape or not len(scores):
        raise ValueError("scores and labels must be aligned non-empty vectors")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must be finite probabilities in [0, 1]")
    if (not np.all(np.isfinite(labels))
            or not set(np.unique(labels)).issubset({0.0, 1.0})):
        raise ValueError("labels must contain only zero and one")
    if sample_weights is None:
        weights = np.ones(len(scores), dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != scores.shape:
            raise ValueError("sample_weights must align with scores")
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or float(weights.sum()) <= 0):
        raise ValueError("sample_weights must be finite, non-negative and non-zero")
    if regularization < 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError("invalid calibrator optimisation settings")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must lie strictly between zero and one half")
    class_mass = [weights[labels == value].sum() for value in (0.0, 1.0)]
    if any(value <= 0 for value in class_mass):
        raise ValueError("calibration requires positive-weight examples of both classes")

    weights = weights / float(weights.sum())
    clipped = np.clip(scores, epsilon, 1.0 - epsilon)
    log_odds = np.log(clipped) - np.log1p(-clipped)
    location = float(np.sum(weights * log_odds))
    variance = float(np.sum(weights * (log_odds - location) ** 2))
    scale = max(float(np.sqrt(variance)), 1e-8)
    design = np.column_stack((np.ones(len(scores)), (log_odds - location) / scale))
    parameters = _fit_monotone_logistic(
        design, labels, weights, regularization, max_iter, tolerance)
    return ProbabilityCalibrator(
        intercept=float(parameters[0]), slope=float(parameters[1]),
        score_location=location, score_scale=scale, epsilon=epsilon)


@dataclass(frozen=True)
class SupervisedContrastExample:
    work: str
    piece: str
    split_index: int
    features: np.ndarray
    label: int
    ignored: bool = False


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


def tolerance_aware_boundary_labels(
    bounds: Sequence[float],
    reference_qb: Sequence[float],
    n_bins: int,
    *,
    ignore_tolerance_qb: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-positive labels and ambiguous-neighbour ignore mask."""
    if not np.isfinite(ignore_tolerance_qb) or ignore_tolerance_qb < 0:
        raise ValueError("ignore_tolerance_qb must be finite and non-negative")
    edges = bounds_to_edges(bounds, n_bins)
    internal = edges[1:-1]
    references = np.asarray(
        sorted(set(float(value) for value in reference_qb)), dtype=float)
    if not np.all(np.isfinite(references)):
        raise ValueError("reference boundaries must be finite")
    labels = np.zeros(max(0, n_bins - 1), dtype=int)
    for boundary in references:
        if len(internal):
            labels[int(np.argmin(np.abs(internal - boundary)))] = 1
    ignored = np.zeros_like(labels, dtype=bool)
    if ignore_tolerance_qb > 0 and len(internal) and len(references):
        distance = np.min(
            np.abs(internal[:, None] - references[None, :]), axis=1)
        ignored = (distance <= ignore_tolerance_qb) & (labels == 0)
    return labels, ignored


def make_supervised_examples(
    pieces: Sequence[Mapping[str, Any]],
    scaler: RobustContrastScaler,
    *,
    ignore_tolerance_qb: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Return work-balanced positives and deterministic hard negatives.

    Each input mapping supplies ``work``, ``piece``, ``features``, ``bounds``
    and ``reference``. For every work all positives are retained; up to the
    same number of negatives with the largest unsupervised contrast are selected.
    Sample weights make each class within each work contribute one half.
    Non-positive candidates within ``ignore_tolerance_qb`` of an annotation
    are audited but excluded from hard-negative sampling. Zero preserves the
    legacy training-array behaviour.
    """
    rows: list[SupervisedContrastExample] = []
    calibrated_by_piece: dict[str, np.ndarray] = {}
    for item in pieces:
        features = item["features"]
        if not isinstance(features, BoundaryFeatureMatrix):
            raise TypeError("features must be BoundaryFeatureMatrix instances")
        labels, ignored = tolerance_aware_boundary_labels(
            item["bounds"], item["reference"], len(features.values) + 1,
            ignore_tolerance_qb=ignore_tolerance_qb)
        calibrated_by_piece[str(item["piece"])] = scaler.transform(features)
        for split_index, (vector, label, is_ignored) in enumerate(
                zip(features.values, labels, ignored), 1):
            rows.append(SupervisedContrastExample(
                str(item["work"]), str(item["piece"]), split_index,
                np.asarray(vector, dtype=float), int(label), bool(is_ignored)))

    selected: list[SupervisedContrastExample] = []
    for work in sorted({row.work for row in rows}):
        positives = [row for row in rows if row.work == work and row.label == 1]
        negatives = [row for row in rows
                     if row.work == work and row.label == 0 and not row.ignored]
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
    selected_indices = {
        (row.work, row.piece, row.split_index): index
        for index, row in enumerate(selected)
    }
    work_counts = {}
    for work in sorted({row.work for row in rows}):
        work_counts[work] = {
            "work_positive_count": sum(
                row.label == 1 for row in rows if row.work == work),
            "work_eligible_negative_count": sum(
                row.label == 0 and not row.ignored
                for row in rows if row.work == work),
            "work_selected_negative_count": sum(
                row.work == work and row.label == 0 for row in selected),
        }
    audit = []
    for row in sorted(rows, key=lambda value: (
            value.work, value.piece, value.split_index)):
        key = (row.work, row.piece, row.split_index)
        selected_index = selected_indices.get(key)
        is_selected = selected_index is not None
        if row.ignored:
            status = "ignored_tolerance"
        elif row.label == 1 and is_selected:
            status = "selected_positive"
        elif row.label == 1:
            status = "unselected_positive_no_balanced_work"
        elif is_selected:
            status = "selected_hard_negative"
        else:
            status = "unselected_negative"
        audit.append({
            "work": row.work, "piece": row.piece,
            "split_index": row.split_index,
            "label": None if row.ignored else row.label,
            "selected": is_selected, "ignored": row.ignored,
            "status": status,
            "sample_weight": (float(sample_weights[selected_index])
                              if is_selected else 0.0),
            "ignore_tolerance_qb": float(ignore_tolerance_qb),
            **work_counts[row.work],
        })
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
