from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boundary_contrast import BoundaryFeatureMatrix, RobustContrastScaler
from greedy_clustering import DistanceSpec
from hierarchical_parametric import (
    distance_weights_from_scorer,
    fit_hierarchical_parametric_scorer,
    hierarchical_boundary_targets,
    work_balanced_soft_examples,
)
from parametric_distances import CalibratedRepresentationMixture
from shared_annotation_format import SharedAnnotation, SharedPiece


def _piece():
    annotations = [
        SharedAnnotation("root", "hierarchical_segment", "form", "ROOT",
                         start_qb=0, end_qb=4, level=0, tree_id="form"),
        SharedAnnotation("a", "hierarchical_segment", "form", "A",
                         start_qb=0, end_qb=2, level=1,
                         parent_id="root", tree_id="form"),
        SharedAnnotation("b", "hierarchical_segment", "form", "B",
                         start_qb=2, end_qb=4, level=1,
                         parent_id="root", tree_id="form"),
        SharedAnnotation("a1", "hierarchical_segment", "form", "a1",
                         start_qb=0, end_qb=1, level=2,
                         parent_id="a", tree_id="form"),
        SharedAnnotation("a2", "hierarchical_segment", "form", "a2",
                         start_qb=1, end_qb=2, level=2,
                         parent_id="a", tree_id="form"),
    ]
    return SharedPiece("toy", "toy", "work", "quarterbeat", annotations,
                       duration_qb=4)


def test_hierarchy_targets_preserve_coarse_boundary_strength():
    targets = hierarchical_boundary_targets(
        _piece(), np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]), 4)
    assert targets.tolist() == [0.5, 1.0, 0.0]


def test_soft_parametric_training_is_finite_and_nonnegative():
    names = ("euclidean_w1", "key_profile_w1")
    features = BoundaryFeatureMatrix(
        np.asarray([[0.1, 0.2], [2.0, 1.5], [0.0, 0.1]]),
        names, (1,), ("euclidean", "key_profile"))
    scaler = RobustContrastScaler.fit([features])
    items = [{"work": "w", "piece": "p", "features": features,
              "targets": np.asarray([0.0, 1.0, 0.0])}]
    x, y, weights, audit = work_balanced_soft_examples(items, scaler)
    scorer, history = fit_hierarchical_parametric_scorer(
        scaler, x, y, weights, epochs=25)
    assert np.isclose(scorer.weights.sum(), 1)
    assert np.all(scorer.weights >= 0)
    assert np.isfinite(history[-1]["loss"])
    assert len(audit) == 3
    marginal = distance_weights_from_scorer(
        scorer, ("euclidean", "key_profile"))
    assert np.isclose(marginal.sum(), 1)


def test_representation_mixture_scalar_and_batch_agree():
    identity = DistanceSpec("identity", lambda values: np.asarray(values, dtype=float))
    doubled = DistanceSpec("double", lambda values: 2 * np.asarray(values, dtype=float))
    mixture = CalibratedRepresentationMixture.from_specs(
        {"identity": identity, "double": doubled}, [0.25, 0.75], [1, 2])
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    right = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    batch = mixture.batch_distance(left, right)
    assert np.allclose(batch, [
        mixture(left[index], right[index]) for index in range(2)])
