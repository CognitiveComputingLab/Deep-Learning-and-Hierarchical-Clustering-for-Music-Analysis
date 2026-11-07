from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_distance import (
    NeuralPitchClassDistance,
    build_interval_training_set,
    train_distance_checkpoints,
)


def _piece(work, piece, shift=0):
    matrix = np.zeros((6, 12), dtype=float)
    matrix[:3, shift % 12] = 1.0
    matrix[3:, (shift + 6) % 12] = 1.0
    bounds = [(8.0 * index, 8.0 * (index + 1)) for index in range(6)]
    return {
        "work": work, "piece": piece, "matrix": matrix,
        "bounds": bounds, "reference": [24.0],
    }


def _trained(seed=7):
    training = build_interval_training_set(
        [_piece("w1", "p1"), _piece("w2", "p2", 2)],
        contexts=(1, 2), ignore_tolerance_qb=8.0)
    checkpoints, history = train_distance_checkpoints(
        training, checkpoint_epochs=(2,), batch_size=8, seed=seed)
    return training, checkpoints[0].distance, history


def test_training_examples_keep_all_eligible_and_ignore_tolerance_neighbours():
    training = build_interval_training_set(
        [_piece("w1", "p1"), _piece("w2", "p2", 1)],
        contexts=(1, 2), ignore_tolerance_qb=8.0)
    assert set(training.labels) == {0.0, 1.0}
    assert training.left_sequences.ndim == 3
    assert training.left_sequences.shape[2] == 12
    assert training.left_sequences.shape == training.right_sequences.shape
    assert training.left_masks.shape == training.left_sequences.shape[:2]
    assert np.all(training.left_masks.sum(axis=1) >= 1)
    assert np.all(training.right_masks.sum(axis=1) >= 1)
    ignored = [row for row in training.audit if row["ignored"]]
    assert ignored
    assert all(row["sample_weight"] == 0 for row in ignored)
    assert all(row["status"] == "ignored_tolerance" for row in ignored)
    for work in {"w1", "w2"}:
        indices = np.asarray([value == work for value in training.works])
        assert np.isclose(training.sample_weights[indices].sum(), 0.5)
        assert np.isclose(training.natural_weights[indices].sum(), 0.5)
        for label in (0, 1):
            selected = indices & (training.labels == label)
            assert np.isclose(training.sample_weights[selected].sum(), 0.25)


def test_neural_distance_is_symmetric_batched_and_jointly_transposition_invariant():
    _, distance, _ = _trained()
    left = np.array([4, 0, 2, 0, 1, 0, 0, 2, 0, 1, 0, 0], dtype=float)
    right = np.array([0, 1, 0, 3, 0, 2, 0, 0, 1, 0, 2, 0], dtype=float)
    value = distance(left, right)
    assert value >= 0
    assert np.isclose(value, distance(right, left), atol=1e-7)
    assert np.isclose(distance(left, left), 0.0, atol=1e-7)
    batch = distance.batch_distance(np.vstack((left, right)), np.vstack((right, left)))
    assert np.allclose(batch, [value, value], atol=1e-7)
    for shift in (1, 5, 11):
        assert np.isclose(
            value, distance(np.roll(left, shift), np.roll(right, shift)), atol=1e-6)


def test_temporal_distance_preserves_order_masks_padding_and_transposition():
    _, distance, _ = _trained(seed=19)
    sequence = np.zeros((1, 4, 12), dtype=float)
    sequence[0, np.arange(4), [0, 4, 7, 2]] = 1.0
    mask = np.ones((1, 4), dtype=bool)
    reversed_sequence = sequence[:, ::-1].copy()
    reversed_mask = mask[:, ::-1].copy()
    original = distance.transform_sequence_batch(sequence, mask)
    reversed_embedding = distance.transform_sequence_batch(
        reversed_sequence, reversed_mask)
    assert not np.allclose(original, reversed_embedding, atol=1e-7)

    padded = np.pad(sequence, ((0, 0), (0, 3), (0, 0)))
    padded_mask = np.pad(mask, ((0, 0), (0, 3)), constant_values=False)
    assert np.allclose(
        original, distance.transform_sequence_batch(padded, padded_mask), atol=1e-6)
    comparison = np.zeros((1, 4, 12), dtype=float)
    comparison[0, np.arange(4), [7, 11, 2, 9]] = 1.0
    value = distance.batch_sequence_distance(
        sequence, comparison, mask, mask)[0]
    for shift in (1, 5, 11):
        shifted_left = np.roll(sequence, shift, axis=2)
        shifted_right = np.roll(comparison, shift, axis=2)
        shifted_value = distance.batch_sequence_distance(
            shifted_left, shifted_right, mask, mask)[0]
        assert np.isclose(value, shifted_value, atol=1e-6)


def test_empty_vectors_are_finite_and_serialization_round_trips(tmp_path):
    _, distance, _ = _trained()
    empty = np.zeros(12)
    sounding = np.eye(12)[3]
    assert np.isfinite(distance(empty, empty))
    assert np.isclose(distance(empty, empty), 0.0)
    assert np.isfinite(distance(empty, sounding))
    path = tmp_path / "distance.npz"
    distance.save(path)
    restored = NeuralPitchClassDistance.load(path)
    values = np.vstack((empty, sounding, np.roll(sounding, 4)))
    assert np.allclose(distance.transform_batch(values), restored.transform_batch(values))
    assert distance.as_dict()["probability_calibration"] is not None


def test_training_is_deterministic_and_change_point_has_larger_distance():
    first_training, first, first_history = _trained(seed=13)
    _, second, second_history = _trained(seed=13)
    for name in first.state:
        assert np.array_equal(first.state[name], second.state[name])
    assert first_history == second_history
    probabilities = first.predict_boundary_probability(
        first_training.left, first_training.right)
    assert np.all(np.isfinite(probabilities))
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    distances = first.batch_distance(first_training.left, first_training.right)
    assert distances[first_training.labels == 1].mean() > distances[first_training.labels == 0].mean()
    assert first.architecture == "harmonic_tcn_v3"
    assert first.dilations == (1, 3, 4, 5)
    assert first.temporal_dilations == (1, 2, 4)
    assert any(name.startswith("pitch_branches.") for name in first.state)
    assert any(name.startswith("temporal_blocks.") for name in first.state)
    assert all(np.isfinite(row[key]) for row in first_history
               for key in ("loss", "bce_loss", "metric_loss", "ranking_loss"))
    assert first_history[-1]["learning_rate"] < first_history[0]["learning_rate"]


def test_legacy_architecture_remains_loadable_for_old_ablation(tmp_path):
    training = build_interval_training_set(
        [_piece("w1", "p1"), _piece("w2", "p2", 2)],
        contexts=(1,), ignore_tolerance_qb=8.0)
    checkpoints, _ = train_distance_checkpoints(
        training, architecture="legacy_v1", channels=8,
        checkpoint_epochs=(1,), batch_size=8, seed=17)
    distance = checkpoints[0].distance
    path = tmp_path / "legacy_distance.npz"
    distance.save(path)
    restored = NeuralPitchClassDistance.load(path)
    assert restored.architecture == "legacy_v1"
    values = np.vstack((np.zeros(12), np.eye(12)[0], np.eye(12)[7]))
    assert np.allclose(distance.transform_batch(values), restored.transform_batch(values))


def test_aggregate_harmonic_architecture_remains_available_as_ablation():
    training = build_interval_training_set(
        [_piece("w1", "p1"), _piece("w2", "p2", 2)],
        contexts=(1, 2), ignore_tolerance_qb=8.0)
    checkpoints, _ = train_distance_checkpoints(
        training, architecture="harmonic_v2", channels=8,
        checkpoint_epochs=(1,), batch_size=8, seed=23)
    assert checkpoints[0].distance.architecture == "harmonic_v2"
    assert checkpoints[0].distance.as_dict()["temporal_order_preserved"] is False


def test_diagonal_checkpoint_is_normalised_and_finite():
    training = build_interval_training_set(
        [_piece("w1", "p1"), _piece("w2", "p2", 3)],
        contexts=(1,), ignore_tolerance_qb=8.0)
    checkpoints, _ = train_distance_checkpoints(
        training, kind="diagonal", checkpoint_epochs=(2,), batch_size=8, seed=4)
    distance = checkpoints[0].distance
    assert np.all(distance.weights >= 0)
    assert np.isclose(distance.weights.sum(), 1.0)
    assert np.isfinite(distance(np.eye(12)[0], np.eye(12)[7]))
