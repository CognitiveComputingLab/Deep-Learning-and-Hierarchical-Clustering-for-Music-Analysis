from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dp_clustering import assert_valid_ordered_binary_tree
from greedy_clustering import ClusterNode
from neural_clustering import (
    AdjacentMergeEnvironment, BoundaryDistanceModel, MergePolicy,
    MLPPitchClassEncoder, NeuralEmbeddingDistance, PitchClassEncoder,
    boundary_average_precision, build_pitch_class_encoder, rollout_policy,
    project_reference_boundaries, reference_boundary_indices,
    transpose_pitch_classes,
)
from evaluate_neural_corpus import make_outer_folds
from train_deep_clustering import grouped_macro_mean, summarize_held_out
from train_parametric_distance import split_works


def test_encoder_is_finite_and_l2_normalized():
    torch.manual_seed(1)
    encoder = PitchClassEncoder(dropout=0).eval()
    output = encoder(torch.rand(7, 12))
    assert output.shape == (7, 16)
    assert torch.isfinite(output).all()
    assert torch.allclose(torch.linalg.vector_norm(output, dim=1),
                          torch.ones(7), atol=1e-6)


def test_mlp_and_harmonic_encoder_factory_support_strict_ablation():
    values = torch.rand(5, 12)
    mlp = build_pitch_class_encoder("mlp", dropout=0).eval()
    harmonic = build_pitch_class_encoder("harmonic_cnn", dropout=0).eval()
    assert isinstance(mlp, MLPPitchClassEncoder)
    assert isinstance(harmonic, PitchClassEncoder)
    for encoder in (mlp, harmonic):
        output = encoder(values)
        assert output.shape == (5, 16)
        assert torch.isfinite(output).all()
        assert torch.allclose(torch.linalg.vector_norm(output, dim=1),
                              torch.ones(5), atol=1e-6)
    assert mlp.architecture_config()[
        "joint_transposition_distance_invariance"] == "augmentation_only"
    assert harmonic.architecture_config()[
        "joint_transposition_distance_invariance"] == "architectural"


def test_neural_distance_scalar_batch_and_transform_agree():
    torch.manual_seed(2)
    encoder = PitchClassEncoder(dropout=0).eval()
    distance = NeuralEmbeddingDistance(encoder)
    rng = np.random.default_rng(2)
    left, right = rng.random((5, 12)), rng.random((5, 12))
    batch = distance.batch_distance(left, right)
    scalar = np.asarray([distance(a, b) for a, b in zip(left, right)])
    transformed = np.linalg.norm(distance.transform_batch(left)
                                 - distance.transform_batch(right), axis=1)
    assert np.all(np.isfinite(batch)) and np.all(batch >= 0)
    assert np.allclose(batch, scalar)
    assert np.allclose(batch, transformed)
    assert np.allclose(batch, distance.batch_distance(right, left))
    assert np.allclose(distance.batch_distance(left, left), 0, atol=1e-7)


def test_joint_transposition_distance_is_architecturally_invariant():
    torch.manual_seed(12)
    model = BoundaryDistanceModel(PitchClassEncoder(dropout=0)).eval()
    left, right = torch.rand(9, 12), torch.rand(9, 12)
    original = model.distance(left, right)
    for shift in range(12):
        shifted = model.distance(
            transpose_pitch_classes(left, shift),
            transpose_pitch_classes(right, shift))
        assert torch.allclose(original, shifted, atol=2e-6, rtol=2e-6)
    assert not torch.allclose(model.encoder(left), model.encoder(
        transpose_pitch_classes(left, 1)))


def test_policy_candidate_features_are_joint_transposition_invariant():
    torch.manual_seed(13)
    matrix = np.random.default_rng(13).random((7, 12))
    bounds = np.arange(8, dtype=float)
    encoder = PitchClassEncoder(dropout=0).eval()
    original = AdjacentMergeEnvironment(matrix, bounds).candidate_features(
        encoder, torch.device("cpu"))
    shifted = AdjacentMergeEnvironment(
        np.roll(matrix, 5, axis=1), bounds).candidate_features(
            encoder, torch.device("cpu"))
    assert original.shape[1] == MergePolicy.input_dim == 44
    assert torch.allclose(original, shifted, atol=2e-6, rtol=2e-6)
    policy = MergePolicy().eval()
    original_rollout = rollout_policy(
        matrix, bounds, encoder, policy, deterministic=True)
    shifted_rollout = rollout_policy(
        np.roll(matrix, 5, axis=1), bounds, encoder, policy, deterministic=True)
    assert [row["action"] for row in original_rollout.trajectory] == [
        row["action"] for row in shifted_rollout.trajectory]


def test_pitch_transposition_is_circular_and_mass_preserving():
    values = torch.arange(24, dtype=torch.float32).reshape(2, 12)
    shifted = transpose_pitch_classes(values, torch.tensor([1, 5]))
    assert torch.equal(shifted[0], torch.roll(values[0], 1))
    assert torch.equal(shifted[1], torch.roll(values[1], 5))
    assert torch.equal(shifted.sum(dim=1), values.sum(dim=1))


def test_environment_only_merges_adjacent_clusters_and_finishes():
    matrix = np.eye(12)[:5]
    bounds = np.arange(6, dtype=float)
    environment = AdjacentMergeEnvironment(matrix, bounds)
    with pytest.raises(IndexError):
        environment.step(5)
    environment.step(1)
    assert [(item.first, item.last) for item in environment.clusters] == [
        (0, 1), (1, 3), (3, 4), (4, 5)]
    while not environment.done:
        environment.step(0)
    assert environment.steps == len(matrix) - 1
    assert_valid_ordered_binary_tree(environment.root, bounds)


def test_deterministic_rollout_is_reproducible_and_annotation_free():
    torch.manual_seed(4)
    matrix = np.eye(12)[:6]
    bounds = np.arange(7, dtype=float)
    encoder = PitchClassEncoder(dropout=0).eval()
    policy = MergePolicy().eval()
    first = rollout_policy(matrix, bounds, encoder, policy, deterministic=True)
    second = rollout_policy(matrix, bounds, encoder, policy, deterministic=True)
    assert [row["action"] for row in first.trajectory] == [
        row["action"] for row in second.trajectory]
    assert len(first.trajectory) == len(matrix) - 1
    assert_valid_ordered_binary_tree(first.root, bounds)
    parameters = set(inspect.signature(rollout_policy).parameters)
    assert "reference_indices" not in parameters
    assert "annotations" not in parameters


def _leaf(index):
    vector = np.eye(12)[index]
    return ClusterNode(index, index + 1, vector)


def _merge(left, right, order):
    return ClusterNode(left.start, right.end, left.feature + right.feature,
                       [left, right], order)


def test_correct_top_level_boundary_has_higher_ap_than_comb_tree():
    leaves = [_leaf(i) for i in range(4)]
    correct = _merge(_merge(leaves[0], leaves[1], 0),
                     _merge(leaves[2], leaves[3], 1), 2)
    wrong = _merge(_merge(_merge(_leaf(0), _leaf(1), 0), _leaf(2), 1),
                   _leaf(3), 2)
    bounds = np.arange(5, dtype=float)
    assert boundary_average_precision(correct, bounds, {2}) > boundary_average_precision(
        wrong, bounds, {2})


def test_work_split_has_no_overlap():
    pairs = [(f"work{index:02d}_01", Path("notes"), Path("harmonies"))
             for index in range(16)]
    _, names = split_works(pairs, 7, 10, 3, 3)
    sets = {name: set(values) for name, values in names.items()}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])


def test_outer_folds_test_every_work_once_without_within_fold_overlap():
    pairs = [(f"work{index:02d}_01", Path("notes"), Path("harmonies"))
             for index in range(16)]
    args = SimpleNamespace(
        split_seed=7, quick=False, outer_folds=4, validation_works=3)
    folds = make_outer_folds(pairs, args)
    tested = []
    for fold in folds:
        roles = {role: set(works) for role, works in fold["works"].items()}
        assert not (roles["train"] & roles["validation"])
        assert not (roles["train"] & roles["test"])
        assert not (roles["validation"] & roles["test"])
        tested.extend(roles["test"])
    assert len(tested) == len(set(tested)) == 16


def test_work_macro_gives_each_work_equal_weight():
    assert grouped_macro_mean([0.0, 0.0, 1.0], ["long", "long", "short"]) == pytest.approx(0.5)


def test_reference_projection_preserves_close_boundaries_when_possible():
    bounds = np.asarray([0.0, 8.0, 16.0, 24.0])
    references = [9.0, 10.0]
    projection = project_reference_boundaries(bounds, references)
    assert len(projection) == len(references)
    assert reference_boundary_indices(bounds, references) == {1, 2}
    assert len({row["boundary_index"] for row in projection}) == len(references)


def test_held_out_standard_deviation_is_across_seed_work_macros():
    frame = pd.DataFrame([
        {"seed": 1, "model": "m", "work": "a", "f1": 0.0},
        {"seed": 1, "model": "m", "work": "b", "f1": 1.0},
        {"seed": 2, "model": "m", "work": "a", "f1": 1.0},
        {"seed": 2, "model": "m", "work": "b", "f1": 1.0},
    ])
    for column in ("precision", "recall", "boundary_ap", "runtime_seconds"):
        frame[column] = frame.f1
    per_seed, summary = summarize_held_out(frame)
    assert per_seed.work_macro_f1.tolist() == pytest.approx([0.5, 1.0])
    assert summary.loc[0, "mean_f1"] == pytest.approx(0.75)
    assert summary.loc[0, "std_f1"] == pytest.approx(np.std([0.5, 1.0], ddof=1))


@pytest.mark.skipif(not (ROOT / "external" / "ABC" / "notes").is_dir(),
                    reason="ABC corpus is not installed")
def test_quick_cli_writes_reproducibility_outputs(tmp_path):
    output = tmp_path / "deep"
    command = [
        sys.executable, str(ROOT / "scripts" / "train_deep_clustering.py"),
        "--quick", "--device", "cpu", "--output-dir", str(output),
        "--metric-epochs", "1", "--imitation-epochs", "1", "--rl-epochs", "1",
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=120,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    required = {
        "config.json", "data_split.csv", "experiment_state.json",
        "metric_training_history.csv",
        "rl_training_history.csv", "held_out_per_piece.csv",
        "held_out_per_work.csv", "held_out_per_seed.csv", "held_out_summary.csv",
        "ablation_summary.csv", "access_audit.csv", "boundary_projection_audit.csv",
        "tree_diagnostics.csv", "action_trajectories.csv",
        "metric_held_out_per_work.csv", "learning_curves.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    state = json.loads((output / "experiment_state.json").read_text())
    assert state["phase"] == "complete"
    models = set(pd.read_csv(output / "held_out_summary.csv").model)
    assert {"key_profile_affinity_greedy", "siamese_affinity_greedy"} <= models
    access = pd.read_csv(output / "access_audit.csv")
    frozen = int(access.loc[
        access.event == "all_checkpoints_and_budgets_frozen", "sequence"].iloc[0])
    test_loaded = int(access.loc[
        (access.event == "annotations_loaded") & (access.split == "test"),
        "sequence"].iloc[0])
    assert test_loaded > frozen
    checkpoint = torch.load(output / "checkpoint_seed_20260827.pt",
                            map_location="cpu", weights_only=False)
    assert checkpoint["encoder_config"]["name"] == "circular_harmonic_cnn"
    config = json.loads((output / "config.json").read_text())
    assert config["encoder_architecture"]["embedding_dim"] == 16
    budgets = checkpoint["selected_budgets"]
    assert budgets["key_profile_affinity_greedy"] == budgets["key_profile_dp"]
    assert budgets["siamese_affinity_greedy"] == budgets["siamese_dp"]
    tensors = [value for state in checkpoint.values() if isinstance(state, dict)
               for value in state.values() if torch.is_tensor(value)]
    assert tensors and all(value.device.type == "cpu" for value in tensors)


@pytest.mark.skipif(not (ROOT / "external" / "ABC" / "notes").is_dir(),
                    reason="ABC corpus is not installed")
def test_neural_corpus_quick_cli_is_nested_and_complete(tmp_path):
    output = tmp_path / "corpus"
    command = [
        sys.executable, str(ROOT / "scripts" / "evaluate_neural_corpus.py"),
        "--quick", "--device", "cpu", "--output-dir", str(output),
        "--metric-epochs", "1",
    ]
    subprocess.run(command, cwd=ROOT, check=True, timeout=120,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    required = {
        "config.json", "experiment_state.json", "fold_assignments.csv",
        "metric_training_history.csv", "validation_budget_selection.csv",
        "held_out_per_piece.csv", "held_out_per_work.csv",
        "held_out_per_seed.csv", "held_out_summary.csv",
        "ablation_summary.csv", "tree_diagnostics.csv",
        "ted_auxiliary_per_piece.csv", "metric_held_out_per_work.csv",
        "paired_work_tests.csv", "access_audit.csv", "run_status.csv",
        "metric_learning_curves.png",
    }
    assert required <= {path.name for path in output.iterdir()}
    state = json.loads((output / "experiment_state.json").read_text())
    assert state["phase"] == "complete"
    assert state["n_outer_test_works"] == 6

    assignments = pd.read_csv(output / "fold_assignments.csv")
    outer_test = assignments[assignments.role == "test"]
    assert outer_test.groupby("work").fold.nunique().eq(1).all()
    assert outer_test.work.nunique() == 6
    for _, group in assignments.groupby("fold"):
        role_works = {role: set(rows.work) for role, rows in group.groupby("role")}
        assert not (role_works["train"] & role_works["validation"])
        assert not (role_works["train"] & role_works["test"])
        assert not (role_works["validation"] & role_works["test"])

    access = pd.read_csv(output / "access_audit.csv")
    for _, group in access.groupby("fold"):
        frozen = int(group.loc[
            group.event == "fold_model_selection_frozen", "sequence"].iloc[0])
        test_loaded = int(group.loc[
            group.event == "outer_test_annotations_loaded", "sequence"].iloc[0])
        assert test_loaded > frozen
    models = set(pd.read_csv(output / "held_out_summary.csv").model)
    assert {
        "key_profile_affinity_greedy", "key_profile_dp",
        "mlp_affinity_greedy", "mlp_dp",
        "harmonic_cnn_affinity_greedy", "harmonic_cnn_dp",
    } == models
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False)
                   for path in (output / "checkpoints").glob("*.pt")]
    checkpoint_names = {row["encoder_config"]["name"] for row in checkpoints}
    assert checkpoint_names == {"mlp", "circular_harmonic_cnn"}
    budgets_by_fold_seed = {}
    for row in checkpoints:
        budgets_by_fold_seed.setdefault((row["fold"], row["seed"]), set()).add(
            row["validation_selected_budget"])
    assert all(len(values) == 1 for values in budgets_by_fold_seed.values())
