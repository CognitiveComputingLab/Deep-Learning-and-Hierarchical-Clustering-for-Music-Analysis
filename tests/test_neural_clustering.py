from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

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
    AdjacentMergeEnvironment, MergePolicy, NeuralEmbeddingDistance,
    PitchClassEncoder, boundary_average_precision, rollout_policy,
    project_reference_boundaries, reference_boundary_indices,
    transpose_pitch_classes,
)
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
    budgets = checkpoint["selected_budgets"]
    assert budgets["key_profile_affinity_greedy"] == budgets["key_profile_dp"]
    assert budgets["siamese_affinity_greedy"] == budgets["siamese_dp"]
    tensors = [value for state in checkpoint.values() if isinstance(state, dict)
               for value in state.values() if torch.is_tensor(value)]
    assert tensors and all(value.device.type == "cpu" for value in tensors)
