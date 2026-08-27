"""Neural metric learning and reinforcement learning for ordered music trees.

The inference APIs in this module deliberately do not accept annotations.
DCML local-key boundaries are used by the training script only to construct
metric-learning labels and terminal policy-gradient rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from greedy_clustering import ClusterNode
from greedy_evaluation import boundary_prominence_scores


def l1_normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    totals = values.sum(dim=-1, keepdim=True)
    return torch.where(totals > 1e-12, values / totals.clamp_min(1e-12),
                       torch.zeros_like(values))


def transpose_pitch_classes(values: torch.Tensor, shifts: torch.Tensor | int) -> torch.Tensor:
    """Circularly transpose pitch-class rows without changing their mass."""
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("values must have shape (batch, 12)")
    if isinstance(shifts, int):
        return torch.roll(values, shifts=int(shifts), dims=1)
    shifts = torch.as_tensor(shifts, device=values.device, dtype=torch.long)
    if shifts.shape != (len(values),):
        raise ValueError("shifts must contain one value per row")
    columns = (torch.arange(12, device=values.device)[None, :] - shifts[:, None]) % 12
    return torch.gather(values, 1, columns)


def _circular_conv(in_channels: int, out_channels: int, *, dilation: int) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels, out_channels, kernel_size=3, dilation=dilation,
        padding=dilation, padding_mode="circular",
    )


class _CircularResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.first = _circular_conv(channels, channels, dilation=1)
        self.first_norm = nn.GroupNorm(4, channels)
        self.second = _circular_conv(channels, channels, dilation=1)
        self.second_norm = nn.GroupNorm(4, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.relu(self.first_norm(self.first(values)))
        values = self.second_norm(self.second(values))
        return F.relu(values + residual)


class MLPPitchClassEncoder(nn.Module):
    """Original non-equivariant encoder retained only for strict ablation."""

    architecture_name = "mlp"
    output_dim = 16

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, self.output_dim),
        )

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": self.architecture_name,
            "input_pitch_classes": 12,
            "hidden_dimensions": [32, 16],
            "embedding_dim": self.output_dim,
            "joint_transposition_distance_invariance": "augmentation_only",
        }

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = l1_normalize_tensor(values.float())
        return F.normalize(self.network(values), p=2, dim=-1, eps=1e-12)


class PitchClassEncoder(nn.Module):
    """Circular harmonic CNN with exactly joint-transposition-invariant distance.

    Circular convolutions are equivariant to pitch-class rotation.  Selected
    Fourier coefficients are stored as real/imaginary pairs, so a common
    transposition rotates each pair orthogonally.  Euclidean distance between
    two embeddings is therefore unchanged by a common transposition while the
    embedding itself still retains tonic phase.
    """

    architecture_name = "circular_harmonic_cnn"
    dilations = (1, 3, 4, 5)
    harmonics = (1, 3, 4, 5)
    branch_channels = 4
    harmonic_channels = 2
    complex_pair_count = harmonic_channels * len(harmonics)
    output_dim = complex_pair_count * 2
    pair_feature_dim = complex_pair_count * 5

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        channels = self.branch_channels * len(self.dilations)
        self.branches = nn.ModuleList([
            _circular_conv(1, self.branch_channels, dilation=dilation)
            for dilation in self.dilations
        ])
        self.stem_norm = nn.GroupNorm(4, channels)
        self.residual = _CircularResidualBlock(channels)
        self.dropout = nn.Dropout1d(dropout)
        self.harmonic_projection = nn.Conv1d(
            channels, self.harmonic_channels, kernel_size=1, bias=False)
        self.register_buffer(
            "harmonic_indices", torch.tensor(self.harmonics, dtype=torch.long),
            persistent=False)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": self.architecture_name,
            "input_pitch_classes": 12,
            "dilations": list(self.dilations),
            "branch_channels": self.branch_channels,
            "residual_channels": self.branch_channels * len(self.dilations),
            "harmonic_channels": self.harmonic_channels,
            "selected_harmonics": list(self.harmonics),
            "embedding_dim": self.output_dim,
            "joint_transposition_distance_invariance": "architectural",
        }

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = l1_normalize_tensor(values.float()).unsqueeze(1)
        features = torch.cat([branch(values) for branch in self.branches], dim=1)
        features = F.relu(self.stem_norm(features))
        features = self.dropout(self.residual(features))
        harmonic_signal = self.harmonic_projection(features)
        spectrum = torch.fft.rfft(harmonic_signal, n=12, dim=-1, norm="ortho")
        selected = spectrum.index_select(-1, self.harmonic_indices)
        embedding = torch.view_as_real(selected).reshape(len(values), self.output_dim)
        return F.normalize(embedding, p=2, dim=-1, eps=1e-12)


def build_pitch_class_encoder(name: str, dropout: float = 0.1) -> nn.Module:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"harmonic", "harmonic_cnn", "circular_harmonic_cnn"}:
        return PitchClassEncoder(dropout=dropout)
    if normalized == "mlp":
        return MLPPitchClassEncoder(dropout=dropout)
    raise ValueError(f"unknown pitch-class encoder: {name}")


def invariant_harmonic_pair_features(left: torch.Tensor,
                                     right: torch.Tensor) -> torch.Tensor:
    """Construct policy features invariant to a common harmonic phase rotation."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != PitchClassEncoder.output_dim:
        raise ValueError("left/right must be matching harmonic embedding batches")
    left_pairs = left.reshape(len(left), PitchClassEncoder.complex_pair_count, 2)
    right_pairs = right.reshape(len(right), PitchClassEncoder.complex_pair_count, 2)
    left_amplitude = torch.linalg.vector_norm(left_pairs, dim=-1)
    right_amplitude = torch.linalg.vector_norm(right_pairs, dim=-1)
    difference_amplitude = torch.linalg.vector_norm(left_pairs - right_pairs, dim=-1)
    # z_left * conjugate(z_right): relative phase is unchanged when both
    # coefficients receive the same transposition-dependent phase rotation.
    relative_real = (left_pairs * right_pairs).sum(dim=-1)
    relative_imag = (left_pairs[..., 1] * right_pairs[..., 0]
                     - left_pairs[..., 0] * right_pairs[..., 1])
    relative = torch.stack((relative_real, relative_imag), dim=-1).flatten(1)
    return torch.cat((left_amplitude, right_amplitude, difference_amplitude,
                      relative), dim=1)


class BoundaryDistanceModel(nn.Module):
    """Siamese distance plus a monotonic boundary-classification head."""

    def __init__(self, encoder: nn.Module | None = None):
        super().__init__()
        self.encoder = encoder or PitchClassEncoder()
        self.raw_scale = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(self.encoder(left) - self.encoder(right), dim=-1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.raw_scale) * self.distance(left, right) + self.bias


class NeuralEmbeddingDistance:
    """NumPy-compatible distance adapter for existing Greedy and DP code."""

    name = "siamese_embedding"

    def __init__(self, encoder: nn.Module, device: str | torch.device = "cpu"):
        self.encoder = encoder
        self.device = torch.device(device)
        self.encoder.to(self.device)

    def transform_batch(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError("values must have shape (batch, 12)")
        training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            result = self.encoder(torch.from_numpy(values).to(self.device)).cpu().numpy()
        self.encoder.train(training)
        return result.astype(float, copy=False)

    def batch_distance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_repr = self.transform_batch(left)
        right_repr = self.transform_batch(right)
        if left_repr.shape != right_repr.shape:
            raise ValueError("left and right batches must have matching shapes")
        return np.linalg.norm(left_repr - right_repr, axis=1)

    def __call__(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(self.batch_distance(np.asarray(left)[None, :],
                                         np.asarray(right)[None, :])[0])


class MergePolicy(nn.Module):
    input_dim = PitchClassEncoder.pair_feature_dim + 4

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 2 or candidates.shape[1] != self.input_dim:
            raise ValueError(f"candidates must have shape (n, {self.input_dim})")
        return self.network(candidates).squeeze(-1)

    @staticmethod
    def embedding_distance(candidates: torch.Tensor) -> torch.Tensor:
        start = 2 * PitchClassEncoder.complex_pair_count
        end = 3 * PitchClassEncoder.complex_pair_count
        return torch.linalg.vector_norm(candidates[:, start:end], dim=1)


@dataclass
class _RLCluster:
    first: int
    last: int
    vector: np.ndarray
    node: ClusterNode

    @property
    def leaf_count(self) -> int:
        return self.last - self.first


class AdjacentMergeEnvironment:
    """Annotation-free ordered adjacent-merge environment used at inference."""

    def __init__(self, matrix: np.ndarray, bounds: Sequence[float]):
        matrix = np.asarray(matrix, dtype=float)
        bounds = np.asarray(bounds, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 12 or len(matrix) < 1:
            raise ValueError("matrix must have shape (n>=1, 12)")
        if bounds.ndim == 2 and bounds.shape == (len(matrix), 2):
            if len(bounds) > 1 and not np.allclose(bounds[:-1, 1], bounds[1:, 0]):
                raise ValueError("bounds must be contiguous")
            edges = np.r_[bounds[0, 0], bounds[:, 1]]
        elif bounds.ndim == 1 and len(bounds) == len(matrix) + 1:
            edges = bounds
        else:
            raise ValueError("bounds must have shape (n,2) or (n+1,)")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("matrix must contain finite non-negative values")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("bounds must be strictly increasing")
        self.matrix = matrix
        self.edges = np.asarray(edges, dtype=float)
        self.n_leaves = len(matrix)
        self.steps = 0
        self.clusters = [
            _RLCluster(i, i + 1, row.copy(),
                       ClusterNode(edges[i], edges[i + 1], row.copy()))
            for i, row in enumerate(matrix)
        ]

    @property
    def done(self) -> bool:
        return len(self.clusters) == 1

    @property
    def root(self) -> ClusterNode:
        if not self.done:
            raise RuntimeError("episode is not complete")
        return self.clusters[0].node

    def candidate_features(self, encoder: PitchClassEncoder,
                           device: torch.device) -> torch.Tensor:
        if self.done:
            raise RuntimeError("complete environment has no actions")
        values = torch.as_tensor(np.stack([item.vector for item in self.clusters]),
                                 dtype=torch.float32, device=device)
        embeddings = encoder(values)
        left, right = embeddings[:-1], embeddings[1:]
        pair_features = invariant_harmonic_pair_features(left, right)
        counts_left = torch.tensor([item.leaf_count for item in self.clusters[:-1]],
                                   dtype=torch.float32, device=device)
        counts_right = torch.tensor([item.leaf_count for item in self.clusters[1:]],
                                    dtype=torch.float32, device=device)
        parent = counts_left + counts_right
        scalars = torch.stack((
            torch.log1p(counts_left) / np.log1p(self.n_leaves),
            torch.log1p(counts_right) / np.log1p(self.n_leaves),
            torch.abs(counts_left - counts_right) / parent,
            torch.tensor([item.last for item in self.clusters[:-1]],
                         dtype=torch.float32, device=device) / self.n_leaves,
        ), dim=1)
        return torch.cat((pair_features, scalars), dim=1)

    def safe_actions(self, reference_indices: Iterable[int]) -> list[int]:
        boundaries = set(int(value) for value in reference_indices)
        safe = []
        for action, (left, right) in enumerate(zip(self.clusters[:-1], self.clusters[1:])):
            if not any(left.first < split < right.last for split in boundaries):
                safe.append(action)
        return safe

    def step(self, action: int) -> None:
        if self.done:
            raise RuntimeError("cannot step a completed episode")
        action = int(action)
        if not 0 <= action < len(self.clusters) - 1:
            raise IndexError("action must identify an adjacent cluster pair")
        left, right = self.clusters[action:action + 2]
        if left.last != right.first:
            raise RuntimeError("environment lost leaf contiguity")
        vector = left.vector + right.vector
        node = ClusterNode(left.node.start, right.node.end, vector.copy(),
                           [left.node, right.node], self.steps)
        self.clusters[action:action + 2] = [
            _RLCluster(left.first, right.last, vector, node)
        ]
        self.steps += 1


@dataclass
class PolicyRollout:
    root: ClusterNode
    log_probability: torch.Tensor
    mean_entropy: torch.Tensor
    trajectory: list[dict[str, float | int]]


def rollout_policy(matrix: np.ndarray, bounds: Sequence[float],
                   encoder: PitchClassEncoder, policy: MergePolicy, *,
                   deterministic: bool, device: str | torch.device = "cpu",
                   generator: torch.Generator | None = None) -> PolicyRollout:
    """Roll out a policy without any annotation or oracle input."""
    device = torch.device(device)
    environment = AdjacentMergeEnvironment(matrix, bounds)
    log_probabilities: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    trajectory: list[dict[str, float | int]] = []
    while not environment.done:
        features = environment.candidate_features(encoder, device)
        logits = policy(features)
        probabilities = torch.softmax(logits, dim=0)
        if deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(torch.multinomial(probabilities, 1, generator=generator).item())
        log_probabilities.append(torch.log(probabilities[action].clamp_min(1e-12)))
        entropies.append(-(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())
        left, right = environment.clusters[action:action + 2]
        trajectory.append({
            "step": environment.steps, "action": action,
            "left_first": left.first, "left_last": left.last,
            "right_first": right.first, "right_last": right.last,
            "action_probability": float(probabilities[action].detach().cpu()),
        })
        environment.step(action)
    zero = torch.zeros((), device=device)
    return PolicyRollout(
        root=environment.root,
        log_probability=torch.stack(log_probabilities).sum() if log_probabilities else zero,
        mean_entropy=torch.stack(entropies).mean() if entropies else zero,
        trajectory=trajectory,
    )


def _ordered_min_cost_assignment(shorter: np.ndarray,
                                 longer: np.ndarray) -> list[tuple[int, int]]:
    """Match every item in ``shorter`` to a unique ordered item in ``longer``."""
    m, n = len(shorter), len(longer)
    if m > n:
        raise ValueError("shorter must not contain more items than longer")
    costs = np.full((m + 1, n + 1), np.inf, dtype=float)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    costs[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(i, n + 1):
            skip = costs[i, j - 1]
            match = costs[i - 1, j - 1] + abs(shorter[i - 1] - longer[j - 1])
            # Prefer skipping a tied later candidate, which makes the projection
            # deterministic and leftmost without changing its total error.
            if match < skip:
                costs[i, j] = match
                take[i, j] = True
            else:
                costs[i, j] = skip
    assignments: list[tuple[int, int]] = []
    i, j = m, n
    while i:
        if take[i, j]:
            assignments.append((i - 1, j - 1))
            i -= 1
        j -= 1
    assignments.reverse()
    return assignments


def project_reference_boundaries(
        bounds: Sequence[float], boundaries_qb: Sequence[float]) -> list[dict[str, float | int]]:
    """Project reference times to distinct ordered internal bin boundaries.

    Independent nearest-neighbour rounding can collapse multiple annotations
    onto one bin.  This order-preserving minimum-error projection keeps every
    reference distinct whenever the tree has enough internal boundaries.
    """
    bounds = np.asarray(bounds, dtype=float)
    edges = np.r_[bounds[0, 0], bounds[:, 1]] if bounds.ndim == 2 else bounds
    internal = np.asarray(edges[1:-1], dtype=float)
    references = np.sort(np.asarray(list(boundaries_qb), dtype=float))
    if np.any(~np.isfinite(references)):
        raise ValueError("reference boundaries must be finite")
    if not len(internal) or not len(references):
        return []
    if len(references) <= len(internal):
        pairs = _ordered_min_cost_assignment(references, internal)
        mapped = [(reference_index, edge_index)
                  for reference_index, edge_index in pairs]
    else:
        # A tree with n leaves exposes only n-1 candidate boundaries.  In this
        # exceptional case retain the minimum-error ordered subset and make the
        # unavoidable loss explicit in the projection audit.
        pairs = _ordered_min_cost_assignment(internal, references)
        mapped = [(reference_index, edge_index)
                  for edge_index, reference_index in pairs]
    return [{
        "reference_qb": float(references[reference_index]),
        "boundary_index": int(edge_index + 1),
        "projected_qb": float(internal[edge_index]),
        "absolute_error_qb": float(abs(references[reference_index] - internal[edge_index])),
    } for reference_index, edge_index in mapped]


def reference_boundary_indices(bounds: Sequence[float],
                               boundaries_qb: Sequence[float]) -> set[int]:
    return {int(row["boundary_index"])
            for row in project_reference_boundaries(bounds, boundaries_qb)}


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be matching vectors")
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def boundary_average_precision(root: Any, bounds: Sequence[float],
                               reference_indices: Iterable[int]) -> float:
    """Score a completed tree; kept separate from annotation-free rollout."""
    bounds = np.asarray(bounds, dtype=float)
    edges = np.r_[bounds[0, 0], bounds[:, 1]] if bounds.ndim == 2 else bounds
    references = set(int(value) for value in reference_indices)
    labels = np.asarray([int(index in references) for index in range(1, len(edges) - 1)])
    scores = boundary_prominence_scores(root, edges)
    return average_precision(labels, scores)


def state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
