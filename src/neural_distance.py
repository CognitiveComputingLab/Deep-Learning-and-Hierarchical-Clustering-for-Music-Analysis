"""Small, transposition-equivariant distances for the advanced DP stage.

The neural model is deliberately compact: it learns a symmetric distance
between pitch-class intervals, while exact tree search remains the
responsibility of :mod:`ordered_affinity`.  ``harmonic_tcn_v3`` preserves the
ordered sequence of chroma bins; the older aggregate encoders remain available
as explicit ablations.  DCML annotations are used only by the training-example
builder; frozen distance objects never read labels at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from boundary_contrast import (
    ProbabilityCalibrator,
    fit_probability_calibrator,
    tolerance_aware_boundary_labels,
)


NEURAL_ARCHITECTURES = ("legacy_v1", "harmonic_v2", "harmonic_tcn_v3")
DEFAULT_HARMONIC_DILATIONS = (1, 3, 4, 5)
DEFAULT_TEMPORAL_DILATIONS = (1, 2, 4)
TEMPORAL_ARCHITECTURES = frozenset({"harmonic_tcn_v3"})


def _require_torch():
    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - exercised without torch only
        raise RuntimeError("PyTorch is required for neural distance training") from error
    return torch, nn, functional


def _l1_batch(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("pitch-class values must have shape (n, 12)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("pitch-class values must be finite and non-negative")
    totals = values.sum(axis=1, keepdims=True)
    return np.divide(values, totals, out=np.zeros_like(values), where=totals > eps)


def _l1_sequence_batch(
    values: np.ndarray,
    masks: np.ndarray | None = None,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise pitch classes per time bin without confusing padding and rests."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 2 and values.shape[1:] == (12,):
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[2] != 12:
        raise ValueError("pitch-class sequences must have shape (n, time, 12)")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("pitch-class sequences must be finite and non-negative")
    if masks is None:
        masks = np.ones(values.shape[:2], dtype=bool)
    else:
        masks = np.asarray(masks, dtype=bool)
        if masks.shape != values.shape[:2]:
            raise ValueError("sequence masks must have shape (n, time)")
    totals = values.sum(axis=2, keepdims=True)
    normalised = np.divide(
        values, totals, out=np.zeros_like(values), where=totals > eps)
    normalised *= masks[:, :, None]
    return normalised, masks


def _pad_sequence_list(
    sequences: Sequence[np.ndarray],
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    checked = [np.asarray(value, dtype=float) for value in sequences]
    if any(value.ndim != 2 or value.shape[1:] != (12,) for value in checked):
        raise ValueError("each sequence must have shape (time, 12)")
    length = max(len(value) for value in checked) if max_length is None else int(max_length)
    if length < 1 or any(not len(value) or len(value) > length for value in checked):
        raise ValueError("max_length must contain every non-empty sequence")
    values = np.zeros((len(checked), length, 12), dtype=float)
    masks = np.zeros((len(checked), length), dtype=bool)
    for index, value in enumerate(checked):
        values[index, :len(value)] = value
        masks[index, :len(value)] = True
    return values, masks


def _encoder_module(
    architecture: str,
    channels: int,
    dilations: Sequence[int] = DEFAULT_HARMONIC_DILATIONS,
    temporal_dilations: Sequence[int] = DEFAULT_TEMPORAL_DILATIONS,
    temporal_dropout: float = 0.15,
):
    """Build a compact jointly-transposition-equivariant interval encoder."""
    torch, nn, functional = _require_torch()
    dilations = tuple(int(value) for value in dilations)
    temporal_dilations = tuple(int(value) for value in temporal_dilations)
    if architecture not in NEURAL_ARCHITECTURES:
        raise ValueError(f"unknown neural architecture: {architecture}")
    if (channels < 1 or not dilations or any(value < 1 for value in dilations)
            or not temporal_dilations or any(value < 1 for value in temporal_dilations)
            or not 0 <= temporal_dropout < 1):
        raise ValueError("channels, dilations and dropout are invalid")

    class LegacyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(1, channels, 3, padding=1,
                                   padding_mode="circular")
            self.conv2 = nn.Conv1d(channels, channels, 3, padding=1,
                                   padding_mode="circular")

        def forward(self, values, mask=None):
            if values.ndim == 3:
                if mask is not None:
                    values = values * mask[:, :, None].to(values.dtype)
                values = values.sum(dim=1)
            total = values.sum(dim=1, keepdim=True)
            active = (total > 1e-12).to(values.dtype)
            values = torch.where(total > 1e-12, values / total.clamp_min(1e-12),
                                 torch.zeros_like(values))
            hidden = functional.gelu(self.conv1(values[:, None, :]))
            hidden = functional.gelu(self.conv2(hidden)).flatten(1)
            return 0.5 * functional.normalize(hidden, dim=1, eps=1e-12) * active

    class HarmonicEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = nn.ModuleList([
                nn.Conv1d(1, channels, 3, padding=value, dilation=value,
                          padding_mode="circular") for value in dilations])
            self.mix = nn.Conv1d(channels * len(dilations), channels, 1)
            self.skip = nn.Conv1d(1, channels, 1, bias=False)
            self.norm1 = nn.GroupNorm(1, channels)
            self.context = nn.Conv1d(channels, channels, 3, padding=1,
                                     padding_mode="circular")
            self.norm2 = nn.GroupNorm(1, channels)

        def forward(self, values, mask=None):
            if values.ndim == 3:
                if mask is not None:
                    values = values * mask[:, :, None].to(values.dtype)
                values = values.sum(dim=1)
            total = values.sum(dim=1, keepdim=True)
            active = (total > 1e-12).to(values.dtype)
            values = torch.where(total > 1e-12, values / total.clamp_min(1e-12),
                                 torch.zeros_like(values))
            source = values[:, None, :]
            branches = torch.cat(
                [functional.gelu(branch(source)) for branch in self.branches], dim=1)
            hidden = functional.gelu(
                self.norm1(self.mix(branches) + self.skip(source)))
            hidden = functional.gelu(self.norm2(self.context(hidden) + hidden))
            hidden = functional.normalize(hidden.flatten(1), dim=1, eps=1e-12)
            return 0.5 * hidden * active

    class ChannelLayerNorm(nn.Module):
        """Layer-normalise channels independently at each time/pitch cell."""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(channels)

        def forward(self, values):
            values = values.permute(0, 2, 3, 1)
            values = self.norm(values)
            return values.permute(0, 3, 1, 2)

    class TemporalBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.conv1 = nn.Conv2d(
                channels, channels, (3, 1), padding=(dilation, 0),
                dilation=(dilation, 1))
            self.conv2 = nn.Conv2d(
                channels, channels, (3, 1), padding=(dilation, 0),
                dilation=(dilation, 1))
            self.norm1 = ChannelLayerNorm()
            self.norm2 = ChannelLayerNorm()
            self.dropout = nn.Dropout(temporal_dropout)

        def forward(self, values, mask):
            hidden = functional.gelu(self.norm1(self.conv1(values)))
            hidden = self.dropout(hidden)
            hidden = hidden * mask[:, None, :, None].to(hidden.dtype)
            hidden = self.conv2(hidden)
            hidden = functional.gelu(self.norm2(hidden + values))
            return hidden * mask[:, None, :, None].to(hidden.dtype)

    class HarmonicTemporalEncoder(nn.Module):
        """Circular pitch CNN followed by a residual, dilated temporal TCN."""

        def __init__(self):
            super().__init__()
            self.pitch_branches = nn.ModuleList([
                nn.Conv2d(1, channels, (1, 3), dilation=(1, value))
                for value in dilations
            ])
            self.mix = nn.Conv2d(channels * len(dilations), channels, 1)
            self.skip = nn.Conv2d(1, channels, 1, bias=False)
            self.pitch_norm = ChannelLayerNorm()
            self.temporal_blocks = nn.ModuleList(
                [TemporalBlock(value) for value in temporal_dilations])

        @staticmethod
        def _pitch_convolution(layer, values, padding):
            padded = functional.pad(
                values, (padding, padding, 0, 0), mode="circular")
            return layer(padded)

        def forward(self, values, mask=None):
            if values.ndim == 2:
                values = values[:, None, :]
            if values.ndim != 3 or values.shape[-1] != 12:
                raise ValueError("temporal encoder expects (batch, time, 12)")
            if mask is None:
                mask = torch.ones(
                    values.shape[:2], dtype=torch.bool, device=values.device)
            total = values.sum(dim=2, keepdim=True)
            sounding = (total > 1e-12) & mask[:, :, None]
            values = torch.where(
                total > 1e-12, values / total.clamp_min(1e-12),
                torch.zeros_like(values))
            values = values * mask[:, :, None].to(values.dtype)
            source = values[:, None, :, :]
            branches = torch.cat([
                functional.gelu(self._pitch_convolution(branch, source, dilation))
                for branch, dilation in zip(self.pitch_branches, dilations)
            ], dim=1)
            hidden = functional.gelu(
                self.pitch_norm(self.mix(branches) + self.skip(source)))
            hidden = hidden * mask[:, None, :, None].to(hidden.dtype)
            for block in self.temporal_blocks:
                hidden = block(hidden, mask)

            weights = mask[:, None, :, None].to(hidden.dtype)
            mean = (hidden * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
            masked = hidden.masked_fill(~mask[:, None, :, None], float("-inf"))
            maximum = masked.max(dim=2).values
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
            pooled = torch.cat((mean, maximum), dim=1).flatten(1)
            active = sounding.any(dim=1).to(values.dtype)
            pooled = functional.normalize(pooled, dim=1, eps=1e-12)
            return 0.5 * pooled * active

    if architecture == "legacy_v1":
        return LegacyEncoder()
    if architecture == "harmonic_v2":
        return HarmonicEncoder()
    return HarmonicTemporalEncoder()


@dataclass(frozen=True)
class NeuralTrainingSet:
    """Eligible local interval pairs and a complete candidate audit."""

    left: np.ndarray
    right: np.ndarray
    left_sequences: np.ndarray
    right_sequences: np.ndarray
    left_masks: np.ndarray
    right_masks: np.ndarray
    labels: np.ndarray
    sample_weights: np.ndarray
    natural_weights: np.ndarray
    works: tuple[str, ...]
    pieces: tuple[str, ...]
    split_indices: np.ndarray
    contexts: np.ndarray
    audit: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        left = np.asarray(self.left, dtype=float)
        right = np.asarray(self.right, dtype=float)
        left_sequences = np.asarray(self.left_sequences, dtype=float)
        right_sequences = np.asarray(self.right_sequences, dtype=float)
        left_masks = np.asarray(self.left_masks, dtype=bool)
        right_masks = np.asarray(self.right_masks, dtype=bool)
        labels = np.asarray(self.labels, dtype=float)
        sample = np.asarray(self.sample_weights, dtype=float)
        natural = np.asarray(self.natural_weights, dtype=float)
        if left.ndim != 2 or left.shape[1:] != (12,) or right.shape != left.shape:
            raise ValueError("training interval pairs must have shape (n, 12)")
        if (left_sequences.ndim != 3 or left_sequences.shape[0] != len(left)
                or left_sequences.shape[2] != 12
                or right_sequences.shape != left_sequences.shape
                or left_masks.shape != left_sequences.shape[:2]
                or right_masks.shape != left_masks.shape):
            raise ValueError("sequence pairs and masks must align as (n, time, 12)")
        if np.any(left_masks.sum(axis=1) < 1) or np.any(right_masks.sum(axis=1) < 1):
            raise ValueError("every interval sequence must contain at least one valid bin")
        if labels.shape != (len(left),) or sample.shape != labels.shape or natural.shape != labels.shape:
            raise ValueError("labels and weights must align with interval pairs")
        if not len(left) or set(np.unique(labels)) != {0.0, 1.0}:
            raise ValueError("training requires non-empty examples of both classes")
        if (not np.all(np.isfinite(left)) or not np.all(np.isfinite(right))
                or not np.all(np.isfinite(left_sequences))
                or not np.all(np.isfinite(right_sequences))
                or not np.all(np.isfinite(sample)) or not np.all(np.isfinite(natural))):
            raise ValueError("training arrays contain non-finite values")
        if (np.any(left < 0) or np.any(right < 0)
                or np.any(left_sequences < 0) or np.any(right_sequences < 0)
                or np.any(sample < 0) or np.any(natural < 0)):
            raise ValueError("training features and weights must be non-negative")
        if not np.isclose(sample.sum(), 1.0) or not np.isclose(natural.sum(), 1.0):
            raise ValueError("training weights must each sum to one")


def build_interval_training_set(
    pieces: Sequence[Mapping[str, Any]],
    *,
    contexts: Sequence[int] = (1, 2, 4),
    ignore_tolerance_qb: float = 8.0,
) -> NeuralTrainingSet:
    """Build all eligible local left/right pairs without work leakage.

    The nearest candidate to each reference boundary is positive.  Other
    candidates inside the evaluation tolerance are audited and ignored rather
    than incorrectly treated as hard negatives.  Every work has equal total
    mass; within a work, present classes have equal mass.  No negative
    subsampling is performed.
    """

    contexts = tuple(sorted(set(int(value) for value in contexts)))
    if not contexts or any(value < 1 for value in contexts):
        raise ValueError("contexts must contain positive integers")
    if not pieces:
        raise ValueError("at least one training piece is required")

    rows: list[dict[str, Any]] = []
    for item in sorted(pieces, key=lambda value: (str(value["work"]), str(value["piece"]))):
        matrix = np.asarray(item["matrix"], dtype=float)
        if matrix.ndim != 2 or matrix.shape[1:] != (12,) or len(matrix) < 2:
            raise ValueError("each training piece needs at least two 12-D bins")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("training matrices must be finite and non-negative")
        labels, ignored = tolerance_aware_boundary_labels(
            item["bounds"], item["reference"], len(matrix),
            ignore_tolerance_qb=ignore_tolerance_qb)
        prefix = np.vstack((np.zeros(12), np.cumsum(matrix, axis=0)))
        for split in range(1, len(matrix)):
            for context in contexts:
                left_start = max(0, split - context)
                right_end = min(len(matrix), split + context)
                rows.append({
                    "work": str(item["work"]), "piece": str(item["piece"]),
                    "split_index": split, "context_bins": context,
                    "left": prefix[split] - prefix[left_start],
                    "right": prefix[right_end] - prefix[split],
                    "left_sequence": matrix[left_start:split].copy(),
                    "right_sequence": matrix[split:right_end].copy(),
                    "label": int(labels[split - 1]),
                    "ignored": bool(ignored[split - 1]),
                })

    eligible = [row for row in rows if not row["ignored"]]
    if not eligible or set(row["label"] for row in eligible) != {0, 1}:
        raise ValueError("eligible training candidates must contain both classes")
    works = sorted({row["work"] for row in eligible})
    sample = np.zeros(len(eligible), dtype=float)
    natural = np.zeros(len(eligible), dtype=float)
    index_by_key = {
        (row["work"], row["piece"], row["split_index"], row["context_bins"]): index
        for index, row in enumerate(eligible)
    }
    for work in works:
        work_indices = [index for index, row in enumerate(eligible) if row["work"] == work]
        natural[work_indices] = 1.0 / len(works) / len(work_indices)
        present = sorted({eligible[index]["label"] for index in work_indices})
        for label in present:
            indices = [index for index in work_indices if eligible[index]["label"] == label]
            sample[indices] = 1.0 / len(works) / len(present) / len(indices)
    sample /= sample.sum()
    natural /= natural.sum()

    counts: dict[str, dict[str, int]] = {}
    for work in sorted({row["work"] for row in rows}):
        work_rows = [row for row in rows if row["work"] == work]
        counts[work] = {
            "work_positive_rows": sum(row["label"] == 1 and not row["ignored"] for row in work_rows),
            "work_negative_rows": sum(row["label"] == 0 and not row["ignored"] for row in work_rows),
            "work_ignored_rows": sum(row["ignored"] for row in work_rows),
        }
    audit = []
    for row in rows:
        key = (row["work"], row["piece"], row["split_index"], row["context_bins"])
        index = index_by_key.get(key)
        audit.append({
            "work": row["work"], "piece": row["piece"],
            "split_index": row["split_index"], "context_bins": row["context_bins"],
            "left_sequence_bins": len(row["left_sequence"]),
            "right_sequence_bins": len(row["right_sequence"]),
            "label": None if row["ignored"] else row["label"],
            "ignored": row["ignored"],
            "status": "ignored_tolerance" if row["ignored"] else "eligible",
            "sample_weight": 0.0 if index is None else float(sample[index]),
            "natural_weight": 0.0 if index is None else float(natural[index]),
            "ignore_tolerance_qb": float(ignore_tolerance_qb),
            **counts[row["work"]],
        })
    maximum_context = max(contexts)
    left_sequences, left_masks = _pad_sequence_list(
        [row["left_sequence"] for row in eligible], maximum_context)
    right_sequences, right_masks = _pad_sequence_list(
        [row["right_sequence"] for row in eligible], maximum_context)
    return NeuralTrainingSet(
        left=np.vstack([row["left"] for row in eligible]),
        right=np.vstack([row["right"] for row in eligible]),
        left_sequences=left_sequences,
        right_sequences=right_sequences,
        left_masks=left_masks,
        right_masks=right_masks,
        labels=np.asarray([row["label"] for row in eligible], dtype=float),
        sample_weights=sample,
        natural_weights=natural,
        works=tuple(row["work"] for row in eligible),
        pieces=tuple(row["piece"] for row in eligible),
        split_indices=np.asarray([row["split_index"] for row in eligible], dtype=int),
        contexts=np.asarray([row["context_bins"] for row in eligible], dtype=int),
        audit=tuple(audit),
    )


class _FrozenDistance:
    """DistanceSpec-compatible frozen model plus its monotone training head."""

    name: str

    def __init__(self, *, epoch: int, seed: int, intercept: float,
                 raw_gain: float, calibrator: ProbabilityCalibrator | None = None):
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.intercept = float(intercept)
        self.raw_gain = float(raw_gain)
        self.calibrator = calibrator

    @property
    def gain(self) -> float:
        return float(np.logaddexp(0.0, self.raw_gain)) + 1e-6

    def transform_batch(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, value: np.ndarray) -> np.ndarray:
        return self.transform_batch(np.asarray(value, dtype=float)[None, :])[0]

    def __call__(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.linalg.norm(self.transform(left) - self.transform(right)))

    def batch_distance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_repr = self.transform_batch(left)
        right_repr = self.transform_batch(right)
        if left_repr.shape != right_repr.shape:
            raise ValueError("left and right batches must have matching shapes")
        return np.linalg.norm(left_repr - right_repr, axis=1)

    def predict_boundary_probability(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = self.batch_distance(left, right)
        linear = np.clip(self.intercept + self.gain * distance, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return self.calibrator.transform(probability) if self.calibrator else probability

    def with_calibrator(self, calibrator: ProbabilityCalibrator) -> "_FrozenDistance":
        return replace(self, calibrator=calibrator)  # dataclass subclasses

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "epoch": self.epoch, "seed": self.seed,
            "boundary_head_intercept": self.intercept,
            "boundary_head_gain": self.gain,
            "probability_calibration": (
                self.calibrator.as_dict() if self.calibrator else None),
        }


@dataclass(frozen=True)
class NeuralPitchClassDistance(_FrozenDistance):
    """Frozen circular-convolution Siamese pitch-class distance."""

    state: Mapping[str, np.ndarray]
    epoch: int
    seed: int
    intercept: float
    raw_gain: float
    calibrator: ProbabilityCalibrator | None = None
    channels: int = 16
    architecture: str = "harmonic_tcn_v3"
    dilations: tuple[int, ...] = DEFAULT_HARMONIC_DILATIONS
    temporal_dilations: tuple[int, ...] = DEFAULT_TEMPORAL_DILATIONS
    temporal_dropout: float = 0.15
    name: str = "neural_circular_siamese"

    def __post_init__(self) -> None:
        if self.channels < 1 or self.epoch < 1:
            raise ValueError("channels and epoch must be positive")
        required = set(_encoder_module(
            self.architecture, self.channels, self.dilations,
            self.temporal_dilations, self.temporal_dropout).state_dict())
        if set(self.state) != required:
            raise ValueError(f"neural state must contain {sorted(required)}")

    @property
    def gain(self) -> float:
        return float(np.logaddexp(0.0, self.raw_gain)) + 1e-6

    def _module(self):
        torch, _, _ = _require_torch()
        module = _encoder_module(
            self.architecture, self.channels, self.dilations,
            self.temporal_dilations, self.temporal_dropout)
        state = {name: torch.as_tensor(np.asarray(value), dtype=torch.float32)
                 for name, value in self.state.items()}
        module.load_state_dict(state)
        module.eval()
        return module

    def transform_batch(self, values: np.ndarray) -> np.ndarray:
        if self.architecture in TEMPORAL_ARCHITECTURES:
            values = np.asarray(values, dtype=float)
            if values.ndim == 1:
                values = values[None, :]
            return self.transform_sequence_batch(values[:, None, :])
        values = _l1_batch(values)
        torch, _, _ = _require_torch()
        with torch.no_grad():
            output = self._module()(torch.as_tensor(values, dtype=torch.float32))
        return output.cpu().numpy().astype(float)

    def transform_sequence_batch(
        self,
        values: np.ndarray,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        values, masks = _l1_sequence_batch(values, masks)
        if self.architecture not in TEMPORAL_ARCHITECTURES:
            aggregate = (values * masks[:, :, None]).sum(axis=1)
            return self.transform_batch(aggregate)
        torch, _, _ = _require_torch()
        with torch.no_grad():
            output = self._module()(
                torch.as_tensor(values, dtype=torch.float32),
                torch.as_tensor(masks, dtype=torch.bool))
        return output.cpu().numpy().astype(float)

    def batch_sequence_distance(
        self,
        left: np.ndarray,
        right: np.ndarray,
        left_masks: np.ndarray | None = None,
        right_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        left_repr = self.transform_sequence_batch(left, left_masks)
        right_repr = self.transform_sequence_batch(right, right_masks)
        if left_repr.shape != right_repr.shape:
            raise ValueError("left and right sequence batches must match")
        return np.linalg.norm(left_repr - right_repr, axis=1)

    def predict_boundary_probability_sequence(
        self,
        left: np.ndarray,
        right: np.ndarray,
        left_masks: np.ndarray | None = None,
        right_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        distance = self.batch_sequence_distance(
            left, right, left_masks, right_masks)
        linear = np.clip(self.intercept + self.gain * distance, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        return self.calibrator.transform(probability) if self.calibrator else probability

    def boundary_contrast_batch(
        self,
        matrix: np.ndarray,
        contexts: Sequence[int],
    ) -> np.ndarray:
        """Return learned left/right sequence distances for every split/context."""
        matrix = np.asarray(matrix, dtype=float)
        contexts = tuple(int(value) for value in contexts)
        if self.architecture not in TEMPORAL_ARCHITECTURES:
            representation = self.transform_batch(matrix)
            pairwise = np.linalg.norm(
                representation[:, None, :] - representation[None, :, :], axis=2)
            output = np.zeros((max(0, len(matrix) - 1), len(contexts)), dtype=float)
            for column, context in enumerate(contexts):
                for split in range(1, len(matrix)):
                    left_indices = np.arange(max(0, split - context), split)
                    right_indices = np.arange(split, min(len(matrix), split + context))
                    cross = float(pairwise[np.ix_(left_indices, right_indices)].mean())

                    def within(indices):
                        if len(indices) < 2:
                            return 0.0
                        block = pairwise[np.ix_(indices, indices)]
                        return float(block[np.triu_indices(len(indices), 1)].mean())

                    output[split - 1, column] = cross - 0.5 * (
                        within(left_indices) + within(right_indices))
            return output
        left, right = [], []
        for split in range(1, len(matrix)):
            for context in contexts:
                left.append(matrix[max(0, split - context):split])
                right.append(matrix[split:min(len(matrix), split + context)])
        if not left:
            return np.zeros((0, len(contexts)), dtype=float)
        maximum = max(contexts)
        left_values, left_masks = _pad_sequence_list(left, maximum)
        right_values, right_masks = _pad_sequence_list(right, maximum)
        values = self.batch_sequence_distance(
            left_values, right_values, left_masks, right_masks)
        return values.reshape(len(matrix) - 1, len(contexts))

    def transform_contextual_bins(
        self,
        matrix: np.ndarray,
        context_radius: int,
    ) -> np.ndarray:
        """Embed fixed chronological neighbourhoods used by leaf affinity."""
        matrix = np.asarray(matrix, dtype=float)
        radius = int(context_radius)
        if radius < 0:
            raise ValueError("context_radius must be non-negative")
        if self.architecture not in TEMPORAL_ARCHITECTURES or radius == 0:
            return self.transform_batch(matrix)
        windows = [
            matrix[max(0, index - radius):min(len(matrix), index + radius + 1)]
            for index in range(len(matrix))
        ]
        values, masks = _pad_sequence_list(windows, 2 * radius + 1)
        return self.transform_sequence_batch(values, masks)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        metadata = self.as_dict() | {
            "channels": self.channels,
            "architecture": self.architecture,
            "dilations": list(self.dilations),
            "temporal_dilations": list(self.temporal_dilations),
            "temporal_dropout": self.temporal_dropout,
        }
        arrays = {name.replace(".", "__"): np.asarray(value) for name, value in self.state.items()}
        np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralPitchClassDistance":
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"]))
            state = {name.replace("__", "."): np.asarray(payload[name]).copy()
                     for name in payload.files if name != "metadata"}
        calibration = metadata.get("probability_calibration")
        calibrator = None
        if calibration:
            calibrator = ProbabilityCalibrator(
                intercept=calibration["intercept"], slope=calibration["slope"],
                score_location=calibration["score_location"],
                score_scale=calibration["score_scale"], epsilon=calibration["epsilon"],
                fit_scope=calibration["fit_scope"])
        return cls(
            state=state, epoch=metadata["epoch"], seed=metadata["seed"],
            intercept=metadata["boundary_head_intercept"],
            raw_gain=float(np.log(np.expm1(max(metadata["boundary_head_gain"] - 1e-6, 1e-8)))),
            calibrator=calibrator, channels=metadata["channels"],
            architecture=metadata.get("architecture", "legacy_v1"),
            dilations=tuple(metadata.get("dilations", DEFAULT_HARMONIC_DILATIONS)),
            temporal_dilations=tuple(metadata.get(
                "temporal_dilations", DEFAULT_TEMPORAL_DILATIONS)),
            temporal_dropout=float(metadata.get("temporal_dropout", 0.15)))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name, "architecture": self.architecture,
            "channels": self.channels, "harmonic_dilations": list(self.dilations),
            "temporal_dilations": list(self.temporal_dilations),
            "temporal_dropout": self.temporal_dropout,
            "temporal_order_preserved": self.architecture in TEMPORAL_ARCHITECTURES,
            "distance": "0.5 * L2(L2-normalised shared embeddings)",
            "joint_transposition_invariance": "architectural",
            "epoch": self.epoch, "seed": self.seed,
            "boundary_head_intercept": self.intercept,
            "boundary_head_gain": self.gain,
            "probability_calibration": self.calibrator.as_dict() if self.calibrator else None,
        }
        return payload


@dataclass(frozen=True)
class DiagonalPitchClassDistance(_FrozenDistance):
    """Frozen 12-parameter diagonal Mahalanobis comparison model."""

    weights: np.ndarray
    epoch: int
    seed: int
    intercept: float
    raw_gain: float
    calibrator: ProbabilityCalibrator | None = None
    name: str = "diagonal_mahalanobis"

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=float)
        if weights.shape != (12,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("diagonal weights must be a non-negative normalised 12-vector")
        object.__setattr__(self, "weights", weights)

    @property
    def gain(self) -> float:
        return float(np.logaddexp(0.0, self.raw_gain)) + 1e-6

    def transform_batch(self, values: np.ndarray) -> np.ndarray:
        return _l1_batch(values) * np.sqrt(self.weights)

    def save(self, path: str | Path) -> None:
        metadata = self.as_dict()
        np.savez_compressed(path, metadata=np.asarray(json.dumps(metadata)), weights=self.weights)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "weights": self.weights.tolist(),
            "distance": "L2 after softmax-normalised diagonal pitch-class scaling",
            "epoch": self.epoch, "seed": self.seed,
            "boundary_head_intercept": self.intercept,
            "boundary_head_gain": self.gain,
            "probability_calibration": self.calibrator.as_dict() if self.calibrator else None,
        }


@dataclass(frozen=True)
class DistanceCheckpoint:
    epoch: int
    distance: NeuralPitchClassDistance | DiagonalPitchClassDistance


def _trainable_model(kind: str, channels: int,
                     architecture: str = "harmonic_tcn_v3",
                     dilations: Sequence[int] = DEFAULT_HARMONIC_DILATIONS,
                     temporal_dilations: Sequence[int] = DEFAULT_TEMPORAL_DILATIONS,
                     temporal_dropout: float = 0.15):
    torch, nn, functional = _require_torch()

    class BoundaryDistanceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            if kind == "neural":
                self.encoder = _encoder_module(
                    architecture, channels, dilations,
                    temporal_dilations, temporal_dropout)
            elif kind == "diagonal":
                self.pitch_logits = nn.Parameter(torch.zeros(12))
            else:
                raise ValueError("kind must be neural or diagonal")
            self.intercept = nn.Parameter(torch.tensor(0.0))
            self.raw_gain = nn.Parameter(torch.tensor(0.0))

        def encode(self, values, mask=None):
            if self.kind == "neural":
                return self.encoder(values, mask)
            total = values.sum(dim=1, keepdim=True)
            values = torch.where(total > 1e-12, values / total.clamp_min(1e-12),
                                 torch.zeros_like(values))
            return values * torch.sqrt(torch.softmax(self.pitch_logits, dim=0))

        def forward(self, left, right, left_mask=None, right_mask=None):
            distance = torch.linalg.vector_norm(
                self.encode(left, left_mask) - self.encode(right, right_mask), dim=1)
            gain = functional.softplus(self.raw_gain) + 1e-6
            return self.intercept + gain * distance, distance

    return BoundaryDistanceModel()


def _freeze_model(model, kind: str, epoch: int, seed: int, channels: int,
                  architecture: str, dilations: Sequence[int],
                  temporal_dilations: Sequence[int], temporal_dropout: float):
    torch, _, _ = _require_torch()
    intercept = float(model.intercept.detach().cpu())
    raw_gain = float(model.raw_gain.detach().cpu())
    if kind == "neural":
        state = {name: value.detach().cpu().numpy().copy()
                 for name, value in model.encoder.state_dict().items()}
        return NeuralPitchClassDistance(
            state=state, epoch=epoch, seed=seed, intercept=intercept,
            raw_gain=raw_gain, channels=channels, architecture=architecture,
            dilations=tuple(int(value) for value in dilations),
            temporal_dilations=tuple(int(value) for value in temporal_dilations),
            temporal_dropout=float(temporal_dropout))
    weights = torch.softmax(model.pitch_logits.detach(), dim=0).cpu().numpy()
    return DiagonalPitchClassDistance(
        weights=weights, epoch=epoch, seed=seed, intercept=intercept,
        raw_gain=raw_gain)


def fit_natural_probability_calibration(
    distance: NeuralPitchClassDistance | DiagonalPitchClassDistance,
    training: NeuralTrainingSet,
) -> NeuralPitchClassDistance | DiagonalPitchClassDistance:
    """Fit monotone calibration using training works and natural prevalence."""
    if (isinstance(distance, NeuralPitchClassDistance)
            and distance.architecture in TEMPORAL_ARCHITECTURES):
        raw = distance.predict_boundary_probability_sequence(
            training.left_sequences, training.right_sequences,
            training.left_masks, training.right_masks)
    else:
        raw = distance.predict_boundary_probability(training.left, training.right)
    calibrator = fit_probability_calibrator(
        raw, training.labels, training.natural_weights)
    return replace(distance, calibrator=calibrator)


def train_distance_checkpoints(
    training: NeuralTrainingSet,
    *,
    kind: str = "neural",
    checkpoint_epochs: Sequence[int] = (20, 40, 60, 80, 100),
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 64,
    gradient_clip: float = 1.0,
    channels: int = 16,
    architecture: str = "harmonic_tcn_v3",
    dilations: Sequence[int] = DEFAULT_HARMONIC_DILATIONS,
    temporal_dilations: Sequence[int] = DEFAULT_TEMPORAL_DILATIONS,
    temporal_dropout: float = 0.15,
    metric_loss_weight: float = 0.0,
    ranking_loss_weight: float = 0.0,
    contrastive_margin: float = 0.5,
    ranking_margin: float = 0.05,
    minimum_learning_rate: float = 1e-5,
    device: str = "auto",
    seed: int = 20260818,
) -> tuple[list[DistanceCheckpoint], list[dict[str, Any]]]:
    """Train checkpoints with BCE and optional metric/ranking supervision."""
    checkpoint_epochs = tuple(sorted(set(int(value) for value in checkpoint_epochs)))
    dilations = tuple(int(value) for value in dilations)
    temporal_dilations = tuple(int(value) for value in temporal_dilations)
    if (not checkpoint_epochs or checkpoint_epochs[0] < 1 or learning_rate <= 0
            or weight_decay < 0 or batch_size < 1 or gradient_clip <= 0 or channels < 1):
        raise ValueError("invalid neural training configuration")
    if (architecture not in NEURAL_ARCHITECTURES or not dilations
            or any(value < 1 for value in dilations)
            or not temporal_dilations or any(value < 1 for value in temporal_dilations)
            or not 0 <= temporal_dropout < 1
            or metric_loss_weight < 0 or ranking_loss_weight < 0
            or not 0 < contrastive_margin <= 1 or ranking_margin < 0
            or minimum_learning_rate < 0 or minimum_learning_rate > learning_rate):
        raise ValueError("invalid neural architecture or loss configuration")
    torch, _, functional = _require_torch()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    actual_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if actual_device == "auto":
        actual_device = "cpu"
    torch_device = torch.device(actual_device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:  # pragma: no cover - old torch fallback
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):  # pragma: no cover
        pass
    model = _trainable_model(
        kind, channels, architecture, dilations,
        temporal_dilations, temporal_dropout).to(torch_device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=checkpoint_epochs[-1], eta_min=minimum_learning_rate)
        if kind == "neural" else None)
    temporal = kind == "neural" and architecture in TEMPORAL_ARCHITECTURES
    left_values = training.left_sequences if temporal else training.left
    right_values = training.right_sequences if temporal else training.right
    left = torch.as_tensor(left_values, dtype=torch.float32, device=torch_device)
    right = torch.as_tensor(right_values, dtype=torch.float32, device=torch_device)
    left_masks = (torch.as_tensor(training.left_masks, dtype=torch.bool, device=torch_device)
                  if temporal else None)
    right_masks = (torch.as_tensor(training.right_masks, dtype=torch.bool, device=torch_device)
                   if temporal else None)
    labels = torch.as_tensor(training.labels, dtype=torch.float32, device=torch_device)
    weights = torch.as_tensor(training.sample_weights, dtype=torch.float32, device=torch_device)
    history: list[dict[str, Any]] = []
    checkpoints: list[DistanceCheckpoint] = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, checkpoint_epochs[-1] + 1):
        model.train()
        order = rng.permutation(len(left))
        for start in range(0, len(order), batch_size):
            indices = torch.as_tensor(
                order[start:start + batch_size], dtype=torch.long, device=torch_device)
            optimiser.zero_grad()
            logits, distances = model(
                left[indices], right[indices],
                left_masks[indices] if left_masks is not None else None,
                right_masks[indices] if right_masks is not None else None)
            losses = functional.binary_cross_entropy_with_logits(
                logits, labels[indices], reduction="none")
            batch_weights = weights[indices]
            bce = torch.sum(batch_weights * losses) / batch_weights.sum().clamp_min(1e-12)
            batch_labels = labels[indices]
            metric_losses = torch.where(
                batch_labels > 0.5,
                functional.relu(contrastive_margin - distances).square(),
                distances.square())
            metric = (torch.sum(batch_weights * metric_losses)
                      / batch_weights.sum().clamp_min(1e-12))
            positive = distances[batch_labels > 0.5]
            negative = distances[batch_labels <= 0.5]
            if kind == "neural" and len(positive) and len(negative):
                hard_count = min(len(negative), max(1, len(positive)))
                hard_negative = torch.topk(negative, hard_count).values
                ranking = functional.softplus(
                    ranking_margin + hard_negative[:, None] - positive[None, :]).mean()
            else:
                ranking = distances.sum() * 0.0
            neural_metric = metric_loss_weight if kind == "neural" else 0.0
            neural_ranking = ranking_loss_weight if kind == "neural" else 0.0
            loss = bce + neural_metric * metric + neural_ranking * ranking
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            logits, distances = model(
                left, right, left_masks, right_masks)
            losses = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            bce = torch.sum(weights * losses) / weights.sum()
            metric_losses = torch.where(
                labels > 0.5,
                functional.relu(contrastive_margin - distances).square(),
                distances.square())
            metric = torch.sum(weights * metric_losses) / weights.sum()
            positive_distances = distances[labels > 0.5]
            negative_distances = distances[labels <= 0.5]
            if kind == "neural":
                hard_count = min(len(positive_distances), len(negative_distances))
                hard_negative = torch.topk(negative_distances, hard_count).values
                weak_positive = torch.topk(
                    positive_distances, hard_count, largest=False).values
                ranking = functional.softplus(
                    ranking_margin + hard_negative - weak_positive).mean()
            else:
                ranking = distances.sum() * 0.0
            neural_metric = metric_loss_weight if kind == "neural" else 0.0
            neural_ranking = ranking_loss_weight if kind == "neural" else 0.0
            loss = bce + neural_metric * metric + neural_ranking * ranking
            probability = torch.sigmoid(logits)
            accuracy = float(torch.sum(weights * ((probability >= 0.5) == labels.bool())) / weights.sum())
            positive = training.labels == 1
            negative = ~positive
            row = {
                "epoch": float(epoch), "loss": float(loss),
                "bce_loss": float(bce), "metric_loss": float(metric),
                "ranking_loss": float(ranking),
                "learning_rate": float(optimiser.param_groups[0]["lr"]),
                "device": actual_device,
                "temporal_order_preserved": bool(temporal),
                "weighted_accuracy": accuracy,
                "mean_positive_distance": float(distances.cpu().numpy()[positive].mean()),
                "mean_negative_distance": float(distances.cpu().numpy()[negative].mean()),
            }
            history.append(row)
        if epoch in checkpoint_epochs:
            frozen = _freeze_model(
                model, kind, epoch, seed, channels, architecture, dilations,
                temporal_dilations, temporal_dropout)
            frozen = fit_natural_probability_calibration(frozen, training)
            checkpoints.append(DistanceCheckpoint(epoch, frozen))
        if scheduler is not None:
            scheduler.step()
    return checkpoints, history
