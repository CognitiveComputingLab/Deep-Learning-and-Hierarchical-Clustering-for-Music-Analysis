"""Shared, annotation-safe representation for the three evaluation corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from greedy_clustering import collapse_empty_bins


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    onset_qb: float
    duration_qb: float

    def __post_init__(self) -> None:
        if not 0 <= int(self.pitch) <= 127:
            raise ValueError("pitch must be a MIDI number in [0, 127]")
        if not np.isfinite(self.onset_qb) or self.onset_qb < 0:
            raise ValueError("onset_qb must be finite and non-negative")
        if not np.isfinite(self.duration_qb) or self.duration_qb <= 0:
            raise ValueError("duration_qb must be finite and positive")


@dataclass(frozen=True)
class BoundaryAnnotation:
    time_qb: float
    level: int = 1
    label: str = ""


@dataclass(frozen=True)
class SpanAnnotation:
    start_qb: float
    end_qb: float
    level: int = 1
    label: str = ""


@dataclass(frozen=True)
class IntervalAnnotation:
    start_qb: float
    end_qb: float
    kind: str
    label: str = ""


@dataclass(frozen=True)
class AnnotationBundle:
    boundaries: tuple[BoundaryAnnotation, ...] = ()
    spans: tuple[SpanAnnotation, ...] = ()
    intervals: tuple[IntervalAnnotation, ...] = ()


@dataclass(frozen=True)
class FeaturePiece:
    """Inference input.  It intentionally has no annotation field."""

    dataset: str
    work_id: str
    piece_id: str
    notes: tuple[NoteEvent, ...]
    duration_qb: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatedPiece:
    dataset: str
    work_id: str
    piece_id: str
    notes: tuple[NoteEvent, ...]
    duration_qb: float
    annotations: AnnotationBundle
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset or not self.work_id or not self.piece_id:
            raise ValueError("dataset, work_id, and piece_id are required")
        if not np.isfinite(self.duration_qb) or self.duration_qb <= 0:
            raise ValueError("duration_qb must be finite and positive")
        if any(note.onset_qb + note.duration_qb > self.duration_qb + 1e-6
               for note in self.notes):
            raise ValueError("a note extends beyond duration_qb")
        for boundary in self.annotations.boundaries:
            if not 0 < boundary.time_qb < self.duration_qb:
                raise ValueError("boundary must be internal to the piece")
        for span in self.annotations.spans:
            if not 0 <= span.start_qb < span.end_qb <= self.duration_qb + 1e-6:
                raise ValueError("invalid annotation span")
        for interval in self.annotations.intervals:
            if not 0 <= interval.start_qb < interval.end_qb <= self.duration_qb + 1e-6:
                raise ValueError("invalid annotation interval")

    def inference_view(self) -> FeaturePiece:
        return FeaturePiece(self.dataset, self.work_id, self.piece_id, self.notes,
                            self.duration_qb, self.metadata)


def select_bin_size(duration_qb: float, candidates: Sequence[float] = (8, 16, 32),
                    max_leaves: int = 192) -> float:
    """Select the finest pre-registered resolution satisfying the DP cap."""
    candidates = tuple(sorted(float(value) for value in candidates))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("candidate bin sizes must be positive")
    if max_leaves < 2:
        raise ValueError("max_leaves must be at least two")
    for value in candidates:
        if int(np.ceil(float(duration_qb) / value)) <= max_leaves:
            return value
    raise ValueError(
        f"piece needs more than {max_leaves} leaves even at {candidates[-1]} quarterbeats")


def notes_to_pc_bins(piece: FeaturePiece, bin_size_qb: float,
                     collapse_empty: bool = True) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Create duration-weighted pitch-class bins from annotation-free input."""
    if bin_size_qb <= 0:
        raise ValueError("bin_size_qb must be positive")
    n_bins = int(np.ceil(piece.duration_qb / bin_size_qb))
    matrix = np.zeros((n_bins, 12), dtype=float)
    for note in piece.notes:
        start = float(note.onset_qb)
        end = min(float(note.onset_qb + note.duration_qb), piece.duration_qb)
        first = max(0, int(start // bin_size_qb))
        last = min(n_bins - 1, int(min(end, piece.duration_qb - 1e-9) // bin_size_qb))
        for index in range(first, last + 1):
            lo = index * bin_size_qb
            hi = min((index + 1) * bin_size_qb, piece.duration_qb)
            matrix[index, int(note.pitch) % 12] += max(0.0, min(end, hi) - max(start, lo))
    bounds = [(index * bin_size_qb, min((index + 1) * bin_size_qb, piece.duration_qb))
              for index in range(n_bins)]
    return collapse_empty_bins(matrix, bounds) if collapse_empty else (matrix, bounds)


def intervals_iou(predicted: Iterable[tuple[float, float]],
                  reference: Iterable[tuple[float, float]]) -> float:
    """Duration-weighted intersection-over-union for interval annotations."""
    predicted = sorted((float(a), float(b)) for a, b in predicted if b > a)
    reference = sorted((float(a), float(b)) for a, b in reference if b > a)
    points = sorted({value for interval in predicted + reference for value in interval})
    if len(points) < 2:
        return 1.0 if not predicted and not reference else 0.0
    intersection = union = 0.0
    for left, right in zip(points[:-1], points[1:]):
        midpoint = (left + right) / 2
        p = any(a <= midpoint < b for a, b in predicted)
        r = any(a <= midpoint < b for a, b in reference)
        intersection += (right - left) * bool(p and r)
        union += (right - left) * bool(p or r)
    return intersection / union if union else 1.0
