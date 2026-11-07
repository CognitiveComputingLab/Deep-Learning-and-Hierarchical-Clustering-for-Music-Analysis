"""Versioned, loss-aware shared representation for structural annotations.

The format deliberately separates tree-compatible form segments from flat
partitions, overlapping spans, and point events.  Converters must not invent a
hierarchy merely because a source contains structural information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0.0"
ANNOTATION_KINDS = {
    "hierarchical_segment", "partition_segment", "span", "point",
}
COORDINATE_SYSTEMS = {"quarterbeat", "measure_beat", "mixed"}


@dataclass
class SharedAnnotation:
    annotation_id: str
    kind: str
    layer: str
    label: str
    start_qb: float | None = None
    end_qb: float | None = None
    start_measure: int | None = None
    start_beat: float | None = None
    end_measure: int | None = None
    end_beat: float | None = None
    level: int | None = None
    parent_id: str | None = None
    tree_id: str | None = None
    staff: str | None = None
    source_type: str | None = None
    source: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SharedAnnotation":
        return cls(**value)


@dataclass
class SharedPiece:
    dataset: str
    piece_id: str
    work_id: str
    coordinate_system: str
    annotations: list[SharedAnnotation]
    duration_qb: float | None = None
    final_measure: int | None = None
    final_beat: float | None = None
    score_path: str | None = None
    annotation_path: str | None = None
    layer_definitions: dict[str, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SharedPiece":
        payload = dict(value)
        payload["annotations"] = [
            item if isinstance(item, SharedAnnotation)
            else SharedAnnotation.from_dict(item)
            for item in payload.get("annotations", [])
        ]
        return cls(**payload)

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8")
        return target

    @classmethod
    def read_json(cls, path: str | Path) -> "SharedPiece":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _finite_or_none(value: float | None) -> bool:
    return value is None or (isinstance(value, (int, float)) and math.isfinite(value))


def _position(annotation: SharedAnnotation, end: bool = False):
    qb = annotation.end_qb if end else annotation.start_qb
    if qb is not None:
        return ("quarterbeat", float(qb))
    measure = annotation.end_measure if end else annotation.start_measure
    beat = annotation.end_beat if end else annotation.start_beat
    if measure is not None:
        return ("measure_beat", (int(measure), float(beat or 0.0)))
    return None


def validate_piece(piece: SharedPiece, *, raise_on_error: bool = True) -> list[str]:
    """Validate syntax and tree claims without rejecting legal overlaps."""
    errors: list[str] = []
    if piece.schema_version != SCHEMA_VERSION:
        errors.append(f"unsupported schema_version {piece.schema_version!r}")
    if piece.coordinate_system not in COORDINATE_SYSTEMS:
        errors.append(f"invalid coordinate_system {piece.coordinate_system!r}")
    if not piece.dataset or not piece.piece_id or not piece.work_id:
        errors.append("dataset, piece_id and work_id are required")
    if not _finite_or_none(piece.duration_qb) or (
            piece.duration_qb is not None and piece.duration_qb <= 0):
        errors.append("duration_qb must be finite and positive")

    by_id: dict[str, SharedAnnotation] = {}
    for item in piece.annotations:
        if item.annotation_id in by_id:
            errors.append(f"duplicate annotation_id {item.annotation_id!r}")
        by_id[item.annotation_id] = item
        if item.kind not in ANNOTATION_KINDS:
            errors.append(f"{item.annotation_id}: invalid kind {item.kind!r}")
        if not item.layer:
            errors.append(f"{item.annotation_id}: layer is required")
        for name in ("start_qb", "end_qb", "start_beat", "end_beat"):
            if not _finite_or_none(getattr(item, name)):
                errors.append(f"{item.annotation_id}: {name} is not finite")
        start, end = _position(item), _position(item, end=True)
        if start is None:
            errors.append(f"{item.annotation_id}: no start coordinate")
        if item.kind != "point":
            if end is None:
                errors.append(f"{item.annotation_id}: segment/span has no end")
            elif start and start[0] == end[0] and end[1] <= start[1]:
                errors.append(f"{item.annotation_id}: end must be after start")
        if item.kind == "hierarchical_segment":
            if item.tree_id is None or item.level is None:
                errors.append(f"{item.annotation_id}: hierarchy needs tree_id and level")
            if item.level is not None and item.level < 0:
                errors.append(f"{item.annotation_id}: level must be non-negative")

    for item in piece.annotations:
        if item.parent_id is None:
            continue
        parent = by_id.get(item.parent_id)
        if parent is None:
            errors.append(f"{item.annotation_id}: missing parent {item.parent_id!r}")
            continue
        if item.kind != "hierarchical_segment" or parent.kind != "hierarchical_segment":
            errors.append(f"{item.annotation_id}: parent links are only valid for hierarchy segments")
        if item.tree_id != parent.tree_id:
            errors.append(f"{item.annotation_id}: parent belongs to another tree")
        if item.level is not None and parent.level is not None and item.level <= parent.level:
            errors.append(f"{item.annotation_id}: child level must exceed parent level")
        child_start, child_end = _position(item), _position(item, end=True)
        parent_start, parent_end = _position(parent), _position(parent, end=True)
        if (child_start and child_end and parent_start and parent_end
                and child_start[0] == parent_start[0] == child_end[0] == parent_end[0]
                and not (parent_start[1] <= child_start[1] < child_end[1] <= parent_end[1])):
            errors.append(f"{item.annotation_id}: child is not contained by parent")

    if raise_on_error and errors:
        raise ValueError("Invalid shared annotation piece:\n- " + "\n- ".join(errors))
    return errors


def annotations_by_layer(piece: SharedPiece, layer: str) -> list[SharedAnnotation]:
    return [item for item in piece.annotations if item.layer == layer]


def hierarchy_roots(piece: SharedPiece, tree_id: str) -> list[SharedAnnotation]:
    return [item for item in piece.annotations
            if item.tree_id == tree_id and item.parent_id is None]


def iter_boundaries(piece: SharedPiece, *, layers: Iterable[str] | None = None,
                    max_level: int | None = None) -> list[float]:
    """Return unique internal quarterbeat boundaries for compatible layers."""
    selected = set(layers) if layers is not None else None
    values: set[float] = set()
    for item in piece.annotations:
        if selected is not None and item.layer not in selected:
            continue
        if item.kind not in {"hierarchical_segment", "partition_segment"}:
            continue
        if max_level is not None and item.level is not None and item.level > max_level:
            continue
        if item.start_qb is not None and item.start_qb > 0:
            values.add(float(item.start_qb))
        if (item.end_qb is not None and piece.duration_qb is not None
                and item.end_qb < piece.duration_qb):
            values.add(float(item.end_qb))
    return sorted(values)
