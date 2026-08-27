"""Score/annotation adapters for zero-shot external-corpus evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from annotated_piece import (
    AnnotatedPiece, AnnotationBundle, BoundaryAnnotation, FeaturePiece,
    IntervalAnnotation, NoteEvent, SpanAnnotation,
)
from taking_form_loader import read_taking_form_events


def _music21():
    try:
        from music21 import converter
    except ImportError as error:
        raise RuntimeError(
            "music21 is required to convert the pinned Humdrum scores; "
            "install the locked project requirements") from error
    return converter


def parse_symbolic_score(score_path: str | Path, *, dataset: str, work_id: str,
                         piece_id: str) -> tuple[FeaturePiece, Any]:
    """Parse a symbolic score and return an annotation-free feature view."""
    path = Path(score_path)
    score = _music21().parse(str(path))
    notes: list[NoteEvent] = []
    for element in score.flatten().notes:
        onset = float(element.getOffsetInHierarchy(score))
        duration = float(element.duration.quarterLength)
        pitches = list(element.pitches) if hasattr(element, "pitches") else [element.pitch]
        for pitch in pitches:
            if duration > 0:
                notes.append(NoteEvent(int(pitch.midi), onset, duration))
    if not notes:
        raise ValueError(f"score contains no timed notes: {path}")
    duration = max(float(score.highestTime),
                   max(note.onset_qb + note.duration_qb for note in notes))
    feature = FeaturePiece(
        dataset, work_id, piece_id, tuple(notes), duration,
        {"score_path": str(path), "score_format": path.suffix.lower()})
    return feature, score


def _measure_map(score: Any) -> dict[int, Any]:
    parts = list(score.parts)
    source = parts[0] if parts else score
    result: dict[int, Any] = {}
    for measure in source.getElementsByClass("Measure"):
        number = int(measure.number)
        result.setdefault(number, measure)
    return result


def _measure_beat_qb(score: Any, measures: dict[int, Any], measure_number: int,
                     beat_text: str) -> float:
    if measure_number not in measures:
        raise ValueError(f"measure {measure_number} is absent from the score")
    measure = measures[measure_number]
    offset = float(measure.getOffsetInHierarchy(score))
    try:
        beat = float(beat_text) if str(beat_text).strip() else 1.0
    except ValueError as error:
        raise ValueError(
            f"non-numeric beat {beat_text!r} at measure {measure_number}") from error
    signature = measure.getContextByClass("TimeSignature")
    beat_qb = float(signature.beatDuration.quarterLength) if signature else 1.0
    # Positive music21 beat numbers are one-based.  Manual negative numbers
    # denote an event before the target downbeat.
    local = (beat - 1.0) * beat_qb if beat > 0 else beat * beat_qb
    return max(0.0, offset + local)


def load_taking_form_piece(annotation_path: str | Path, score_path: str | Path,
                           *, work_id: str, piece_id: str) -> AnnotatedPiece:
    feature, score = parse_symbolic_score(
        score_path, dataset="taking_form", work_id=work_id, piece_id=piece_id)
    events, audit = read_taking_form_events(annotation_path)
    if audit.issues:
        raise ValueError("; ".join(audit.issues))
    measures = _measure_map(score)
    timed = [(_measure_beat_qb(score, measures, event.measure, event.beat), event)
             for event in events]
    boundaries: list[BoundaryAnnotation] = []
    spans: list[SpanAnnotation] = []
    for level in sorted({event.level for _, event in timed}):
        level_events = sorted((time, event) for time, event in timed
                              if event.level == level)
        unique: list[tuple[float, Any]] = []
        for time, event in level_events:
            if unique and np.isclose(time, unique[-1][0]):
                unique[-1] = (time, event)
            else:
                unique.append((time, event))
        for index, (start, event) in enumerate(unique):
            end = unique[index + 1][0] if index + 1 < len(unique) else feature.duration_qb
            if 0 < start < feature.duration_qb:
                boundaries.append(BoundaryAnnotation(start, level, event.label))
            if end > start + 1e-9:
                spans.append(SpanAnnotation(start, min(end, feature.duration_qb),
                                            level, event.label))
    return AnnotatedPiece(
        feature.dataset, feature.work_id, feature.piece_id, feature.notes,
        feature.duration_qb, AnnotationBundle(tuple(boundaries), tuple(spans), ()),
        {**feature.metadata, "annotation_path": str(annotation_path),
         "taking_form_audit": audit.__dict__})


def read_fugue_dez(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    if not isinstance(labels, list):
        raise ValueError("fugue .dez labels must be a list")
    result: dict[str, list[dict[str, Any]]] = {
        "cadences": [], "pedals": [], "overlapping_motives": []}
    for label in labels:
        kind = str(label.get("type", "")).strip()
        start = float(label.get("start", 0))
        duration = float(label.get("duration", 0) or 0)
        row = {**label, "start": start, "duration": duration}
        if kind == "Cadence":
            result["cadences"].append(row)
        elif kind == "Pedal":
            result["pedals"].append(row)
        elif kind.startswith(("S", "CS")):
            result["overlapping_motives"].append(row)
    for values in result.values():
        values.sort(key=lambda row: (row["start"], str(row.get("staff", ""))))
    return result


def load_fugue_piece(annotation_path: str | Path, score_path: str | Path,
                     *, work_id: str, piece_id: str) -> AnnotatedPiece:
    feature, _ = parse_symbolic_score(
        score_path, dataset="algomus_fugue", work_id=work_id, piece_id=piece_id)
    annotations = read_fugue_dez(annotation_path)
    boundaries = tuple(
        BoundaryAnnotation(row["start"], 1, str(row.get("tag", "")).strip())
        for row in annotations["cadences"]
        if 0 < row["start"] < feature.duration_qb)
    intervals = tuple(
        IntervalAnnotation(row["start"],
                           min(feature.duration_qb, row["start"] + row["duration"]),
                           "pedal", str(row.get("tag", "")).strip())
        for row in annotations["pedals"]
        if row["duration"] > 0 and row["start"] < feature.duration_qb
    )
    return AnnotatedPiece(
        feature.dataset, feature.work_id, feature.piece_id, feature.notes,
        feature.duration_qb, AnnotationBundle(boundaries, (), intervals),
        {**feature.metadata, "annotation_path": str(annotation_path),
         "overlapping_motive_count": len(annotations["overlapping_motives"])})
