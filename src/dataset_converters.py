"""Converters from source corpora to :mod:`shared_annotation_format`."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from shared_annotation_format import SharedAnnotation, SharedPiece, validate_piece


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return float(text)


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _abc_work_id(piece_id: str) -> str:
    return piece_id.rsplit("_", 1)[0]


def convert_dcml_piece(notes_path: str | Path, harmonies_path: str | Path, *,
                       dataset: str = "ABC_DCML") -> SharedPiece:
    """Convert DCML note/harmony TSVs without claiming a hierarchy."""
    notes_path, harmonies_path = Path(notes_path), Path(harmonies_path)
    piece_id = notes_path.name.removesuffix(".notes.tsv")
    notes = pd.read_csv(notes_path, sep="\t")
    harmonies = pd.read_csv(harmonies_path, sep="\t")
    if harmonies.empty or notes.empty:
        raise ValueError(f"{piece_id}: notes and harmonies must be non-empty")
    for name in ("quarterbeats", "duration_qb"):
        if name not in notes or name not in harmonies:
            raise ValueError(f"{piece_id}: missing required column {name!r}")
    total = max(
        float(_number(row.quarterbeats, 0.0) + _number(row.duration_qb, 0.0))
        for row in notes.itertuples())
    ordered = harmonies.assign(
        _qb=harmonies.quarterbeats.map(lambda x: _number(x, 0.0)),
        _dur=harmonies.duration_qb.map(lambda x: _number(x, 0.0)),
    ).sort_values(["_qb", "mc" if "mc" in harmonies else "_qb"])
    annotations: list[SharedAnnotation] = []

    # Consecutive local-key values form one flat partition.  This is structural
    # segmentation but not an expert hierarchy.
    key_starts: list[tuple[float, str, int]] = []
    previous = None
    for source_row, row in ordered.iterrows():
        label = _text(row.get("localkey", "")) or "unknown"
        if label != previous:
            key_starts.append((float(row._qb), label, int(source_row)))
            previous = label
    for index, (start, label, source_row) in enumerate(key_starts):
        end = key_starts[index + 1][0] if index + 1 < len(key_starts) else total
        if end <= start:
            continue
        annotations.append(SharedAnnotation(
            annotation_id=f"local_key:{index}", kind="partition_segment",
            layer="local_key", label=label, start_qb=start, end_qb=end,
            level=0, source_type="localkey",
            source={"row": source_row}))

    # The parsed DCML `pedal` column repeats the active pedal value across all
    # chord rows.  Collapse each consecutive run to one span.
    active: dict[str, Any] | None = None
    pedal_index = 0
    for source_row, row in ordered.iterrows():
        value = _text(row.get("pedal", ""))
        start = float(row._qb)
        row_end = start + float(row._dur)
        if active is not None and value != active["label"]:
            end = max(float(active["last_end"]), start)
            if end > active["start"]:
                annotations.append(SharedAnnotation(
                    annotation_id=f"pedal:{pedal_index}", kind="span",
                    layer="pedal", label=active["label"],
                    start_qb=active["start"], end_qb=end,
                    source_type="pedal",
                    source={"start_row": active["row"], "end_row": int(source_row)}))
                pedal_index += 1
            active = None
        if value:
            if active is None:
                active = {"label": value, "start": start,
                          "last_end": row_end, "row": int(source_row)}
            else:
                active["last_end"] = max(active["last_end"], row_end)
    if active is not None and active["last_end"] > active["start"]:
        annotations.append(SharedAnnotation(
            annotation_id=f"pedal:{pedal_index}", kind="span",
            layer="pedal", label=active["label"],
            start_qb=active["start"], end_qb=min(total, active["last_end"]),
            source_type="pedal", source={"start_row": active["row"]}))

    for column, layer in (("cadence", "cadence"), ("phraseend", "phrase_end")):
        if column not in ordered:
            continue
        event_index = 0
        for source_row, row in ordered.iterrows():
            value = _text(row.get(column, ""))
            if not value:
                continue
            annotations.append(SharedAnnotation(
                annotation_id=f"{layer}:{event_index}", kind="point",
                layer=layer, label=value, start_qb=float(row._qb),
                source_type=column, source={"row": int(source_row)}))
            event_index += 1

    piece = SharedPiece(
        dataset=dataset, piece_id=piece_id, work_id=_abc_work_id(piece_id),
        coordinate_system="quarterbeat", duration_qb=total,
        score_path=str(notes_path), annotation_path=str(harmonies_path),
        annotations=annotations,
        layer_definitions={
            "local_key": {"semantics": "flat_partition", "overlap": "disjoint",
                          "primary_boundary_reference": True},
            "pedal": {"semantics": "local_span", "overlap": "allowed"},
            "cadence": {"semantics": "point_event", "overlap": "allowed"},
            "phrase_end": {"semantics": "point_event", "overlap": "allowed"},
        },
        provenance={"format": "DCML TSV", "annotation_standard": "DCML harmony"},
        warnings=["local_key is a flat partition, not a hierarchical ground-truth tree"],
    )
    validate_piece(piece)
    return piece


def _measure_range(value: Any) -> tuple[int | None, int | None, str | None]:
    text = _text(value)
    match = re.match(r"^\s*(\d+)(?:[a-z])?(?:\s*-\s*(\d+)(?:[a-z])?)?", text, re.I)
    if not match:
        return None, None, None
    start = int(match.group(1))
    end = int(match.group(2) or start)
    repeat = text.split("=", 1)[1].strip() if "=" in text else None
    return start, end, repeat


def _taking_work_id(piece_id: str) -> str:
    return re.sub(r"(?:movt|movement|mvt)[-_ ]*\d+.*$", "", piece_id, flags=re.I).rstrip("-_")


def convert_taking_form(path: str | Path, *, dataset: str = "Taking_Form") -> SharedPiece:
    """Convert Taking Form's coarse-to-fine columns into an explicit tree."""
    path = Path(path)
    frame = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    parsed_rows: list[dict[str, Any]] = []
    unresolved = 0
    non_numeric_beats = 0
    for source_row, row in frame.iterrows():
        start, range_end, repeat = _measure_range(row.iloc[0])
        if start is None:
            if source_row == 0:
                continue
            unresolved += 1
            continue
        raw_beat = _text(row.iloc[1])
        beat_marker = None
        try:
            beat = _number(raw_beat, 1.0) or 0.0
        except (ValueError, ZeroDivisionError):
            # Some source rows contain a structural comment such as
            # X (Cadenza) in the beat column. Preserve it and use beat 1 as
            # an explicitly documented coordinate approximation.
            beat = 1.0
            beat_marker = raw_beat
            non_numeric_beats += 1
        parsed_rows.append({
            'source_row': int(source_row),
            'measure': int(start),
            'range_end': int(range_end or start),
            'raw_measure': _text(row.iloc[0]),
            'beat': float(beat),
            'raw_beat': raw_beat,
            'beat_marker': beat_marker,
            'repeat': repeat,
            'row': row,
            'coordinate_repair': None,
        })
    if not parsed_rows:
        raise ValueError(f"{path}: no parseable Taking Form rows")
    coordinate_repairs = 0
    for index in range(1, len(parsed_rows)):
        current = parsed_rows[index]
        previous = parsed_rows[index - 1]
        if current['measure'] >= previous['measure']:
            continue
        next_measure = next(
            (item['measure'] for item in parsed_rows[index + 1:]
             if item['measure'] > previous['measure']), None)
        candidate = previous['measure'] + 1
        repaired = candidate if next_measure is None or candidate < next_measure else previous['measure']
        delta = repaired - current['measure']
        current['coordinate_repair'] = {
            'reason': 'non_monotonic_source_measure',
            'original_measure': current['measure'],
            'repaired_measure': repaired,
        }
        current['measure'] = repaired
        current['range_end'] += delta
        coordinate_repairs += 1
    maximum_measure = max(item['range_end'] for item in parsed_rows)
    first_measure = parsed_rows[0]['measure']
    first_beat = parsed_rows[0]['beat']
    final_measure, final_beat = maximum_measure + 1, 1.0
    annotations: list[SharedAnnotation] = [SharedAnnotation(
        annotation_id="form:root", kind="hierarchical_segment",
        layer="form", label="ROOT", start_measure=first_measure,
        start_beat=first_beat, end_measure=final_measure, end_beat=final_beat,
        level=0, tree_id="form")]
    active: dict[int, SharedAnnotation] = {0: annotations[0]}
    next_id = 0
    repeat_rows = 0
    marker_index = 0
    for parsed in parsed_rows:
        source_row = parsed['source_row']
        measure = parsed['measure']
        beat = parsed['beat']
        repeat = parsed['repeat']
        row = parsed['row']
        if parsed['beat_marker']:
            annotations.append(SharedAnnotation(
                annotation_id=f'source_marker:{marker_index}', kind='point',
                layer='source_marker', label=parsed['beat_marker'],
                start_measure=measure, start_beat=beat,
                source_type='non_numeric_beat_marker',
                source={'row': source_row, 'raw_beat': parsed['raw_beat']}))
            marker_index += 1
        if repeat:
            repeat_rows += 1
        for column in range(2, len(row)):
            label = _text(row.iloc[column])
            if not label:
                continue
            depth = column - 1
            # The current position is an exclusive end for the previous node at
            # this level and all finer active levels.
            for old_depth in sorted([key for key in active if key >= depth], reverse=True):
                old = active.pop(old_depth)
                old.end_measure, old.end_beat = measure, beat
            parent_depths = [key for key in active if key < depth]
            parent = active[max(parent_depths)] if parent_depths else annotations[0]
            source = {"row": source_row, "column": column}
            if repeat:
                source["repeat_source"] = repeat
            if parsed['coordinate_repair']:
                source['coordinate_repair'] = parsed['coordinate_repair']
                source['raw_measure'] = parsed['raw_measure']
            if parsed['beat_marker']:
                source['raw_beat'] = parsed['raw_beat']
                source['approximated_beat'] = beat
            node = SharedAnnotation(
                annotation_id=f"form:{next_id}", kind="hierarchical_segment",
                layer="form", label=label, start_measure=measure,
                start_beat=beat, end_measure=final_measure, end_beat=final_beat,
                level=depth, parent_id=parent.annotation_id,
                tree_id="form", source_type="form_column", source=source)
            next_id += 1
            annotations.append(node)
            active[depth] = node
    for node in active.values():
        node.end_measure, node.end_beat = final_measure, final_beat

    # Remove zero-length artefacts but preserve their occurrence as a warning.
    zero_ids = {item.annotation_id for item in annotations
                if item.annotation_id != "form:root"
                and item.kind == 'hierarchical_segment'
                and (item.start_measure, item.start_beat) >= (item.end_measure, item.end_beat)}
    if zero_ids:
        annotations = [item for item in annotations if item.annotation_id not in zero_ids]
        for item in annotations:
            if item.parent_id in zero_ids:
                item.parent_id = "form:root"
    warnings = [
        "measure/beat coordinates require a matching symbolic score before pitch-based clustering",
        "the exclusive piece end is approximated as one measure after the largest encoded measure",
    ]
    if repeat_rows:
        warnings.append(
            f"{repeat_rows} repeat/comparison rows retain source references but are not score-expanded")
    if unresolved:
        warnings.append(f"{unresolved} non-coordinate rows were skipped")
    if zero_ids:
        warnings.append(f"{len(zero_ids)} zero-length annotations were removed")
    if coordinate_repairs:
        warnings.append(
            f'{coordinate_repairs} non-monotonic source measure values were repaired deterministically; original values are retained in annotation provenance')
    if non_numeric_beats:
        warnings.append(
            f'{non_numeric_beats} non-numeric beat markers were preserved as source_marker points and approximated at beat 1')
    piece_id = path.stem
    piece = SharedPiece(
        dataset=dataset, piece_id=piece_id, work_id=_taking_work_id(piece_id),
        coordinate_system="measure_beat", final_measure=final_measure,
        final_beat=final_beat, annotation_path=str(path), annotations=annotations,
        layer_definitions={
            "form": {"semantics": "hierarchical_form", "overlap": "nested",
                     "primary_tree_reference": True},
            "source_marker": {
                "semantics": "source_point_marker", "overlap": "allowed"},
        },
        provenance={"format": "Taking Form CSV", "level_order": "coarse_to_fine"},
        warnings=warnings,
    )
    validate_piece(piece)
    return piece


def _dez_work_id(piece_id: str, profile: str) -> str:
    if profile == "fugue":
        match = re.search(r"bwv\d+", piece_id, re.I)
        return match.group(0).lower() if match else piece_id.removesuffix("-ref")
    match = re.match(r"(k\d+)", piece_id, re.I)
    return match.group(1).lower() if match else piece_id.removesuffix("-ref")


def _dez_kind_layer(label: dict[str, Any], profile: str):
    source_type = _text(label.get("type")) or "untyped"
    line = _text(label.get("line"))
    duration = _number(label.get("duration"), None)
    if duration is None or duration <= 0:
        duration = _number(label.get("ioi"), None)
    if source_type == "Structure" and profile == "mozart_sonata" and line.startswith("top.") and duration:
        match = re.match(r"top\.(\d+)", line)
        return "hierarchical_segment", "form", int(match.group(1)) if match else 1, duration
    if source_type == "Tonality" and duration:
        return "partition_segment", "tonality", 0, duration
    if source_type == "Pedal" and duration:
        return "span", "pedal", None, duration
    if source_type == "Cadence":
        return "point", "cadence", None, None
    if source_type == "Structure" and duration:
        return "span", "structure", None, duration
    if source_type == "Structure":
        return "point", "structural_marker", None, None
    layer = re.sub(r"[^a-z0-9]+", "_", source_type.lower()).strip("_") or "other"
    return ("span" if duration else "point"), layer, None, duration


def convert_dezrann(path: str | Path, *, dataset: str, profile: str) -> SharedPiece:
    """Convert annotation-generic Dezrann while preserving overlap semantics."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations: list[SharedAnnotation] = []
    layer_definitions: dict[str, dict[str, Any]] = {}
    maximum = 0.0
    for index, label in enumerate(payload.get("labels", [])):
        if "start" not in label:
            continue
        start = float(label["start"])
        kind, layer, level, duration = _dez_kind_layer(label, profile)
        end = start + float(duration) if duration else None
        maximum = max(maximum, end if end is not None else start)
        source_type = _text(label.get("type")) or "untyped"
        tag = _text(label.get("tag")) or _text(label.get("comment")) or source_type
        tree_id = "form" if kind == "hierarchical_segment" else None
        annotations.append(SharedAnnotation(
            annotation_id=f"dez:{index}", kind=kind, layer=layer, label=tag,
            start_qb=start, end_qb=end, level=level, tree_id=tree_id,
            staff=_text(label.get("staff")) or None, source_type=source_type,
            source={key: value for key, value in label.items()
                    if key not in {"start", "duration", "ioi", "tag", "staff", "type"}},
        ))
        if layer not in layer_definitions:
            semantics = {
                "hierarchical_segment": "hierarchical_form",
                "partition_segment": "flat_partition",
                "span": "overlapping_span",
                "point": "point_event",
            }[kind]
            layer_definitions[layer] = {
                "semantics": semantics,
                "overlap": "nested" if kind == "hierarchical_segment" else (
                    "disjoint" if kind == "partition_segment" else "allowed"),
            }

    # Infer parent links only for explicitly numbered top.* form levels.
    hierarchy = [item for item in annotations if item.kind == "hierarchical_segment"]
    for child in hierarchy:
        candidates = [parent for parent in hierarchy
                      if parent.level is not None and child.level is not None
                      and parent.level < child.level
                      and parent.start_qb <= child.start_qb
                      and child.end_qb <= parent.end_qb]
        if candidates:
            parent = max(candidates, key=lambda item: (item.level, -(item.end_qb-item.start_qb)))
            child.parent_id = parent.annotation_id
    piece_id = path.name.removesuffix(".dez")
    warnings = [
        "Dezrann musical time is interpreted as quarter notes from piece start",
        "overlapping motif/theme spans are preserved and are not treated as a tree",
    ]
    if profile == "fugue":
        warnings.append("subject/countersubject repetitions are overlap-aware spans, not hierarchy nodes")
    piece = SharedPiece(
        dataset=dataset, piece_id=piece_id,
        work_id=_dez_work_id(piece_id, profile), coordinate_system="quarterbeat",
        duration_qb=maximum if maximum > 0 else None,
        annotation_path=str(path), annotations=annotations,
        layer_definitions=layer_definitions,
        provenance={"format": "Dezrann", "profile": profile,
                    "source_meta": payload.get("meta", {})}, warnings=warnings,
    )
    validate_piece(piece)
    return piece
