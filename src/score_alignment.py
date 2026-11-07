"""Auditable alignment from measure/beat annotations to DCML quarterbeats."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
from fractions import Fraction
from pathlib import Path
import re
from typing import Any

import pandas as pd

from shared_annotation_format import SharedPiece, validate_piece


@dataclass
class AlignmentDiagnostics:
    score_piece: str
    aligned_positions: int
    negative_beat_positions: int
    pickup_snapped_positions: int
    minimum_qb: float
    maximum_qb: float
    duration_qb: float
    measure_number_source: str = "mn"
    beat_convention: str = "positive=(beat-1)*unit; negative=beat*unit"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def taking_form_score_key(piece_id: str, dataset: str) -> str:
    """Map documented Taking Form identifiers to official DCML score IDs."""
    if dataset == "Taking_Form_Beethoven":
        match = re.search(r"sonata(\d+).*?(?:movt|mvt)(\d+)", piece_id, re.I)
        if not match:
            raise ValueError(f"Unrecognised Beethoven Taking Form ID: {piece_id}")
        return f"{int(match.group(1)):02d}-{int(match.group(2))}"
    if dataset == "Taking_Form_Mozart":
        match = re.search(r"k(\d+).*?(?:movt|mvt)(\d+)", piece_id, re.I)
        if not match:
            raise ValueError(f"Unrecognised Mozart Taking Form ID: {piece_id}")
        return f"K{int(match.group(1))}-{int(match.group(2))}"
    raise ValueError(f"No Taking Form score-key rule for {dataset!r}")


def _float(value: Any) -> float:
    return float(Fraction(str(value)))


def _measure_number(value: Any) -> int | None:
    match = re.match(r"^\s*(-?\d+)", str(value))
    return int(match.group(1)) if match else None


def beat_unit_qb(timesig: str) -> float:
    """Return notated beat length in quarterbeats.

    Simple metres use the denominator unit. Compound metres (6/8, 9/8,
    12/8, etc.) use a dotted beat, matching music21's beat convention used by
    Taking Form's MusicXML extractor.
    """
    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(timesig))
    if not match:
        raise ValueError(f"Unsupported time signature: {timesig!r}")
    numerator, denominator = map(int, match.groups())
    denominator_unit = 4.0 / denominator
    return 3.0 * denominator_unit if numerator > 3 and numerator % 3 == 0 else denominator_unit


class MeasureBeatAligner:
    """Map printed measure numbers and notated beats to absolute quarterbeats."""

    def __init__(self, measures_path: str | Path, duration_qb: float):
        self.measures_path = Path(measures_path)
        self.duration_qb = float(duration_qb)
        self.pickup_snaps = 0
        frame = pd.read_csv(self.measures_path, sep="\t", dtype=str)
        required = {"mn", "quarterbeats", "duration_qb", "timesig"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{self.measures_path}: missing columns {sorted(missing)}")
        frame["timesig"] = frame["timesig"].ffill()
        self.rows: dict[int, dict[str, Any]] = {}
        ambiguous: set[int] = set()
        for row_index, row in frame.iterrows():
            number = _measure_number(row["mn"])
            if number is None or pd.isna(row["quarterbeats"]):
                continue
            record = {
                "row": int(row_index),
                "mn_raw": str(row["mn"]),
                "start_qb": _float(row["quarterbeats"]),
                "duration_qb": _float(row["duration_qb"]),
                "timesig": str(row["timesig"]),
            }
            if number in self.rows:
                ambiguous.add(number)
            else:
                self.rows[number] = record
        for number in ambiguous:
            self.rows.pop(number, None)
        self.ambiguous_measures = sorted(ambiguous)
        if not self.rows:
            raise ValueError(f"{self.measures_path}: no unique printed measures")

    def position(self, measure: int, beat: float, *, allow_piece_end: bool = False) -> float:
        measure = int(measure)
        beat = float(beat)
        if allow_piece_end and measure > max(self.rows):
            return self.duration_qb
        if measure not in self.rows:
            reason = "ambiguous" if measure in self.ambiguous_measures else "missing"
            raise ValueError(f"printed measure {measure} is {reason} in {self.measures_path.name}")
        row = self.rows[measure]
        unit = beat_unit_qb(row["timesig"])
        offset = beat * unit if beat < 0 else (beat - 1.0) * unit
        qb = row["start_qb"] + offset
        tolerance = 0.02
        if qb < -tolerance and beat < 0 and measure == 1 and row["start_qb"] > 0:
            # A coarse negative opening beat can overshoot the exact pickup.
            # The score start is the only valid position before bar 1.
            self.pickup_snaps += 1
            return 0.0
        if qb < -tolerance or qb > self.duration_qb + tolerance:
            raise ValueError(
                f"measure={measure}, beat={beat} maps outside score: qb={qb:g}")
        return min(self.duration_qb, max(0.0, qb))


def score_duration(notes_path: str | Path) -> float:
    frame = pd.read_csv(notes_path, sep="\t")
    required = {"quarterbeats", "duration_qb"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{notes_path}: missing columns {sorted(missing)}")
    valid = frame["quarterbeats"].notna() & frame["duration_qb"].notna()
    if not valid.any():
        raise ValueError(f"{notes_path}: no timed notes")
    starts = frame.loc[valid, "quarterbeats"].map(_float)
    durations = frame.loc[valid, "duration_qb"].map(_float)
    return float((starts + durations).max())


def align_piece_to_dcml_score(
    piece: SharedPiece,
    measures_path: str | Path,
    notes_path: str | Path,
    *,
    score_piece: str,
) -> tuple[SharedPiece, AlignmentDiagnostics]:
    """Return a mixed-coordinate copy aligned to one exact DCML score."""
    if piece.coordinate_system != "measure_beat":
        raise ValueError(f"{piece.piece_id}: expected measure_beat coordinates")
    duration = score_duration(notes_path)
    aligner = MeasureBeatAligner(measures_path, duration)
    aligned = deepcopy(piece)
    positions: list[float] = []
    negative = 0
    for item in aligned.annotations:
        if item.start_measure is None or item.start_beat is None:
            raise ValueError(f"{item.annotation_id}: missing start measure/beat")
        item.start_qb = aligner.position(item.start_measure, item.start_beat)
        positions.append(item.start_qb)
        negative += int(item.start_beat < 0)
        if item.kind != "point":
            if item.end_measure is None or item.end_beat is None:
                raise ValueError(f"{item.annotation_id}: missing end measure/beat")
            item.end_qb = aligner.position(
                item.end_measure,
                item.end_beat,
                allow_piece_end=item.end_measure == aligned.final_measure,
            )
            positions.append(item.end_qb)
    aligned.coordinate_system = "mixed"
    aligned.duration_qb = duration
    aligned.score_path = str(Path(notes_path))
    aligned.provenance = {
        **aligned.provenance,
        "alignment": {
            "score_piece": score_piece,
            "measures_path": str(Path(measures_path)),
            "notes_path": str(Path(notes_path)),
            "measure_number_source": "mn",
            "beat_convention": "positive=(beat-1)*unit; negative=beat*unit",
            "compound_meter": "dotted beat",
            "ambiguous_printed_measures": aligner.ambiguous_measures,
        },
    }
    aligned.warnings = [
        warning for warning in aligned.warnings
        if "matching symbolic score" not in warning
    ]
    aligned.warnings.append(
        "quarterbeat coordinates are aligned to the named DCML score edition; repeat-expansion compatibility must be evaluated separately")
    validate_piece(aligned)
    diagnostics = AlignmentDiagnostics(
        score_piece=score_piece,
        aligned_positions=len(positions),
        negative_beat_positions=negative,
        pickup_snapped_positions=aligner.pickup_snaps,
        minimum_qb=min(positions),
        maximum_qb=max(positions),
        duration_qb=duration,
    )
    return aligned, diagnostics
