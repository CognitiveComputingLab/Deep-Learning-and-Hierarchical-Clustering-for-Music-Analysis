from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_converters import convert_taking_form
from score_alignment import (
    MeasureBeatAligner,
    align_piece_to_dcml_score,
    beat_unit_qb,
    taking_form_score_key,
)
from shared_annotation_format import validate_piece
from shared_evaluation import boundary_reference


def _write_score(tmp_path):
    measures = tmp_path / "01-1.measures.tsv"
    notes = tmp_path / "01-1.notes.tsv"
    pd.DataFrame([
        {"mc": 1, "mn": 0, "quarterbeats": 0, "duration_qb": 1,
         "timesig": "2/2"},
        {"mc": 2, "mn": 1, "quarterbeats": 1, "duration_qb": 4,
         "timesig": "2/2"},
        {"mc": 3, "mn": 2, "quarterbeats": 5, "duration_qb": 4,
         "timesig": "2/2"},
        {"mc": 4, "mn": 3, "quarterbeats": 9, "duration_qb": 4,
         "timesig": "2/2"},
    ]).to_csv(measures, sep="\t", index=False)
    pd.DataFrame([
        {"quarterbeats": 0, "duration_qb": 1, "midi": 60},
        {"quarterbeats": 1, "duration_qb": 4, "midi": 64},
        {"quarterbeats": 5, "duration_qb": 4, "midi": 67},
        {"quarterbeats": 9, "duration_qb": 4, "midi": 60},
    ]).to_csv(notes, sep="\t", index=False)
    return measures, notes


def test_score_key_mapping():
    assert taking_form_score_key(
        "sonata14op27no2(Moonlight)movt3", "Taking_Form_Beethoven") == "14-3"
    assert taking_form_score_key("K279-mvt1", "Taking_Form_Mozart") == "K279-1"


def test_beat_units_and_negative_pickup(tmp_path):
    measures, _ = _write_score(tmp_path)
    aligner = MeasureBeatAligner(measures, 13)
    assert beat_unit_qb("2/2") == 2
    assert beat_unit_qb("6/8") == 1.5
    assert aligner.position(1, -0.5) == 0
    assert aligner.position(1, 1) == 1
    assert aligner.position(1, 2) == 3
    with pytest.raises(ValueError, match="outside score"):
        aligner.position(2, -10)


def test_taking_form_alignment_produces_quarterbeat_reference(tmp_path):
    measures, notes = _write_score(tmp_path)
    source = tmp_path / "sonata1op2no1movt1.csv"
    source.write_text(
        "1,-0.5,Exposition,P\n"
        "2,1,Development,\n"
        "3,1,Recapitulation,P'\n",
        encoding="utf-8")
    piece = convert_taking_form(source, dataset="Taking_Form_Beethoven")
    aligned, diagnostics = align_piece_to_dcml_score(
        piece, measures, notes, score_piece="01-1")
    assert not validate_piece(aligned, raise_on_error=False)
    assert aligned.coordinate_system == "mixed"
    assert aligned.duration_qb == 13
    assert diagnostics.negative_beat_positions >= 1
    assert boundary_reference(aligned, layers=["form"], max_level=1) == [5.0, 9.0]
