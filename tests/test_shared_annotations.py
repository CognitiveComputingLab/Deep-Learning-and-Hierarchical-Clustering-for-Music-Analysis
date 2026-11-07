from pathlib import Path
import json
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_converters import convert_dcml_piece, convert_dezrann, convert_taking_form
from shared_annotation_format import SharedPiece, validate_piece
from shared_evaluation import algorithm_readiness, boundary_reference, reference_forest


def test_dcml_conversion_keeps_flat_keys_pedals_and_points(tmp_path):
    notes = tmp_path / "w1_01.notes.tsv"
    harmonies = tmp_path / "w1_01.harmonies.tsv"
    pd.DataFrame([
        {"quarterbeats": 0, "duration_qb": 4, "midi": 60},
        {"quarterbeats": 4, "duration_qb": 4, "midi": 67},
        {"quarterbeats": 8, "duration_qb": 4, "midi": 62},
    ]).to_csv(notes, sep="\t", index=False)
    pd.DataFrame([
        {"quarterbeats": 0, "duration_qb": 2, "mc": 1, "localkey": "I",
         "pedal": "V", "cadence": "", "phraseend": ""},
        {"quarterbeats": 2, "duration_qb": 2, "mc": 1, "localkey": "I",
         "pedal": "V", "cadence": "PAC", "phraseend": ""},
        {"quarterbeats": 4, "duration_qb": 4, "mc": 2, "localkey": "V",
         "pedal": "", "cadence": "", "phraseend": "}"},
        {"quarterbeats": 8, "duration_qb": 4, "mc": 3, "localkey": "V",
         "pedal": "", "cadence": "", "phraseend": ""},
    ]).to_csv(harmonies, sep="\t", index=False)
    piece = convert_dcml_piece(notes, harmonies)
    assert not validate_piece(piece, raise_on_error=False)
    assert [item.kind for item in piece.annotations if item.layer == "local_key"] == [
        "partition_segment", "partition_segment"]
    assert boundary_reference(piece, layers=["local_key"]) == [4.0]
    pedal = [item for item in piece.annotations if item.layer == "pedal"]
    assert [(item.start_qb, item.end_qb) for item in pedal] == [(0.0, 4.0)]
    assert {item.layer for item in piece.annotations if item.kind == "point"} == {
        "cadence", "phrase_end"}
    assert algorithm_readiness(piece)["pitch_clustering_ready"] is True


def test_taking_form_conversion_builds_measure_beat_tree(tmp_path):
    path = tmp_path / "sonata1op1movt1.csv"
    path.write_text(
        "1,1,Exposition,P,Theme A\n"
        "5,1,,TR,\n"
        "9,1,Development,,\n"
        "13,1,Recapitulation,P',Theme A'\n"
        "16,1,,,\n", encoding="utf-8")
    piece = convert_taking_form(path)
    assert not validate_piece(piece, raise_on_error=False)
    roots = reference_forest(piece, "form")
    assert len(roots) == 1 and roots[0].label == "ROOT"
    assert [child.label for child in roots[0].children] == [
        "Exposition", "Development", "Recapitulation"]
    assert roots[0].children[0].children[0].label == "P"
    with pytest.raises(ValueError, match="align it to a symbolic score"):
        boundary_reference(piece, layers=["form"])
    readiness = algorithm_readiness(piece)
    assert readiness["hierarchical_evaluation_ready"] is True
    assert readiness["pitch_clustering_ready"] is False


def test_dezrann_preserves_overlap_and_infers_only_explicit_form_hierarchy(tmp_path):
    fugue_path = tmp_path / "01-bwv846-ref.dez"
    fugue_path.write_text(json.dumps({"labels": [
        {"type": "S", "start": 0.5, "duration": 6, "staff": "1"},
        {"type": "S", "start": 2.0, "duration": 6, "staff": "2"},
        {"type": "Cadence", "start": 10, "tag": "PAC"},
        {"type": "Pedal", "start": 8, "duration": 4, "tag": "V"},
    ]}), encoding="utf-8")
    fugue = convert_dezrann(fugue_path, dataset="Algomus_Fugue", profile="fugue")
    assert not validate_piece(fugue, raise_on_error=False)
    subjects = [item for item in fugue.annotations if item.layer == "s"]
    assert subjects[0].end_qb > subjects[1].start_qb
    assert not reference_forest(fugue, "form")

    mozart_path = tmp_path / "k155.1-ref.dez"
    mozart_path.write_text(json.dumps({"labels": [
        {"type": "Structure", "line": "top.1", "start": 0,
         "duration": 100, "tag": "EXPOSITION"},
        {"type": "Structure", "line": "top.2", "start": 0,
         "duration": 40, "tag": "P"},
        {"type": "Structure", "line": "top.3", "start": 4,
         "duration": 12, "tag": "Theme"},
        {"type": "Tonality", "line": "bot.1", "start": 0,
         "duration": 100, "tag": "D"},
    ]}), encoding="utf-8")
    mozart = convert_dezrann(
        mozart_path, dataset="Algomus_Mozart_Quartets", profile="mozart_sonata")
    forest = reference_forest(mozart, "form")
    assert forest[0].label == "EXPOSITION"
    assert forest[0].children[0].label == "P"
    assert forest[0].children[0].children[0].label == "Theme"
    assert algorithm_readiness(mozart)["boundary_evaluation_ready"] is True


def test_shared_json_round_trip(tmp_path):
    path = tmp_path / "simple.dez"
    path.write_text(json.dumps({"labels": [
        {"type": "Cadence", "start": 4, "tag": "HC"}
    ]}), encoding="utf-8")
    piece = convert_dezrann(path, dataset="F", profile="fugue")
    output = piece.write_json(tmp_path / "piece.json")
    restored = SharedPiece.read_json(output)
    assert restored.to_dict() == piece.to_dict()


def test_taking_form_repairs_are_explicit_and_markers_survive(tmp_path):
    path = tmp_path / "sonata_typo_movt1.csv"
    path.write_text(
        "192,1,Development,\n"
        "191,1,,Transition\n"
        "199,1,Recapitulation,\n"
        "232,X (Cadenza),,Cadenza theme\n",
        encoding="utf-8")
    piece = convert_taking_form(path)
    assert not validate_piece(piece, raise_on_error=False)
    repaired = [item for item in piece.annotations
                if "coordinate_repair" in item.source]
    assert repaired
    assert repaired[0].source["coordinate_repair"] == {
        "reason": "non_monotonic_source_measure",
        "original_measure": 191,
        "repaired_measure": 193,
    }
    markers = [item for item in piece.annotations
               if item.layer == "source_marker"]
    assert len(markers) == 1
    assert markers[0].label == "X (Cadenza)"
    assert markers[0].kind == "point"
    assert any("non-monotonic" in warning for warning in piece.warnings)
    assert any("non-numeric beat" in warning for warning in piece.warnings)
