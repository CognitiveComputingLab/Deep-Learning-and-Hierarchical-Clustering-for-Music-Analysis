from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from annotated_piece import (
    AnnotatedPiece, AnnotationBundle, BoundaryAnnotation, NoteEvent,
    notes_to_pc_bins, select_bin_size,
)
from corpus_loaders import read_fugue_dez
from data_manifest import build_data_manifest
from external_evaluation import holm_adjust, span_scores
from greedy_clustering import balanced_tree
from greedy_evaluation import (
    boundary_prominence_scores, collect_prominent_splits,
)
from taking_form_loader import load_taking_form_csv, read_taking_form_events
from train_deep_clustering import _internal_spans


def test_annotation_is_removed_from_inference_view():
    piece = AnnotatedPiece(
        "x", "work", "piece", (NoteEvent(60, 0, 4),), 4,
        AnnotationBundle((BoundaryAnnotation(2),), (), ()))
    view = piece.inference_view()
    assert not hasattr(view, "annotations")
    assert "annotations" not in inspect.signature(notes_to_pc_bins).parameters


def test_multiresolution_selects_finest_feasible_bin_size():
    assert select_bin_size(100, (8, 16, 32), 20) == 8
    assert select_bin_size(200, (8, 16, 32), 20) == 16
    assert select_bin_size(500, (8, 16, 32), 20) == 32
    with pytest.raises(ValueError):
        select_bin_size(1000, (8, 16, 32), 20)


def test_taking_form_comparator_copies_source_and_explicit_exception(tmp_path):
    path = tmp_path / "form.csv"
    path.write_text(
        "1,1,Exposition,Theme A\n"
        "3,1,,Transition\n"
        "5-8=1-4,1,Recapitulation,\n"
        "8,1,,Coda\n", encoding="utf-8")
    events, audit = read_taking_form_events(path)
    copied = {(event.measure, event.level, event.label)
              for event in events if event.provenance == "comparator_copy"}
    assert (5, 2, "Theme A") in copied
    assert (7, 2, "Transition") in copied
    assert any(event.measure == 5 and event.level == 1
               and event.label == "Recapitulation" for event in events)
    assert audit.comparator_rows == 1 and audit.copied_events == 3
    tree = load_taking_form_csv(path)
    assert tree.end_measure == 8 and tree.size() > 1


def test_taking_form_repeat_marker_is_audited_not_expanded(tmp_path):
    path = tmp_path / "repeat.csv"
    path.write_text("1,1,A\nRepeat: 1-4,,\n4,1,B\n", encoding="utf-8")
    events, audit = read_taking_form_events(path)
    assert audit.score_repeat_rows == 1
    assert len(events) == 2


def test_fugue_parser_separates_tree_and_overlapping_annotations(tmp_path):
    path = tmp_path / "f.dez"
    path.write_text(
        '{"labels":['
        '{"type":"S","start":1,"duration":2},'
        '{"type":"Cadence","start":8,"tag":"PAC"},'
        '{"type":"Pedal","start":6,"duration":3,"tag":"I"}]}',
        encoding="utf-8")
    parsed = read_fugue_dez(path)
    assert len(parsed["cadences"]) == 1
    assert len(parsed["pedals"]) == 1
    assert len(parsed["overlapping_motives"]) == 1


def test_canonical_prominence_drives_fixed_budget_selection():
    matrix = np.eye(12)[:5]
    bounds = [(float(i), float(i + 1)) for i in range(5)]
    tree = balanced_tree(matrix, bounds)
    scores = boundary_prominence_scores(tree, bounds)
    selected = collect_prominent_splits(tree, 2)
    edges = np.arange(1, 5, dtype=float)
    expected = sorted(edges[np.argsort(-scores, kind="stable")[:2]])
    assert selected == pytest.approx(expected)
    assert np.all((0 <= scores) & (scores <= 1))


def test_span_matching_is_ordered_one_to_one():
    score = span_scores([(0, 4), (4, 8)], [(0.2, 4.1), (4.2, 8)], 0.25)
    assert score["tp"] == 2 and score["f1"] == pytest.approx(1)
    duplicate = span_scores([(0, 4), (0.1, 4.1)], [(0, 4)], 0.25)
    assert duplicate["tp"] == 1 and duplicate["fp"] == 1


def test_holm_adjustment_is_monotonic_in_sorted_order():
    raw = np.asarray([0.03, 0.001, 0.2])
    adjusted = holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted >= raw)


@pytest.mark.skipif(not (ROOT / "external" / "ABC" / "notes").is_dir(),
                    reason="ABC corpus is not installed")
def test_manifest_never_silently_drops_a_dataset_piece():
    frame = build_data_manifest(ROOT)
    assert len(frame[frame.dataset == "abc"]) == 70
    assert len(frame[frame.dataset == "taking_form"]) == 103
    assert len(frame[frame.dataset == "algomus_fugue"]) >= 23
    assert frame.status.isin(["included", "excluded"]).all()
    assert frame[frame.status == "excluded"].reason.ne("").all()
