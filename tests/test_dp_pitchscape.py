'''Tests for DP Pitch Scapes tree geometry and validation.'''

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from dp_pitchscape import collect_tree_geometry, render_dp_pitchscape
from greedy_clustering import balanced_tree


def test_collect_geometry_honours_inclusive_depth():
    matrix=np.eye(4,12)
    bounds=[(0,1),(1,2),(2,3),(3,4)]
    tree=balanced_tree(matrix,bounds)
    root_only,edges=collect_tree_geometry(tree,0)
    assert len(root_only)==1 and edges==[]
    points,edges=collect_tree_geometry(tree,1)
    assert len(points)==3 and len(edges)==2
    assert all(point[2]<=1 for point in points)


def test_collect_geometry_rejects_negative_depth():
    matrix=np.eye(2,12)
    tree=balanced_tree(matrix,[(0,1),(1,2)])
    with pytest.raises(ValueError,match='non-negative'):
        collect_tree_geometry(tree,-1)


def test_renderer_reports_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError,match='MIDI'):
        render_dp_pitchscape(
            method='key_profile',midi_path=tmp_path/'missing.mid',
            notes_path=tmp_path/'missing.tsv',output_path=tmp_path/'out.png')
