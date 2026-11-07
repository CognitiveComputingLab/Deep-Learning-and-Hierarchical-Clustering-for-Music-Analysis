from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from render_parametric_case_pitchscapes import paired_case_table, select_cases


def _rows():
    rows = []
    for piece, gains in {
        "strong": (0.30, 0.15, 0.20, 0.05),
        "weak_full": (0.40, 0.01, 0.30, 0.06),
        "negative_lca": (0.35, 0.20, -0.10, 0.07),
    }.items():
        reduced, full, lca, objective = gains
        base = {
            "source": "formal", "dataset": "test", "piece": piece,
            "work": piece, "lca_normalized_mae": 0.1,
        }
        rows.extend([
            {**base, "search": "greedy", "objective": 1.0,
             "t_measure_reduced": 0.2, "t_measure_full": 0.5,
             "lca_spearman": 0.4},
            {**base, "search": "dp", "objective": 1.0 + objective,
             "t_measure_reduced": 0.2 + reduced,
             "t_measure_full": 0.5 + full,
             "lca_spearman": 0.4 + lca},
        ])
    return pd.DataFrame(rows)


def test_case_selection_requires_consistent_substantial_dp_gain():
    paired = paired_case_table([_rows()])
    selected = select_cases(
        paired, top_k=1, minimum_full_gain=0.08,
        minimum_lca_gain=0.05)
    assert selected.piece.tolist() == ["strong"]


def test_case_selection_refuses_to_silently_return_too_few_cases():
    paired = paired_case_table([_rows()])
    with pytest.raises(RuntimeError, match="only 1 cases"):
        select_cases(
            paired, top_k=2, minimum_full_gain=0.08,
            minimum_lca_gain=0.05)
