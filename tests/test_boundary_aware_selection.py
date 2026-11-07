from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boundary_contrast import (
    BoundaryContrastScorer,
    BoundaryFeatureMatrix,
    ProbabilityCalibrator,
    RobustContrastScaler,
)
from scripts.evaluate_boundary_aware_stage import (
    ProbabilityCalibratedScorer,
    f_beta,
    fit_contrast_models,
    natural_calibration_examples,
    select_candidate,
)


def candidate_frame(searches=("greedy", "dp")):
    rows = []
    operating_points = {
        0.2: {"precision": 0.60, "recall": 0.90, "f1": 0.72},
        0.8: {"precision": 0.85, "recall": 0.55, "f1": 0.668},
    }
    for work in ("w1", "w2"):
        for search in searches:
            for threshold, metrics in operating_points.items():
                rows.append({
                    "work": work, "model": "supervised_contrast",
                    "search": search, "selection_mode": "threshold",
                    "contrast_weight": 0.5, "balance_weight": 0.1,
                    "candidate_value": threshold,
                    "min_separation_qb": 0.0, "local_max_radius_qb": 0.0,
                    **metrics,
                })
    frame = pd.DataFrame(rows)
    frame["f0.5"] = f_beta(frame.precision, frame.recall)
    return frame


def test_f_beta_and_f1_default_selection_are_distinct_and_stable():
    assert f_beta(0.85, 0.55) > f_beta(0.60, 0.90)
    frame = candidate_frame()
    f1_choice, _ = select_candidate(
        frame, "supervised_contrast", "threshold")
    precision_choice, _ = select_candidate(
        frame, "supervised_contrast", "threshold", selection_metric="f0.5")
    assert f1_choice["candidate_value"] == pytest.approx(0.2)
    assert precision_choice["candidate_value"] == pytest.approx(0.8)


def test_precision_recall_floor_is_relative_to_f1_choice_unless_explicit():
    frame = candidate_frame()
    constrained, candidates = select_candidate(
        frame, "supervised_contrast", "threshold",
        selection_metric="precision_at_recall_floor", max_recall_drop=0.05)
    assert constrained["candidate_value"] == pytest.approx(0.2)
    assert constrained["recall_floor"] == pytest.approx(0.85)
    assert candidates.loc[
        candidates.candidate_value.eq(0.8), "eligible_for_selection"].eq(False).all()

    relaxed, _ = select_candidate(
        frame, "supervised_contrast", "threshold",
        selection_metric="precision_at_recall_floor", recall_floor=0.50)
    assert relaxed["candidate_value"] == pytest.approx(0.8)


def test_search_specific_selection_can_hold_shared_tree_parameters_fixed():
    frame = candidate_frame(searches=("greedy",))
    alternative = frame.copy()
    alternative["contrast_weight"] = 0.75
    alternative["balance_weight"] = 0.6
    alternative.loc[alternative.candidate_value.eq(0.8), ["precision", "recall", "f1"]] = [
        0.99, 0.99, 0.99]
    combined = pd.concat([frame, alternative], ignore_index=True)
    chosen, candidates = select_candidate(
        combined, "supervised_contrast", "threshold", search="greedy",
        fixed_tree_parameters={"contrast_weight": 0.5, "balance_weight": 0.1})
    assert chosen["contrast_weight"] == pytest.approx(0.5)
    assert chosen["balance_weight"] == pytest.approx(0.1)
    assert set(candidates.contrast_weight) == {0.5}


def calibration_items():
    def item(work, piece, n_bins, reference):
        values = np.linspace(0.0, 1.0, n_bins - 1)[:, None]
        features = BoundaryFeatureMatrix(
            values, ("feature_w1",), (1,), ("feature",))
        return {
            "work": work, "piece": piece, "features": features,
            "bounds": np.arange(n_bins + 1, dtype=float),
            "reference": [reference],
        }
    return [item("w1", "w1_01", 5, 2.2), item("w2", "w2_01", 3, 1.1)]


def test_natural_calibration_is_work_balanced_and_excludes_ignore_band():
    items = calibration_items()
    scaler = RobustContrastScaler.fit([item["features"] for item in items])
    scorer = BoundaryContrastScorer.unsupervised(scaler)
    scores, labels, weights, audit = natural_calibration_examples(
        items, scorer, ignore_tolerance_qb=1.0)
    assert len(scores) == len(labels) == len(weights) == 4
    assert weights.sum() == pytest.approx(1.0)
    work_mass = {
        work: sum(row["calibration_sample_weight"] for row in audit
                  if row["work"] == work)
        for work in ("w1", "w2")
    }
    assert work_mass == pytest.approx({"w1": 0.5, "w2": 0.5})
    ignored = [row for row in audit if not row["calibration_eligible"]]
    assert {(row["piece"], row["split_index"]) for row in ignored} == {
        ("w1_01", 3), ("w2_01", 2)}
    assert all(row["calibration_sample_weight"] == 0 for row in ignored)


def test_probability_wrapper_transforms_scores_monotonically_and_delegates():
    items = calibration_items()
    scaler = RobustContrastScaler.fit([item["features"] for item in items])
    base = BoundaryContrastScorer.unsupervised(scaler)
    calibrator = ProbabilityCalibrator(
        intercept=-1.0, slope=0.5, score_location=0.0, score_scale=1.0)
    wrapped = ProbabilityCalibratedScorer(base, calibrator)
    raw = base.score(items[0]["features"])
    transformed = wrapped.score(items[0]["features"])
    assert np.allclose(transformed, calibrator.transform(raw))
    assert not np.allclose(transformed, raw)
    assert np.all(np.diff(transformed) >= 0)
    assert wrapped.scaler is base.scaler and wrapped.mode == base.mode
    assert wrapped.as_dict()["probability_calibration"]["slope"] == 0.5


def test_fit_contrast_models_none_preserves_unwrapped_legacy_scorer():
    args = SimpleNamespace(
        ignore_tolerance_qb=0.0, probability_calibration="none",
        epochs=2, learning_rate=0.05, regularization=1e-3)
    models, audit, _ = fit_contrast_models(calibration_items(), args)
    supervised = models["supervised_contrast"]
    assert isinstance(supervised, BoundaryContrastScorer)
    assert not isinstance(supervised, ProbabilityCalibratedScorer)
    assert {row["probability_calibration"] for row in audit} == {"none"}
    assert not any(row["calibration_eligible"] for row in audit)

    platt_args = SimpleNamespace(
        ignore_tolerance_qb=0.0, probability_calibration="platt",
        epochs=2, learning_rate=0.05, regularization=1e-3)
    calibrated_models, calibrated_audit, _ = fit_contrast_models(
        calibration_items(), platt_args)
    calibrated_scorer = calibrated_models["supervised_contrast"]
    assert isinstance(calibrated_scorer, ProbabilityCalibratedScorer)
    features = calibration_items()[0]["features"]
    assert not np.allclose(calibrated_scorer.score(features),
                           supervised.score(features))
    assert {row["probability_calibration"]
            for row in calibrated_audit} == {"platt"}
    assert any(row["calibration_eligible"] for row in calibrated_audit)


def test_fit_contrast_models_rejects_unknown_calibration_mode():
    args = SimpleNamespace(
        ignore_tolerance_qb=0.0, probability_calibration="plat",
        epochs=1, learning_rate=0.05, regularization=1e-3)
    with pytest.raises(ValueError, match="either 'platt' or 'none'"):
        fit_contrast_models(calibration_items(), args)


def test_work_f0_5_is_mean_of_piece_scores_not_score_of_mean_metrics():
    rows = []
    for precision, recall in ((1.0, 0.1), (0.1, 1.0)):
        rows.append({
            "work": "w1", "model": "supervised_contrast", "search": "dp",
            "selection_mode": "threshold", "contrast_weight": 0.5,
            "balance_weight": 0.1, "candidate_value": 0.4,
            "min_separation_qb": 0.0, "local_max_radius_qb": 0.0,
            "precision": precision, "recall": recall, "f1": 0.18,
            "f0.5": float(f_beta(precision, recall)),
        })
    _, candidates = select_candidate(
        pd.DataFrame(rows), "supervised_contrast", "threshold")
    expected = sum(row["f0.5"] for row in rows) / len(rows)
    assert candidates.iloc[0]["f0.5"] == pytest.approx(expected)
    assert candidates.iloc[0]["f0.5"] != pytest.approx(f_beta(0.55, 0.55))
