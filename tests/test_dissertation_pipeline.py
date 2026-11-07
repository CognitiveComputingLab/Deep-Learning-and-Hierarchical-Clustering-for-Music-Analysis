from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_dissertation_results import generate_pack
from data_manifest import build_abc_manifest, manifest_summary
from run_dissertation_experiments import _commands


def test_manifest_records_complete_and_incomplete_pairs(tmp_path):
    notes = tmp_path / "notes"
    harmonies = tmp_path / "harmonies"
    notes.mkdir(); harmonies.mkdir()
    (notes / "n01op1_01.notes.tsv").write_text("onset\tduration\tpitch\n", encoding="utf-8")
    (harmonies / "n01op1_01.harmonies.tsv").write_text("onset\tlocalkey\n", encoding="utf-8")
    (notes / "n02op2_01.notes.tsv").write_text("onset\tduration\tpitch\n", encoding="utf-8")

    frame = build_abc_manifest(tmp_path)
    assert list(frame.piece) == ["n01op1_01", "n02op2_01"]
    assert frame.loc[frame.piece == "n01op1_01", "status"].item() == "included"
    assert frame.loc[frame.piece == "n02op2_01", "status"].item() == "excluded"
    assert len(frame.loc[0, "notes_sha256"]) == 64
    assert manifest_summary(frame) == {
        "discovered_pieces": 2, "included_pieces": 1,
        "excluded_pieces": 1, "included_works": 1,
    }


def _write_status(directory):
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"piece": "p", "status": "success", "message": ""}]).to_csv(
        directory / "run_status.csv", index=False)


    status_path = directory / 'run_status.csv'
    status = pd.read_csv(status_path)
    status = pd.concat([status, pd.DataFrame([{'piece': 'p2', 'status': 'success', 'message': ''}])], ignore_index=True)
    status.to_csv(status_path, index=False)


def _fake_evidence(root: Path, objective_gap: float = 0.1):
    root.mkdir()
    (root / "data_manifest_summary.json").write_text(json.dumps({
        "complete": True, "included_pieces": 2, "included_works": 2,
    }), encoding="utf-8")
    greedy, dp, neural = root / "greedy", root / "dp", root / "neural"
    for directory in (greedy, dp, neural):
        _write_status(directory)

    pd.DataFrame([
        {"work": work, "method": "key_profile", "bin_size_qb": 8,
         "tolerance_qb": 8, "depth": 4, "tp": 2, "fp": 1, "fn": 1}
        for work in ("w1", "w2")
    ]).to_csv(greedy / "boundary_per_piece.csv", index=False)

    pd.DataFrame([
        {"work": work, "search": search, "method": "key_profile",
         "precision": 0.7, "recall": 0.6, "f1": 0.64}
        for work in ("w1", "w2") for search in ("greedy", "dp")
    ]).to_csv(dp / "per_work_results.csv", index=False)
    pd.DataFrame([{"objective_gap": objective_gap}]).to_csv(
        dp / "objective_comparison.csv", index=False)

    pd.DataFrame([
        {"work": work, "family": family, "search": search,
         "selection_mode": "threshold", "training_seed": 20260818,
         "tolerance_qb": 8, "oracle": False, "precision": 0.75,
         "recall": 0.65, "f1": 0.696}
        for work in ("w1", "w2")
        for family in ("handcrafted_key_profile", "neural_circular_siamese")
        for search in ("greedy", "dp")
    ]).to_csv(neural / "held_out_per_work.csv", index=False)
    pd.DataFrame([
        {"outer_work": outer, "work": work,
         "split": "outer_test" if outer == work else "outer_train"}
        for outer in ("w1", "w2") for work in ("w1", "w2")
    ]).to_csv(neural / "fold_assignments.csv", index=False)
    pd.DataFrame([{"dp": 1.0, "greedy": 0.9, "objective_gap": 0.1}]).to_csv(
        neural / "objective_comparison.csv", index=False)
    (neural / "metadata.json").write_text(json.dumps({
        "test_used_for_selection": False,
        "ground_truth_warning": "DCML boundaries are not hierarchical ground truth",
    }), encoding="utf-8")


def test_evidence_pack_requires_common_leakage_free_scope(tmp_path):
    root = tmp_path / "results"
    _fake_evidence(root)
    audit = generate_pack(root, root / "final_evidence", 2, 20260818)
    assert audit["ready_for_dissertation"] is True
    comparison = pd.read_csv(root / "final_evidence" / "three_stage_comparison.csv")
    assert set(comparison.stage) == {"basic", "intermediate", "advanced"}
    assert {"greedy", "dp"} <= set(comparison.search)


def test_evidence_pack_rejects_dp_objective_violation(tmp_path):
    root = tmp_path / "results"
    _fake_evidence(root, objective_gap=-0.2)
    audit = generate_pack(root, root / "final_evidence", 2, 20260818)
    assert audit["ready_for_dissertation"] is False
    assert audit["dp_objective_not_worse_than_greedy"] is False


def test_dissertation_resume_is_forwarded_to_neural_outer_shards(tmp_path):
    args = SimpleNamespace(
        output_dir=tmp_path, device="cuda", quick=False, resume=True)
    neural = _commands(args)["neural"]
    assert "--resume" in neural
    assert neural[neural.index("--device") + 1] == "cuda"
