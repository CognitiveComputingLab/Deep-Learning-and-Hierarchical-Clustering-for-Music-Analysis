from pathlib import Path
import json
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_paper_release import (
    ROOT as BUILDER_ROOT,
    advanced_status,
    validate_output_path,
)


def test_advanced_release_stays_pending_without_formal_evidence(tmp_path):
    status, detail = advanced_status(tmp_path)
    assert status == "pending"
    assert detail["coverage"]["exists"] is False
    assert any("final_audit.json is missing" in reason
               for reason in detail["reasons"])


def test_advanced_release_accepts_audited_sixteen_work_result(tmp_path):
    neural = tmp_path / "neural"
    evidence = tmp_path / "final_evidence"
    neural.mkdir()
    evidence.mkdir()
    pd.DataFrame([
        {"work": f"work_{index:02d}", "family": "neural_circular_siamese",
         "search": "dp", "f1": 0.5}
        for index in range(16)
    ]).to_csv(neural / "held_out_per_work.csv", index=False)
    (evidence / "final_audit.json").write_text(
        json.dumps({"ready_for_dissertation": True}), encoding="utf-8")

    status, detail = advanced_status(tmp_path)
    assert status == "complete"
    assert detail["coverage"]["works"] == 16
    assert detail["reasons"] == []


@pytest.mark.parametrize("unsafe", [
    BUILDER_ROOT,
    BUILDER_ROOT / "scripts",
    BUILDER_ROOT / "src",
    BUILDER_ROOT / "tests",
    BUILDER_ROOT / "results",
    BUILDER_ROOT.parent,
])
def test_release_refresh_rejects_unsafe_output_paths(unsafe):
    with pytest.raises(ValueError, match="unsafe path"):
        validate_output_path(unsafe)


def test_release_refresh_accepts_dedicated_project_subdirectory():
    path = BUILDER_ROOT / "paper_release_test"
    assert validate_output_path(path) == path.resolve()
