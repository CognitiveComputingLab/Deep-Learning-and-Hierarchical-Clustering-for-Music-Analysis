#!/usr/bin/env python
"""Audit frozen outputs and build a dissertation-ready three-stage comparison.

This script never trains, tunes, or selects a model.  It only reads completed
outputs, checks scope/leakage/DP-optimality invariants, and writes traceable
evidence tables.  Incomplete Quick results can be audited, but are never marked
ready for dissertation claims.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd


class EvidenceAuditError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path("results/dissertation_main"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-works", type=int, default=16)
    parser.add_argument("--primary-seed", type=int, default=20260818)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise EvidenceAuditError(f"missing required file: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise EvidenceAuditError(f"required CSV is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path):
    missing = columns - set(frame.columns)
    if missing:
        raise EvidenceAuditError(f"{source} is missing columns: {sorted(missing)}")


def _status_ok(directory: Path, expected_pieces: int | None = None) -> tuple[bool, list[str]]:
    path = directory / "run_status.csv"
    if not path.is_file():
        return False, [f"missing {path}"]
    frame = pd.read_csv(path)
    if expected_pieces is not None:
        if len(frame) != expected_pieces:
            return False, [f'expected {expected_pieces} status rows, found {len(frame)}']
        if 'piece' in frame and frame.piece.astype(str).nunique() != expected_pieces:
            return False, [f'expected {expected_pieces} unique pieces in status, found {frame.piece.astype(str).nunique()}']
    if "status" not in frame:
        return False, [f"{path} has no status column"]
    failures = frame[~frame.status.astype(str).str.lower().isin({"success", "skipped"})]
    failure_messages = [
        " | ".join(map(str, row))
        for row in failures.astype(str).itertuples(index=False, name=None)
    ]
    return failures.empty, failure_messages


def _metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _greedy_primary(directory: Path) -> tuple[pd.DataFrame, set[str]]:
    path = directory / "boundary_per_piece.csv"
    frame = _read_csv(path)
    _require_columns(frame, {
        "work", "method", "bin_size_qb", "tolerance_qb", "depth",
        "tp", "fp", "fn",
    }, path)
    selected = frame[
        (frame.method == "key_profile")
        & np.isclose(frame.bin_size_qb, 8.0)
        & np.isclose(frame.tolerance_qb, 8.0)
        & (frame.depth == 4)
    ]
    if selected.empty:
        raise EvidenceAuditError("Greedy primary configuration bin=8/tol=8/depth=4/key_profile is missing")
    works = selected.groupby("work", as_index=False)[["tp", "fp", "fn"]].sum()
    metrics = works.apply(lambda row: _metrics(row.tp, row.fp, row.fn), axis=1)
    works[["precision", "recall", "f1"]] = pd.DataFrame(metrics.tolist(), index=works.index)
    row = pd.DataFrame([{
        "stage": "basic", "model": "handcrafted_key_profile",
        "search": "greedy", "selection_mode": "fixed_depth_4",
        "n_works": works.work.nunique(),
        "precision": works.precision.mean(), "recall": works.recall.mean(),
        "f1": works.f1.mean(), "source": str(path),
    }])
    return row, set(works.work.astype(str))


def _dp_primary(directory: Path) -> tuple[pd.DataFrame, set[str], bool]:
    path = directory / "per_work_results.csv"
    frame = _read_csv(path)
    _require_columns(frame, {"work", "search", "method", "precision", "recall", "f1"}, path)
    selected = frame[(frame.method == "key_profile") & frame.search.isin(["greedy", "dp"])]
    if selected.empty:
        raise EvidenceAuditError("DP key_profile Greedy/DP comparison is missing")
    rows = selected.groupby("search", as_index=False).agg(
        n_works=("work", "nunique"), precision=("precision", "mean"),
        recall=("recall", "mean"), f1=("f1", "mean"))
    rows.insert(0, "model", "handcrafted_key_profile")
    rows.insert(0, "stage", "intermediate")
    rows["selection_mode"] = "fixed_depth_4"
    rows["source"] = str(path)

    objective_path = directory / "objective_comparison.csv"
    objective = _read_csv(objective_path)
    if "objective_gap" in objective:
        objective_ok = bool((pd.to_numeric(objective.objective_gap) >= -1e-8).all())
    elif {"dp_objective", "greedy_objective"} <= set(objective):
        objective_ok = bool((objective.dp_objective >= objective.greedy_objective - 1e-8).all())
    else:
        raise EvidenceAuditError(f"{objective_path} has no auditable objective columns")
    return rows, set(selected.work.astype(str)), objective_ok


def _audit_folds(frame: pd.DataFrame, expected_works: int) -> tuple[bool, set[str], list[str]]:
    required = {"outer_work", "work", "split"}
    if not required <= set(frame):
        return False, set(), [f"fold_assignments.csv missing {sorted(required - set(frame))}"]
    errors = []
    outer_works = set(frame.outer_work.astype(str))
    for outer in sorted(outer_works):
        rows = frame[frame.outer_work.astype(str) == outer]
        test = rows[rows.split == "outer_test"]
        if set(test.work.astype(str)) != {outer}:
            errors.append(f"{outer}: outer_test is not exactly the held-out work")
        leaked = rows[(rows.work.astype(str) == outer) & rows.split.isin(["outer_train", "inner_train", "inner_validation"])]
        if not leaked.empty:
            errors.append(f"{outer}: held-out work appears in training/validation")
    if len(outer_works) != expected_works:
        errors.append(f"expected {expected_works} outer works, found {len(outer_works)}")
    return not errors, outer_works, errors


def _neural_primary(directory: Path, expected_works: int, primary_seed: int):
    metrics_path = directory / "held_out_per_work.csv"
    frame = _read_csv(metrics_path)
    _require_columns(frame, {
        "work", "family", "search", "selection_mode", "training_seed",
        "tolerance_qb", "oracle", "precision", "recall", "f1",
    }, metrics_path)
    oracle = frame.oracle.astype(str).str.lower().isin({"true", "1"})
    selected = frame[
        frame.family.isin(["handcrafted_key_profile", "neural_circular_siamese"])
        & (frame.selection_mode == "threshold")
        & np.isclose(frame.tolerance_qb, 8.0)
        & (frame.training_seed == primary_seed)
        & ~oracle
    ]
    parametric_selected = frame[
        (frame.family == 'diagonal_mahalanobis')
        & (frame.selection_mode == 'threshold')
        & np.isclose(frame.tolerance_qb, 8.0)
        & (frame.training_seed == primary_seed)
        & ~oracle
    ]
    selected = pd.concat([selected, parametric_selected], ignore_index=True)
    if selected.empty:
        raise EvidenceAuditError("Neural primary shared-threshold rows are missing")
    rows = selected.groupby(["family", "search"], as_index=False).agg(
        n_works=("work", "nunique"), precision=("precision", "mean"),
        recall=("recall", "mean"), f1=("f1", "mean"))
    rows = rows[
        ((rows.family == "handcrafted_key_profile") & rows.search.isin(["greedy", "dp"]))
        | ((rows.family == "neural_circular_siamese") & rows.search.isin(["greedy", "dp"]))
        | ((rows.family == 'diagonal_mahalanobis') & rows.search.isin(['greedy', 'dp']))
    ].copy()
    rows["model"] = rows.family
    rows["stage"] = np.select(
        [
            (rows.family == "handcrafted_key_profile") & (rows.search == "greedy"),
            (rows.family == "handcrafted_key_profile") & (rows.search == "dp"),
        ],
        ["basic", "intermediate"],
        default="advanced",
    )
    rows["selection_mode"] = "nested_validation_threshold"
    rows["source"] = str(metrics_path)
    rows.loc[rows.family == 'diagonal_mahalanobis', 'stage'] = 'intermediate_parametric'
    rows = rows[[
        "stage", "model", "search", "selection_mode", "n_works",
        "precision", "recall", "f1", "source",
    ]]

    folds = _read_csv(directory / "fold_assignments.csv")
    fold_ok, outer_works, fold_errors = _audit_folds(folds, expected_works)
    objectives = _read_csv(directory / "objective_comparison.csv")
    _require_columns(objectives, {"dp", "greedy", "objective_gap"}, directory / "objective_comparison.csv")
    objective_ok = bool(
        (objectives.dp >= objectives.greedy - 1e-8).all()
        and (objectives.objective_gap >= -1e-8).all())

    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    leakage_claim_ok = metadata.get("test_used_for_selection") is False
    ground_truth_ok = "not hierarchical" in metadata.get("ground_truth_warning", "").lower()
    cell_counts = selected.groupby(["family", "search"]).work.nunique().to_dict()
    cells_complete = all(
        cell_counts.get((family, search), 0) == expected_works
        for family in ("handcrafted_key_profile", "neural_circular_siamese")
        for search in ("greedy", "dp")
    )
    if 'diagonal_mahalanobis' in set(selected.family):
        cells_complete = cells_complete and all(
            cell_counts.get(('diagonal_mahalanobis', search), 0) == expected_works
            for search in ('greedy', 'dp')
        )
    return rows, set(selected.work.astype(str)), {
        "fold_ok": fold_ok,
        "fold_errors": fold_errors,
        "outer_works": sorted(outer_works),
        "objective_ok": objective_ok,
        "metadata_test_used_for_selection_false": leakage_claim_ok,
        "ground_truth_warning_present": ground_truth_ok,
        "all_primary_comparison_cells_complete": cells_complete,
    }


def generate_pack(results_root: Path, output_dir: Path, expected_works: int,
                  primary_seed: int) -> dict:
    manifest_path = results_root / "data_manifest_summary.json"
    if not manifest_path.is_file():
        raise EvidenceAuditError(f"missing required file: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    greedy_dir, dp_dir, neural_dir = (
        results_root / "greedy", results_root / "dp", results_root / "neural")
    status = {}
    status_errors = {}
    for name, directory in (("greedy", greedy_dir), ("dp", dp_dir), ("neural", neural_dir)):
        status[name], status_errors[name] = _status_ok(directory)

    expected_pieces = int(manifest.get('included_pieces', 0))
    for name, directory in (('greedy', greedy_dir), ('dp', dp_dir), ('neural', neural_dir)):
        status_path = directory / 'run_status.csv'
        if status[name] and status_path.is_file():
            status_frame = pd.read_csv(status_path)
            if len(status_frame) != expected_pieces or status_frame.piece.astype(str).nunique() != expected_pieces:
                status[name] = False
                status_errors[name].append(f'expected {expected_pieces} successful piece rows')
            if not status_frame.status.astype(str).str.lower().eq('success').all():
                status[name] = False
                status_errors[name].append('one or more pieces were not successful')

    greedy, greedy_works = _greedy_primary(greedy_dir)
    dp, dp_works, dp_objective_ok = _dp_primary(dp_dir)
    neural, neural_works, neural_audit = _neural_primary(
        neural_dir, expected_works, primary_seed)
    # The main table must use one evaluator and one extraction protocol. The
    # historical standalone Greedy/DP rows use fixed depth and are exported
    # separately rather than mixed with validation-threshold results.
    comparison = neural.copy()
    legacy = pd.concat([greedy, dp], ignore_index=True)
    comparison["same_16_work_scope"] = comparison.n_works.eq(expected_works)

    common_scope = (
        greedy_works == dp_works == neural_works
        and len(neural_works) == expected_works)
    ready = bool(
        manifest.get("complete")
        and all(status.values())
        and common_scope
        and dp_objective_ok
        and neural_audit["objective_ok"]
        and neural_audit["fold_ok"]
        and neural_audit["metadata_test_used_for_selection_false"]
        and neural_audit["ground_truth_warning_present"]
        and neural_audit["all_primary_comparison_cells_complete"]
    )
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ready_for_dissertation": ready,
        "status": "complete_frozen_evidence" if ready else "incomplete_or_noncomparable_evidence",
        "expected_works": expected_works,
        "manifest": manifest,
        "stage_status_ok": status,
        "stage_status_errors": status_errors,
        "same_work_scope": common_scope,
        "work_sets": {
            "greedy": sorted(greedy_works), "dp": sorted(dp_works),
            "neural": sorted(neural_works),
        },
        "dp_objective_not_worse_than_greedy": dp_objective_ok,
        "neural_stage_audit": neural_audit,
        "reporting_scope": (
            "Boundary F1 evaluates DCML local-key boundaries; DCML is not a "
            "hierarchical ground-truth tree. DP optimality is limited to the "
            "fixed ordered leaves and frozen additive objective."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "three_stage_comparison.csv", index=False)
    legacy.to_csv(output_dir / "legacy_fixed_depth_diagnostics.csv", index=False)
    (output_dir / "final_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Frozen dissertation evidence", "",
        f"Status: **{audit['status']}**.", "",
        "This pack performs no training or hyperparameter selection.", "",
        comparison.to_markdown(index=False), "",
        "Boundary F1 is the primary metric. DCML local-key annotations are flat "
        "boundaries, not an expert hierarchical tree. Exact DP is globally optimal "
        "only under the stated frozen ordered additive objective.",
    ]
    (output_dir / "results_narrative.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return audit


def main():
    args = parse_args()
    output = args.output_dir or (args.results_root / "final_evidence")
    try:
        audit = generate_pack(
            args.results_root, output, args.expected_works, args.primary_seed)
    except EvidenceAuditError as error:
        if not args.allow_incomplete:
            raise SystemExit(str(error)) from error
        output.mkdir(parents=True, exist_ok=True)
        audit = {
            "ready_for_dissertation": False,
            "status": "incomplete_or_noncomparable_evidence",
            "error": str(error),
        }
        (output / "final_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if not audit.get("ready_for_dissertation") and not args.allow_incomplete:
        raise SystemExit("Evidence audit failed; see final_audit.json")


if __name__ == "__main__":
    main()
