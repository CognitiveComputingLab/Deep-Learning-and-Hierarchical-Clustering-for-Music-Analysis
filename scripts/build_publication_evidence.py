#!/usr/bin/env python
"""Build a small, auditable snapshot from completed publication experiments.

This script reads frozen outputs only. It does not train, tune, or recompute
model predictions. Run it from the repository root with ``--output`` to save
the JSON used by the publication-preparation notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty result table: {path}")
    return frame


def one(frame: pd.DataFrame, **filters) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build(root: Path) -> dict:
    sources: set[Path] = set()

    def csv(relative: str) -> pd.DataFrame:
        path = root / relative
        sources.add(path)
        return read_csv(path)

    def json_file(relative: str) -> dict:
        path = root / relative
        sources.add(path)
        return json.loads(path.read_text(encoding="utf-8"))

    audit = json_file("results/dissertation_main/final_evidence/final_audit.json")
    if not audit.get("ready_for_dissertation"):
        raise ValueError("ABC final audit is not complete")
    if audit["manifest"]["included_pieces"] != 70 or audit["expected_works"] != 16:
        raise ValueError("ABC scope differs from the frozen 70-movement, 16-work run")
    abc_rows = csv("results/dissertation_main/final_evidence/three_stage_comparison.csv")
    if len(abc_rows) != 6 or not abc_rows.n_works.eq(16).all():
        raise ValueError("ABC primary comparison must have six complete 16-work cells")
    abc = []
    for row in abc_rows.itertuples(index=False):
        abc.append({
            "family": row.model, "search": row.search,
            "precision": float(row.precision), "recall": float(row.recall),
            "f1": float(row.f1), "n_works": int(row.n_works),
        })
    neural_metadata = json_file("results/dissertation_main/neural/metadata.json")
    if (neural_metadata["primary_configuration"]["training_seed"] != 20260818
            or neural_metadata["successful_movements"] != 70):
        raise ValueError("ABC neural metadata does not match the primary run")
    paired_abc = one(
        csv("results/dissertation_main/neural/primary_paired_tests.csv"),
        method_a="handcrafted_key_profile_dp",
        method_b="neural_circular_siamese_dp",
    )

    held_work = csv("results/dissertation_main/neural/held_out_per_work.csv")
    seeds = []
    neural = held_work[
        held_work.family.eq("neural_circular_siamese")
        & held_work.selection_mode.eq("threshold")
        & np.isclose(held_work.tolerance_qb, 8.0)
        & ~held_work.oracle.astype(str).str.lower().isin(("true", "1"))
    ]
    for (seed, search), group in neural.groupby(["training_seed", "search"]):
        if group.work.nunique() != 16 or len(group) != 16:
            raise ValueError(f"incomplete neural work cells for {seed}/{search}")
        seeds.append({"seed": int(seed), "search": search,
                      "n_works": 16, "work_macro_f1": float(group.f1.mean())})
    if len(seeds) != 6:
        raise ValueError("expected three neural seeds and two searches")

    aligned = json_file("data/shared_annotations_v1_aligned/summary.json")
    if (aligned["requested"], aligned["aligned"]) != (120, 89):
        raise ValueError("Taking Form alignment scope changed")

    fixed = []
    for composer, expected_pieces, expected_works in (
        ("mozart", 17, 17), ("beethoven", 72, 27)
    ):
        prefix = f"results/shared_hierarchy_stage_{composer}_tree"
        summary = csv(f"{prefix}/tree_measure_summary.csv")
        bootstrap = csv(f"{prefix}/tree_measure_bootstrap.csv")
        pieces = csv(f"{prefix}/tree_measure_per_piece.csv")
        status = csv(f"{prefix}/run_status.csv")
        metadata = json_file(f"{prefix}/metadata.json")
        if (len(status) != expected_pieces
                or not status.status.astype(str).str.lower().eq("success").all()
                or metadata["configuration"]["tree_window_qb"] is not None):
            raise ValueError(f"Taking Form {composer} run is incomplete or uses a time window")
        levels = csv(f"{prefix}/equal_budget_summary.csv")
        paired_fixed = one(
            csv(f"{prefix}/dp_vs_greedy_tree_tests.csv"),
            method="key_profile", metric="t_measure_full",
        )
        subset = pieces[pieces.method.eq("key_profile")]
        if subset.piece.nunique() != expected_pieces or subset.work.nunique() != expected_works:
            raise ValueError(f"incomplete Taking Form {composer} fixed-representation scope")
        pivot = subset.pivot(index="piece", columns="search",
                             values=["objective_cost", "t_measure_full"])
        cost_gap = pivot["objective_cost"]["dp"] - pivot["objective_cost"]["greedy"]
        t_gap = pivot["t_measure_full"]["dp"] - pivot["t_measure_full"]["greedy"]
        if (cost_gap > 1e-8).any():
            raise ValueError(f"DP has worse objective cost for {composer}")
        for search in ("greedy", "dp"):
            row = one(summary, method="key_profile", search=search)
            metrics = {}
            for metric in ("t_measure_full", "t_measure_reduced"):
                ci = one(bootstrap, method="key_profile", search=search, metric=metric)
                if not np.isclose(float(row[metric]), float(ci["mean"])):
                    raise ValueError(f"summary/bootstrap mismatch: {composer}/{search}/{metric}")
                if int(ci.n_pieces) != expected_works:
                    raise ValueError(f"bootstrap is not work-blocked: {composer}/{search}")
                metrics[metric] = {
                    "mean": float(row[metric]),
                    "ci95_low": float(ci.ci95_low),
                    "ci95_high": float(ci.ci95_high),
                }
            fixed.append({
                "test_corpus": composer, "representation": "key_profile",
                "objective": "additive merge-distance cost; lower is better",
                "search": search, "n_pieces": expected_pieces,
                "n_works": expected_works, **metrics,
                "dp_cost_better_or_equal_pieces": int((cost_gap <= 1e-8).sum()),
                "dp_full_t_worse_pieces": int((t_gap < 0).sum()),
                "paired_dp_minus_greedy_full_t": float(
                    paired_fixed.mean_dp_minus_greedy),
                "paired_p_holm": float(paired_fixed.p_holm),
                "equal_budget_f1": {
                    str(level): float(one(
                        levels, method="key_profile", search=search,
                        reference_max_level=level).f1)
                    for level in (1, 3)
                },
            })

    transferred = []
    for train, test, expected_pieces, expected_works in (
        ("beethoven", "mozart", 17, 17),
        ("mozart", "beethoven", 72, 27),
    ):
        prefix = f"results/parametric_hierarchy_{train}_to_{test}"
        parameters = json_file(f"{prefix}/model_parameters.json")
        if parameters["test_labels_used_during_training_or_selection"] is not False:
            raise ValueError(f"held-out Taking Form labels used for selection: {prefix}")
        summary = csv(f"{prefix}/held_out_summary.csv")
        pieces = csv(f"{prefix}/held_out_per_piece.csv")
        status = csv(f"{prefix}/run_status.csv")
        objectives = csv(f"{prefix}/objective_comparison.csv")
        paired = one(csv(f"{prefix}/paired_dp_vs_greedy.csv"))
        if pieces.piece.nunique() != expected_pieces or pieces.work.nunique() != expected_works:
            raise ValueError(f"incomplete cross-composer scope: {prefix}")
        if (len(status) != aligned["aligned"]
                or not status.status.astype(str).str.lower().eq("success").all()
                or objectives.piece.nunique() != expected_pieces
                or (objectives.dp_minus_greedy < -1e-8).any()):
            raise ValueError(f"cross-composer objective or run status failed: {prefix}")
        if int(paired.n_works) != expected_works:
            raise ValueError(f"incomplete paired test: {prefix}")
        for search in ("greedy", "dp"):
            row = one(summary, search=search)
            if int(row.works) != expected_works:
                raise ValueError(f"incomplete summary: {prefix}/{search}")
            transferred.append({
                "train_corpus": train, "test_corpus": test,
                "representation": "form_supervised_parametric_mixture",
                "objective": "boundary-aware tree objective; higher is better",
                "search": search, "n_pieces": expected_pieces,
                "n_works": expected_works,
                "t_measure_full": float(row.t_measure_full),
                "t_measure_reduced": float(row.t_measure_reduced),
                "objective_value": float(row.objective),
                "dp_objective_better_or_equal_pieces": int(
                    (objectives.dp_minus_greedy >= -1e-8).sum()),
                "paired_dp_minus_greedy_full_t": float(paired.mean_dp_minus_greedy),
                "paired_p_value_unadjusted": float(paired.p_value),
            })

    try:
        # Documentation-only commits do not change the audited method code.
        revision = subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--",
             "src/ordered_affinity.py", "src/neural_distance.py",
             "src/greedy_evaluation.py", "src/hierarchical_evaluation.py",
             "scripts/evaluate_neural_distance_stage.py"],
            cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "code_revision_at_snapshot": revision,
        "abc": {"n_movements": 70, "n_works": 16,
                "primary_seed": 20260818, "primary": abc, "neural_seeds": seeds,
                "neural_dp_minus_key_profile_dp_f1": float(
                    -paired_abc.mean_f1_difference),
                "paired_p_unadjusted": float(paired_abc.p_value),
                "paired_p_holm": float(paired_abc.p_holm)},
        "taking_form_alignment": aligned,
        "taking_form_fixed_key_profile": fixed,
        "taking_form_cross_composer": transferred,
        "sources": [
            {"path": str(path.relative_to(root)).replace("\\", "/"),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(sources)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(build(args.root.resolve()), indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
