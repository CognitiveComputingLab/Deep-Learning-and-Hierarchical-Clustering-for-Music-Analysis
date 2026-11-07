"""Aggregate completed Advanced-stage shards with bounded memory.

This script is intentionally not part of the scientific implementation hash.
It validates the existing shard markers and hashes, streams large CSV tables,
and loads only held-out result tables for the final statistics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_boundary_aware_stage import f_beta, work_id
from evaluate_neural_distance_stage import (
    SHARD_TABLES,
    _atomic_json,
    configuration_fingerprint,
    dataset_hash,
    discover_pairs,
    implementation_hash,
    plot_outputs,
    validate_shard,
)
from greedy_evaluation import (
    bootstrap_summary,
    micro_summary,
    paired_permutation_tests,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-works", type=int, default=16)
    return parser.parse_args()


def read_markers(output_dir: Path, expected_works: int):
    markers = []
    for path in sorted((output_dir / "shards").glob("*/complete.json")):
        marker = json.loads(path.read_text(encoding="utf-8"))
        if marker.get("status") != "success":
            raise RuntimeError(f"Incomplete shard: {path.parent.name}")
        if marker.get("outer_work") != path.parent.name:
            raise RuntimeError(f"Shard name mismatch: {path}")
        markers.append(marker)
    works = [str(marker["outer_work"]) for marker in markers]
    if len(markers) != expected_works or len(set(works)) != expected_works:
        raise RuntimeError(
            f"Expected {expected_works} unique shards, found {len(set(works))}")
    for field in ("configuration_fingerprint", "data_sha256",
                  "implementation_sha256"):
        if len({marker.get(field) for marker in markers}) != 1:
            raise RuntimeError(f"Shards disagree on {field}")
    configs = {json.dumps(marker.get("configuration"), sort_keys=True)
               for marker in markers}
    if len(configs) != 1:
        raise RuntimeError("Shards were produced with different configurations")
    return markers


def configured_pairs(data_root: Path, configuration: dict):
    pairs = discover_pairs(data_root, configuration.get("piece"))
    if configuration.get("quick") and not configuration.get("piece"):
        grouped = {}
        for pair in pairs:
            grouped.setdefault(work_id(pair[0]), []).append(pair)
        pairs = [sorted(grouped[work])[0] for work in sorted(grouped)[:6]]
    return pairs


def validate_provenance(output_dir: Path, markers, pairs):
    first = markers[0]
    data_sha = dataset_hash(pairs)
    code_sha = implementation_hash()
    if data_sha != first["data_sha256"]:
        raise RuntimeError("ABC data differ from those used to create the shards")
    if code_sha != first["implementation_sha256"]:
        raise RuntimeError("Project code differs from that used to create the shards")
    fingerprint = configuration_fingerprint(
        first["configuration"], data_sha, code_sha)
    if fingerprint != first["configuration_fingerprint"]:
        raise RuntimeError("Shard fingerprint is internally inconsistent")
    for marker in markers:
        validate_shard(output_dir, marker["outer_work"], fingerprint)
    return fingerprint, data_sha, code_sha


def stream_table(output_dir: Path, markers, key: str, destination: Path):
    filename = SHARD_TABLES[key]
    temporary = destination.with_name(destination.name + ".tmp")
    columns, count = None, 0
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = None
        for marker in markers:
            work = str(marker["outer_work"])
            source = output_dir / "shards" / work / filename
            if source.stat().st_size <= 1:
                continue
            with source.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                if columns is None:
                    columns = list(reader.fieldnames)
                    writer = csv.DictWriter(
                        target, fieldnames=columns, lineterminator="\n")
                    writer.writeheader()
                elif list(reader.fieldnames) != columns:
                    raise RuntimeError(f"Column mismatch in {source}")
                for row in reader:
                    if str(row.get("outer_work")) != work:
                        raise RuntimeError(f"Cross-work row found in {source}")
                    writer.writerow(row)
                    count += 1
    temporary.replace(destination)
    expected = sum(int(marker["row_counts"].get(filename, 0))
                   for marker in markers)
    if count != expected:
        raise RuntimeError(
            f"{filename}: wrote {count} rows but markers specify {expected}")
    return count


def aggregate_statistics(output_dir: Path, configuration: dict):
    results = pd.read_csv(output_dir / "held_out_per_piece.csv")
    diagnostics = pd.read_csv(output_dir / "tree_diagnostics.csv")
    seed = int(configuration["seed"])
    tolerance = float(configuration["selection_tolerance"])
    group = ["family", "search", "selection_mode", "calibration_scope",
             "training_seed", "tolerance_qb", "oracle"]
    work_results = results.groupby(["work"] + group, as_index=False)[
        ["tp", "fp", "fn", "precision", "recall", "f1", "f0.5"]].mean()
    work_results.to_csv(output_dir / "held_out_per_work.csv", index=False)
    bootstrap_summary(work_results, group, samples=2000, seed=seed).to_csv(
        output_dir / "work_macro_summary.csv", index=False)
    bootstrap_summary(
        work_results, group, value="f0.5", samples=2000, seed=seed).to_csv(
            output_dir / "work_macro_f0_5_summary.csv", index=False)
    micro = micro_summary(results, group)
    micro["f0.5"] = f_beta(micro.precision, micro.recall)
    micro.to_csv(output_dir / "micro_summary.csv", index=False)

    comparison = diagnostics.pivot_table(
        index=["outer_work", "piece", "family", "selection_mode", "training_seed"],
        columns="search", values="objective", aggfunc="first").reset_index()
    if not {"greedy", "dp"} <= set(comparison):
        raise RuntimeError("Greedy or DP objective rows are missing")
    comparison["objective_gap"] = comparison.dp - comparison.greedy
    comparison["relative_objective_gap"] = (
        comparison.objective_gap / comparison.dp.abs().replace(0, np.nan))
    if (comparison.objective_gap < -1e-8).any():
        raise AssertionError("exact DP objective is below Greedy")
    comparison.to_csv(output_dir / "objective_comparison.csv", index=False)

    primary = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & work_results.training_seed.eq(seed)
        & np.isclose(work_results.tolerance_qb, tolerance)].copy()
    primary["method"] = primary.family + "_" + primary.search
    paired_permutation_tests(
        primary[["work", "method", "f1"]], samples=10000,
        seed=seed, unit_column="work").to_csv(
            output_dir / "primary_paired_tests.csv", index=False)
    stability = work_results[
        (~work_results.oracle) & work_results.selection_mode.eq("threshold")
        & np.isclose(work_results.tolerance_qb, tolerance)]
    stability.groupby(["work", "family", "search"], as_index=False).agg(
        seed_mean_f1=("f1", "mean"), seed_std_f1=("f1", "std"),
        seeds=("training_seed", "nunique")).to_csv(
            output_dir / "seed_stability.csv", index=False)
    plot_outputs(results, diagnostics, output_dir)
    return work_results


def main():
    args = parse_args()
    markers = read_markers(args.output_dir, args.expected_works)
    configuration = markers[0]["configuration"]
    pairs = configured_pairs(args.data_root, configuration)
    fingerprint, data_sha, code_sha = validate_provenance(
        args.output_dir, markers, pairs)
    print(f"Validated {len(markers)} complete shards", flush=True)

    for key, filename in SHARD_TABLES.items():
        output_name = "affinity_metadata.csv" if key == "learned_affinity_metadata" else filename
        count = stream_table(
            args.output_dir, markers, key, args.output_dir / output_name)
        print(f"[table] {output_name}: {count} rows", flush=True)

    folds = args.output_dir / "fold_assignments.csv"
    pd.read_csv(folds).drop_duplicates().to_csv(folds, index=False)
    work_results = aggregate_statistics(args.output_dir, configuration)
    works = sorted(str(marker["outer_work"]) for marker in markers)

    status_path = args.output_dir / "run_status.csv"
    if not status_path.is_file():
        raise RuntimeError(
            "run_status.csv is missing; run the original aggregate command until "
            "the 70/70 loading pass completes, then rerun this command")
    status = pd.read_csv(status_path)
    metadata = {
        "primary_metric": "work-level macro DCML local-key Boundary F1",
        "primary_configuration": {
            "bin_size_qb": configuration["bin_size"],
            "selection_tolerance_qb": configuration["selection_tolerance"],
            "selection_mode": "nested-validation shared threshold",
            "training_seed": configuration["seed"],
        },
        "scientific_configuration": configuration,
        "families": configuration["families"],
        "outer_works": works,
        "seed": configuration["seed"],
        "stability_seeds": configuration["stability_seeds"],
        "configuration_fingerprint": fingerprint,
        "data_sha256": data_sha,
        "implementation_sha256": code_sha,
        "complete_outer_work_shards": len(works),
        "expected_learned_checkpoints": sum(
            int(marker.get("checkpoint_count", 0)) for marker in markers),
        "aggregate_only": True,
        "aggregation_implementation": "streaming_low_memory_v1",
        "test_used_for_selection": False,
        "ground_truth_warning": (
            "DCML local-key annotations are flat boundaries, not hierarchical ground-truth trees"),
        "dp_optimality": (
            "exact only for fixed ordered leaves and the frozen additive objective"),
        "reinforcement_learning": "not implemented",
        "successful_movements": int(status.status.eq("success").sum()),
        "skipped_movements": int(status.status.eq("skipped").sum()),
        "failed_movements": int(status.status.eq("failed").sum()),
    }
    _atomic_json(metadata, args.output_dir / "metadata.json")
    _atomic_json({
        "phase": "complete",
        "configuration_fingerprint": fingerprint,
        "requested_outer_works": works,
        "completed_outer_works": works,
        "remaining_outer_works": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, args.output_dir / "run_progress.json")

    seed = int(configuration["seed"])
    tolerance = float(configuration["selection_tolerance"])
    summary = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & work_results.training_seed.eq(seed)
        & np.isclose(work_results.tolerance_qb, tolerance)].groupby(
            ["family", "search"], as_index=False).agg(
                works=("work", "nunique"), precision=("precision", "mean"),
                recall=("recall", "mean"), f0_5=("f0.5", "mean"),
                f1=("f1", "mean"), std_f1=("f1", "std"))
    print("\nHELD-OUT WORK SUMMARY")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nStreaming aggregation completed successfully.")


if __name__ == "__main__":
    main()
