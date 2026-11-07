#!/usr/bin/env python
"""Nested evaluation of handcrafted, diagonal and neural music distances.

The script keeps model fitting, operating-point selection and held-out testing
strictly work-blocked.  Exact DP and adjacent Greedy consume the same learned
affinity/contrast terms, so their difference isolates search rather than
distance supervision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boundary_contrast import boundary_contrast_features
from evaluate_boundary_aware_stage import (
    build_trees,
    candidate_rows,
    discover_pairs,
    f_beta,
    fit_contrast_models,
    items_for_works,
    load_corpus,
    stable_inner_folds,
    work_id,
)
from greedy_evaluation import (
    bootstrap_summary,
    boundary_salience,
    boundary_scores,
    collect_salient_splits,
    micro_summary,
    paired_permutation_tests,
    tree_shape_diagnostics,
)
from neural_distance import (
    DEFAULT_HARMONIC_DILATIONS,
    DEFAULT_TEMPORAL_DILATIONS,
    NEURAL_ARCHITECTURES,
    TEMPORAL_ARCHITECTURES,
    build_interval_training_set,
    train_distance_checkpoints,
)
from ordered_affinity import pairwise_affinity


FAMILIES = ("handcrafted_key_profile", "diagonal_mahalanobis", "neural_circular_siamese")
LEARNED_FAMILIES = FAMILIES[1:]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/neural_distance_stage"))
    parser.add_argument("--piece", action="append")
    parser.add_argument("--outer-work", action="append")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--affinity-method", default="key_profile")
    parser.add_argument("--affinity-context-radius", type=int, default=0)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--checkpoint-epochs", nargs="+", type=int,
                        default=[20, 40, 60, 80, 100])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--neural-architecture", choices=NEURAL_ARCHITECTURES,
                        default="harmonic_tcn_v3")
    parser.add_argument("--neural-dilations", nargs="+", type=int,
                        default=list(DEFAULT_HARMONIC_DILATIONS))
    parser.add_argument("--temporal-dilations", nargs="+", type=int,
                        default=list(DEFAULT_TEMPORAL_DILATIONS))
    parser.add_argument("--temporal-dropout", type=float, default=0.15)
    parser.add_argument("--temporal-affinity-context", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--neural-weight-decay", type=float, default=1e-3)
    parser.add_argument("--neural-batch-size", type=int, default=64)
    parser.add_argument("--metric-loss-weight", type=float, default=0.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-margin", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.25, 0.5])
    parser.add_argument("--balance-weights", nargs="+", type=float, default=[0, 0.1, 0.6])
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[value / 20 for value in range(21)])
    parser.add_argument("--min-separations-qb", nargs="+", type=float, default=[0, 8, 16])
    parser.add_argument("--local-max-radii-qb", nargs="+", type=float, default=[0, 8])
    parser.add_argument("--boundary-budgets", nargs="+", type=int,
                        default=[3, 5, 8, 10, 12, 15, 20])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[4, 8, 12, 24])
    parser.add_argument("--selection-tolerance", type=float, default=8.0)
    parser.add_argument("--selection-metric", choices=["f1", "f0.5"], default="f1")
    parser.add_argument("--ignore-tolerance-qb", type=float, default=None)
    parser.add_argument("--probability-calibration", choices=["none", "platt"], default="platt")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--contrast-epochs", dest="epochs", type=int, default=300)
    parser.add_argument("--contrast-learning-rate", dest="contrast_learning_rate",
                        type=float, default=0.05)
    parser.add_argument("--contrast-regularization", dest="regularization",
                        type=float, default=1e-3)
    parser.add_argument("--tie-break", choices=["earliest", "midpoint", "latest"],
                        default="midpoint")
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--stability-seeds", nargs="*", type=int,
                        default=[20260819, 20260820])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse complete, fingerprint-matched outer-work shards.")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Validate and aggregate existing shards without fitting models.")
    return parser.parse_args()


def validate_args(args):
    if args.ignore_tolerance_qb is None:
        args.ignore_tolerance_qb = float(args.selection_tolerance)
    if (args.bin_size <= 0 or args.selection_tolerance < 0
            or args.ignore_tolerance_qb < 0
            or args.ignore_tolerance_qb > args.selection_tolerance):
        raise SystemExit("invalid bin size or tolerance configuration")
    if any(value < 0 or value > 1 for value in args.lambdas + args.thresholds):
        raise SystemExit("lambdas and thresholds must lie in [0,1]")
    if any(value < 0 for value in args.balance_weights):
        raise SystemExit("balance weights must be non-negative")
    if any(value < 0 for value in args.min_separations_qb + args.local_max_radii_qb):
        raise SystemExit("NMS parameters must be non-negative")
    if any(value < 1 for value in args.checkpoint_epochs):
        raise SystemExit("checkpoint epochs must be positive")
    if (args.channels < 1 or any(value < 1 for value in args.neural_dilations)
            or any(value < 1 for value in args.temporal_dilations)
            or not 0 <= args.temporal_dropout < 1
            or args.temporal_affinity_context < 0
            or args.neural_weight_decay < 0 or args.neural_batch_size < 1
            or args.metric_loss_weight < 0 or args.ranking_loss_weight < 0
            or not 0 < args.contrastive_margin <= 1 or args.ranking_margin < 0
            or args.minimum_learning_rate < 0
            or args.minimum_learning_rate > args.learning_rate):
        raise SystemExit("invalid neural architecture, loss, or optimiser configuration")
    if args.selection_tolerance not in args.tolerances:
        args.tolerances = sorted(set(args.tolerances + [args.selection_tolerance]))


def stable_seed(seed, *parts):
    digest = hashlib.sha256("|".join(map(str, (seed,) + parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


SHARD_TABLES = {
    "assignment_rows": "fold_assignments.csv",
    "inner_rows": "inner_validation_per_piece.csv",
    "inner_diagnostics": "inner_tree_diagnostics.csv",
    "candidate_summary": "inner_candidate_summary.csv",
    "selection_rows": "outer_selections.csv",
    "training_history": "training_history.csv",
    "training_audit": "neural_training_audit.csv",
    "contrast_audit": "contrast_training_audit.csv",
    "model_rows": "model_parameters.csv",
    "learned_affinity_metadata": "learned_affinity_metadata.csv",
    "outer_rows": "held_out_per_piece.csv",
    "outer_diagnostics": "tree_diagnostics.csv",
    "salience_rows": "boundary_salience.csv",
    "distance_rows": "learned_distance_diagnostics.csv",
}

IMPLEMENTATION_FILES = (
    "scripts/evaluate_neural_distance_stage.py",
    "scripts/evaluate_boundary_aware_stage.py",
    "src/neural_distance.py",
    "src/boundary_contrast.py",
    "src/ordered_affinity.py",
    "src/greedy_evaluation.py",
)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def scientific_configuration(args):
    """Return only settings that can change the scientific result."""
    excluded = {
        "aggregate_only", "data_root", "device", "outer_work",
        "output_dir", "resume",
    }
    return {
        key: _json_value(value) for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_hash():
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        path = ROOT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(_sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def dataset_hash(pairs):
    """Content hash of the exact notes/harmonies pairs visible to the run."""
    digest = hashlib.sha256()
    for piece, notes, harmonies in pairs:
        digest.update(piece.encode("utf-8"))
        digest.update(_sha256_file(notes).encode("ascii"))
        digest.update(_sha256_file(harmonies).encode("ascii"))
    return digest.hexdigest()


def configuration_fingerprint(configuration, data_sha256, code_sha256):
    payload = {
        "configuration": configuration,
        "data_sha256": data_sha256,
        "implementation_sha256": code_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def _read_csv_records(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 1:
        return []
    try:
        return pd.read_csv(path).to_dict("records")
    except pd.errors.EmptyDataError:
        return []


def _atomic_save_distance(distance, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    distance.save(temporary)
    temporary.replace(path)


def _shard_marker_path(output_dir, outer_work):
    return Path(output_dir) / "shards" / outer_work / "complete.json"


def validate_shard(output_dir, outer_work, fingerprint):
    """Return a validated marker, or raise when an apparent shard is unsafe."""
    marker_path = _shard_marker_path(output_dir, outer_work)
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unreadable shard marker for {outer_work}: {error}") from error
    if marker.get("status") != "success" or marker.get("outer_work") != outer_work:
        raise RuntimeError(f"Invalid completion marker for shard {outer_work}")
    if marker.get("configuration_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Shard {outer_work} was produced by different data, code, or "
            "settings; use a new output directory.")
    shard_dir = marker_path.parent
    expected_files = marker.get("files", {})
    missing_tables = set(SHARD_TABLES.values()) - set(expected_files)
    if missing_tables:
        raise RuntimeError(
            f"Shard {outer_work} marker omits tables: {sorted(missing_tables)}")
    for relative, expected_hash in expected_files.items():
        path = shard_dir / relative
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"Shard {outer_work} has a missing or corrupt file: {relative}")
    return marker


def load_shard(output_dir, outer_work, collections, fingerprint):
    marker = validate_shard(output_dir, outer_work, fingerprint)
    if marker is None:
        return None
    shard_dir = _shard_marker_path(output_dir, outer_work).parent
    for key, filename in SHARD_TABLES.items():
        records = _read_csv_records(shard_dir / filename)
        mismatched = [
            row.get("outer_work") for row in records
            if str(row.get("outer_work")) != str(outer_work)]
        if mismatched:
            raise RuntimeError(
                f"Shard {outer_work}/{filename} contains another outer work")
        collections[key].extend(records)
    return marker


def write_shard(output_dir, outer_work, collections, starts, fingerprint,
                configuration, data_sha256, code_sha256,
                expected_checkpoint_count):
    shard_dir = Path(output_dir) / "shards" / outer_work
    shard_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    row_counts = {}
    for key, filename in SHARD_TABLES.items():
        rows = collections[key][starts[key]:]
        path = shard_dir / filename
        _atomic_csv(rows, path)
        files[filename] = _sha256_file(path)
        row_counts[filename] = len(rows)
    checkpoint_paths = sorted((shard_dir / "checkpoints").glob("*.npz"))
    for path in checkpoint_paths:
        relative = path.relative_to(shard_dir).as_posix()
        files[relative] = _sha256_file(path)
    if row_counts[SHARD_TABLES["outer_rows"]] == 0:
        raise RuntimeError(f"Refusing to complete empty outer shard {outer_work}")
    if len(checkpoint_paths) != expected_checkpoint_count:
        raise RuntimeError(
            f"Shard {outer_work} has {len(checkpoint_paths)} learned checkpoints; "
            f"expected {expected_checkpoint_count}")
    marker = {
        "schema_version": 1,
        "status": "success",
        "outer_work": outer_work,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "data_sha256": data_sha256,
        "implementation_sha256": code_sha256,
        "git_revision": git_revision(),
        "row_counts": row_counts,
        "checkpoint_count": len(checkpoint_paths),
        "files": files,
    }
    _atomic_json(marker, shard_dir / "complete.json")
    return marker


def training_args(args, kind):
    configuration = {
        "checkpoint_epochs": args.checkpoint_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "gradient_clip": args.gradient_clip,
        "channels": args.channels,
    }
    if kind == "neural":
        configuration.update({
            "architecture": args.neural_architecture,
            "dilations": args.neural_dilations,
            "temporal_dilations": args.temporal_dilations,
            "temporal_dropout": args.temporal_dropout,
            "device": args.device,
            "weight_decay": args.neural_weight_decay,
            "batch_size": args.neural_batch_size,
            "metric_loss_weight": args.metric_loss_weight,
            "ranking_loss_weight": args.ranking_loss_weight,
            "contrastive_margin": args.contrastive_margin,
            "ranking_margin": args.ranking_margin,
            "minimum_learning_rate": args.minimum_learning_rate,
        })
    return configuration


def estimate_training_affinity_scale(
        items, distance, representation_context_radius=0):
    """Equal-work median of non-zero within-work bin distances."""
    work_values = []
    for work in sorted({item["work"] for item in items}):
        values = []
        for item in items:
            if item["work"] != work:
                continue
            contextual = getattr(distance, "transform_contextual_bins", None)
            representation = (contextual(
                item["matrix"], representation_context_radius)
                if contextual is not None else distance.transform_batch(item["matrix"]))
            pairwise = np.linalg.norm(
                representation[:, None, :] - representation[None, :, :], axis=2)
            upper = pairwise[np.triu_indices(len(pairwise), 1)]
            values.extend(upper[np.isfinite(upper) & (upper > 1e-12)])
        if values:
            work_values.append(float(np.median(values)))
    return float(np.median(work_values)) if work_values else 1.0


def materialize_learned_items(items, distance, scale, args):
    result = []
    metadata = []
    for item in items:
        features = boundary_contrast_features(
            item["matrix"], {"learned": distance}, contexts=args.contexts,
            distances=("learned",))
        affinity, info = pairwise_affinity(
            item["matrix"], distance, scale=scale,
            context_radius=args.affinity_context_radius,
            representation_context_radius=getattr(
                args, "temporal_affinity_context", 0))
        current = dict(item)
        current["features"] = features
        current["affinity"] = affinity
        result.append(current)
        metadata.append({"piece": item["piece"], "work": item["work"], **info})
    return result, metadata


def append_fields(rows, **fields):
    return [{**row, **fields} for row in rows]


def select_family_candidate(frame, family, selection_mode, *, metric="f1", epoch=None):
    subset = frame[(frame.family == family) & (frame.selection_mode == selection_mode)].copy()
    if epoch is not None:
        subset = subset[subset.checkpoint_epoch == epoch]
    if subset.empty:
        raise ValueError(f"no candidates for {family}/{selection_mode}")
    keys = [
        "work", "search", "checkpoint_epoch", "contrast_weight", "balance_weight",
        "candidate_value", "min_separation_qb", "local_max_radius_qb",
    ]
    work = subset.groupby(keys, as_index=False)[["precision", "recall", "f1", "f0.5"]].mean()
    candidate_keys = keys[2:]
    candidates = work.groupby(candidate_keys, as_index=False)[
        ["precision", "recall", "f1", "f0.5"]].mean()
    score = "f0.5" if metric == "f0.5" else "f1"
    candidate_ascending = selection_mode == "fixed_budget"
    columns = ([score, "precision", "recall"] if score == "f0.5"
               else [score, "recall", "precision"])
    columns += ["checkpoint_epoch", "candidate_value", "min_separation_qb",
                "local_max_radius_qb", "balance_weight", "contrast_weight"]
    ascending = [False, False, False, True, candidate_ascending, True, True, True, True]
    candidates = candidates.sort_values(columns, ascending=ascending, kind="stable")
    choice = candidates.iloc[0].to_dict()
    choice["selection_metric"] = metric
    choice["selection_score"] = float(choice[score])
    candidates["selection_metric"] = metric
    candidates["selection_score"] = candidates[score]
    candidates["chosen"] = False
    candidates.loc[candidates.index[0], "chosen"] = True
    return choice, candidates


def fit_family_models(train_items, family, args, seed):
    """Return checkpoints, model history and complete local-pair audit."""
    if family == "handcrafted_key_profile":
        return [(0, None)], [], []
    training = build_interval_training_set(
        train_items, contexts=args.contexts,
        ignore_tolerance_qb=args.ignore_tolerance_qb)
    kind = "diagonal" if family == "diagonal_mahalanobis" else "neural"
    checkpoints, history = train_distance_checkpoints(
        training, kind=kind, seed=seed, **training_args(args, kind))
    return [(checkpoint.epoch, checkpoint.distance) for checkpoint in checkpoints], history, list(training.audit)


def family_items(train_items, evaluation_items, family, distance, args):
    if family == "handcrafted_key_profile":
        return train_items, evaluation_items, [], np.nan
    scale = estimate_training_affinity_scale(
        train_items, distance, getattr(args, "temporal_affinity_context", 0))
    transformed_train, train_metadata = materialize_learned_items(
        train_items, distance, scale, args)
    transformed_evaluation, evaluation_metadata = materialize_learned_items(
        evaluation_items, distance, scale, args)
    return transformed_train, transformed_evaluation, train_metadata + evaluation_metadata, scale


def evaluate_selected(items, family, scorer, choices, checkpoint_epoch,
                      training_seed, args, outer_work):
    rows, diagnostics, salience_rows = [], [], []
    for selection_mode, choice in choices.items():
        contrast_weight = float(choice["contrast_weight"])
        balance_weight = float(choice["balance_weight"])
        candidate = float(choice["candidate_value"])
        min_separation = float(choice["min_separation_qb"])
        local_radius = float(choice["local_max_radius_qb"])
        for item in items:
            contrast = scorer.score(item["features"])
            trees = build_trees(
                item, contrast, contrast_weight, balance_weight, args)
            for search, tree, info, seconds in trees:
                if selection_mode == "threshold":
                    predicted = collect_salient_splits(
                        tree, contrast, threshold=candidate,
                        min_separation_qb=min_separation,
                        local_max_radius_qb=local_radius)
                else:
                    predicted = collect_salient_splits(tree, contrast, budget=int(candidate))
                diagnostics.append({
                    "outer_work": outer_work, "piece": item["piece"], "work": item["work"],
                    "family": family, "search": search, "selection_mode": selection_mode,
                    "checkpoint_epoch": checkpoint_epoch, "training_seed": training_seed,
                    "contrast_weight": contrast_weight, "balance_weight": balance_weight,
                    "objective": info.total_objective,
                    "normalized_affinity_revenue": info.normalized_affinity_revenue,
                    "boundary_reward": info.boundary_reward,
                    "balance_penalty": info.balance_penalty,
                    "runtime_seconds": seconds, "evaluated_splits": info.evaluated_splits,
                    "root_split": info.root_split, "tie_count": info.tie_count,
                    **tree_shape_diagnostics(tree),
                })
                for boundary in boundary_salience(tree, contrast):
                    salience_rows.append({
                        "outer_work": outer_work, "piece": item["piece"],
                        "work": item["work"], "family": family, "search": search,
                        "selection_mode": selection_mode,
                        "checkpoint_epoch": checkpoint_epoch,
                        "training_seed": training_seed, **boundary,
                    })
                for tolerance in args.tolerances:
                    score = boundary_scores(predicted, item["reference"], tolerance)
                    rows.append({
                        "outer_work": outer_work, "piece": item["piece"], "work": item["work"],
                        "family": family, "search": search,
                        "selection_mode": selection_mode, "calibration_scope": "shared",
                        "checkpoint_epoch": checkpoint_epoch, "training_seed": training_seed,
                        "selected_value": candidate,
                        "min_separation_qb": min_separation,
                        "local_max_radius_qb": local_radius,
                        "contrast_weight": contrast_weight,
                        "balance_weight": balance_weight,
                        "tolerance_qb": float(tolerance), "oracle": False,
                        "f0.5": float(f_beta(score["precision"], score["recall"])),
                        **{key: score[key] for key in [
                            "tp", "fp", "fn", "precision", "recall", "f1",
                            "predicted_boundary_count", "gt_boundary_count",
                            "mean_match_error_qb"]},
                    })
    return rows, diagnostics, salience_rows


def distance_diagnostics(items, distance, family, outer_work, training_seed, args):
    if distance is None:
        return []
    output = []
    for item in items:
        try:
            examples = build_interval_training_set(
                [item], contexts=args.contexts,
                ignore_tolerance_qb=args.ignore_tolerance_qb)
        except ValueError:
            continue
        if getattr(distance, "architecture", None) in TEMPORAL_ARCHITECTURES:
            values = distance.batch_sequence_distance(
                examples.left_sequences, examples.right_sequences,
                examples.left_masks, examples.right_masks)
            probabilities = distance.predict_boundary_probability_sequence(
                examples.left_sequences, examples.right_sequences,
                examples.left_masks, examples.right_masks)
        else:
            values = distance.batch_distance(examples.left, examples.right)
            probabilities = distance.predict_boundary_probability(
                examples.left, examples.right)
        for index, value in enumerate(values):
            output.append({
                "outer_work": outer_work, "piece": item["piece"], "work": item["work"],
                "family": family, "training_seed": training_seed,
                "label": int(examples.labels[index]),
                "distance": float(value), "boundary_probability": float(probabilities[index]),
                "split_index": int(examples.split_indices[index]),
                "context_bins": int(examples.contexts[index]),
            })
    return output


def plot_outputs(results, diagnostics, output_dir):
    primary = results[(~results.oracle) & results.selection_mode.eq("threshold")]
    if not primary.empty:
        summary = primary.groupby(
            ["family", "search", "training_seed", "tolerance_qb"], as_index=False).f1.mean()
        figure, axis = plt.subplots(figsize=(9, 5))
        for keys, group in summary.groupby(["family", "search"]):
            group = group.groupby("tolerance_qb", as_index=False).f1.mean()
            axis.plot(group.tolerance_qb, group.f1, marker="o", label=" / ".join(keys))
        axis.set(xlabel="Boundary tolerance (quarterbeats)", ylabel="Mean piece F1",
                 title="Neural-distance stage: F1 sensitivity")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output_dir / "f1_tolerance.png", dpi=160)
        plt.close(figure)

        point = primary[np.isclose(primary.tolerance_qb, 8.0)].groupby(
            ["family", "search"], as_index=False)[["precision", "recall"]].mean()
        figure, axis = plt.subplots(figsize=(7, 5))
        for row in point.itertuples():
            axis.scatter(row.recall, row.precision)
            axis.annotate(f"{row.family}/{row.search}", (row.recall, row.precision), fontsize=7)
        axis.set(xlabel="Recall", ylabel="Precision", title="Operating points at 8 qb",
                 xlim=(0, 1), ylim=(0, 1))
        figure.tight_layout()
        figure.savefig(output_dir / "precision_recall.png", dpi=160)
        plt.close(figure)

    if not diagnostics.empty:
        comparison = diagnostics.pivot_table(
            index=["outer_work", "piece", "family", "selection_mode", "training_seed"],
            columns="search", values="objective", aggfunc="first").reset_index()
        if {"greedy", "dp"} <= set(comparison):
            comparison["gap"] = comparison.dp - comparison.greedy
            figure, axis = plt.subplots(figsize=(8, 5))
            groups = [group.gap.to_numpy() for _, group in comparison.groupby("family")]
            labels = [name for name, _ in comparison.groupby("family")]
            axis.boxplot(groups, labels=labels)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set(ylabel="DP objective - Greedy objective",
                     title="Exact-search objective advantage")
            axis.tick_params(axis="x", rotation=15)
            figure.tight_layout()
            figure.savefig(output_dir / "objective_gap.png", dpi=160)
            plt.close(figure)


def main():
    args = parse_args()
    if args.quick:
        args.checkpoint_epochs = [5, 10]
        args.lambdas = [0.0, 0.5]
        args.balance_weights = [0.0, 0.1]
        args.thresholds = [0.3, 0.5, 0.7]
        args.min_separations_qb = [0.0, 16.0]
        args.local_max_radii_qb = [0.0, 8.0]
        args.boundary_budgets = [5, 10]
        args.tolerances = [args.selection_tolerance]
        args.inner_folds = 2
        args.epochs = min(args.epochs, 25)
        args.stability_seeds = []
    # Existing scorer helper expects this historical attribute name.
    args.learning_rate_contrast = args.contrast_learning_rate
    # fit_contrast_models reads args.learning_rate; temporarily expose contrast
    # settings through a small namespace adapter below rather than altering the
    # frozen advanced optimiser configuration.
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shards").mkdir(exist_ok=True)

    pairs = discover_pairs(args.data_root, args.piece)
    if args.quick and not args.piece:
        grouped = {}
        for pair in pairs:
            grouped.setdefault(work_id(pair[0]), []).append(pair)
        selected = sorted(grouped)[:6]
        pairs = [sorted(grouped[work])[0] for work in selected]
    if not pairs:
        raise SystemExit("No paired ABC notes/harmonies files found")
    configuration = scientific_configuration(args)
    data_sha256 = dataset_hash(pairs)
    code_sha256 = implementation_hash()
    fingerprint = configuration_fingerprint(
        configuration, data_sha256, code_sha256)

    # load_corpus uses args.learning_rate only later through fit_contrast_models;
    # keep neural LR separately and present the contrast LR during scorer fits.
    neural_learning_rate = args.learning_rate
    args.learning_rate = args.contrast_learning_rate
    cache, statuses, base_affinity_metadata = load_corpus(pairs, args)
    args.learning_rate = neural_learning_rate
    pd.DataFrame(statuses).to_csv(args.output_dir / "run_status.csv", index=False)
    if not cache:
        raise SystemExit("All pieces failed; inspect run_status.csv")
    all_works = sorted({item["work"] for item in cache.values()})
    requested = set(args.outer_work or all_works)
    outer_works = [work for work in all_works if work in requested]
    if args.quick and not args.outer_work:
        outer_works = outer_works[:2]
    if requested - set(all_works):
        raise SystemExit(f"Unknown outer works: {sorted(requested - set(all_works))}")
    if len(all_works) < 4:
        raise SystemExit("At least four works are required")

    context = {
        "schema_version": 1,
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "data_sha256": data_sha256,
        "implementation_sha256": code_sha256,
        "git_revision": git_revision(),
        "all_works": all_works,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    context_path = args.output_dir / "run_context.json"
    if context_path.is_file():
        existing_context = json.loads(context_path.read_text(encoding="utf-8"))
        if existing_context.get("configuration_fingerprint") != fingerprint:
            raise SystemExit(
                "Output directory contains a different Advanced configuration, "
                "dataset, or implementation. Use a new output directory.")
        if existing_context.get("all_works") != all_works:
            raise SystemExit(
                "Output directory was created with a different work manifest.")
    else:
        _atomic_json(context, context_path)

    assignment_rows, inner_rows, inner_diagnostics = [], [], []
    selection_rows, candidate_summary = [], []
    training_history, training_audit, contrast_audit = [], [], []
    model_rows, learned_affinity_metadata = [], []
    outer_rows, outer_diagnostics, salience_rows, distance_rows = [], [], [], []

    collections = {
        "assignment_rows": assignment_rows,
        "inner_rows": inner_rows,
        "inner_diagnostics": inner_diagnostics,
        "candidate_summary": candidate_summary,
        "selection_rows": selection_rows,
        "training_history": training_history,
        "training_audit": training_audit,
        "contrast_audit": contrast_audit,
        "model_rows": model_rows,
        "learned_affinity_metadata": learned_affinity_metadata,
        "outer_rows": outer_rows,
        "outer_diagnostics": outer_diagnostics,
        "salience_rows": salience_rows,
        "distance_rows": distance_rows,
    }
    works_to_run = []
    completed_works = []
    for outer_work in outer_works:
        marker = validate_shard(args.output_dir, outer_work, fingerprint)
        if marker is not None:
            if not (args.resume or args.aggregate_only):
                raise SystemExit(
                    f"Complete shard already exists for {outer_work}; use "
                    "--resume or a new output directory.")
            load_shard(args.output_dir, outer_work, collections, fingerprint)
            completed_works.append(outer_work)
            print(f"[resume] {outer_work}: validated and loaded", flush=True)
        elif args.aggregate_only:
            raise SystemExit(
                f"Cannot aggregate: complete shard is missing for {outer_work}")
        else:
            works_to_run.append(outer_work)
    _atomic_json({
        "phase": "aggregating" if args.aggregate_only else "running",
        "configuration_fingerprint": fingerprint,
        "requested_outer_works": outer_works,
        "completed_outer_works": completed_works,
        "remaining_outer_works": works_to_run,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, args.output_dir / "run_progress.json")

    learned_seed_count = len({
        args.seed, *[value for value in args.stability_seeds
                    if value != args.seed]})
    expected_checkpoint_count = (
        sum(family in LEARNED_FAMILIES for family in args.families)
        * learned_seed_count)

    for outer_number, outer_work in enumerate(works_to_run, 1):
        starts = {key: len(rows) for key, rows in collections.items()}
        train_works = [work for work in all_works if work != outer_work]
        folds = stable_inner_folds(train_works, args.inner_folds, args.seed, outer_work)
        fold_candidates = []
        for fold_index, validation_works in enumerate(folds):
            inner_train_works = [work for work in train_works if work not in set(validation_works)]
            base_train = items_for_works(cache, inner_train_works)
            base_validation = items_for_works(cache, validation_works)
            for work in inner_train_works:
                assignment_rows.append({"outer_work": outer_work, "inner_fold": fold_index,
                                        "work": work, "split": "inner_train"})
            for work in validation_works:
                assignment_rows.append({"outer_work": outer_work, "inner_fold": fold_index,
                                        "work": work, "split": "inner_validation"})

            for family in args.families:
                seed = stable_seed(args.seed, outer_work, fold_index, family)
                checkpoints, history, audit = fit_family_models(base_train, family, args, seed)
                training_history.extend(append_fields(
                    history, outer_work=outer_work, inner_fold=fold_index,
                    stage="inner", family=family, training_seed=seed))
                training_audit.extend(append_fields(
                    audit, outer_work=outer_work, inner_fold=fold_index,
                    stage="inner", family=family, training_seed=seed))
                for epoch, distance in checkpoints:
                    current_train, current_validation, affinity_meta, scale = family_items(
                        base_train, base_validation, family, distance, args)
                    learned_affinity_metadata.extend(append_fields(
                        affinity_meta, outer_work=outer_work, inner_fold=fold_index,
                        stage="inner", family=family, checkpoint_epoch=epoch,
                        training_seed=seed, training_affinity_scale=scale))
                    args.learning_rate = args.contrast_learning_rate
                    contrast_models, current_audit, current_history = fit_contrast_models(
                        current_train, args)
                    args.learning_rate = neural_learning_rate
                    scorer = contrast_models["supervised_contrast"]
                    contrast_audit.extend(append_fields(
                        current_audit, outer_work=outer_work, inner_fold=fold_index,
                        stage="inner", family=family, checkpoint_epoch=epoch))
                    training_history.extend(append_fields(
                        current_history, outer_work=outer_work, inner_fold=fold_index,
                        stage="inner_contrast", family=family,
                        checkpoint_epoch=epoch, training_seed=seed))
                    rows, diagnostics = candidate_rows(
                        current_validation, family, scorer, args.lambdas,
                        args.balance_weights, args,
                        {"outer_work": outer_work, "inner_fold": fold_index})
                    rows = append_fields(rows, family=family, checkpoint_epoch=epoch,
                                         training_seed=seed)
                    diagnostics = append_fields(
                        diagnostics, family=family, checkpoint_epoch=epoch,
                        training_seed=seed)
                    fold_candidates.extend(rows)
                    inner_diagnostics.extend(diagnostics)
                    if distance is not None:
                        model_rows.append({
                            "outer_work": outer_work, "inner_fold": fold_index,
                            "stage": "inner", "family": family,
                            "checkpoint_epoch": epoch, "training_seed": seed,
                            "training_affinity_scale": scale,
                            "model": json.dumps(distance.as_dict(), sort_keys=True),
                        })

        frame = pd.DataFrame(fold_candidates)
        inner_rows.extend(fold_candidates)
        selections = {}
        for family in args.families:
            threshold_choice, threshold_candidates = select_family_candidate(
                frame, family, "threshold", metric=args.selection_metric)
            epoch = int(threshold_choice["checkpoint_epoch"])
            budget_choice, budget_candidates = select_family_candidate(
                frame, family, "fixed_budget", metric=args.selection_metric,
                epoch=epoch)
            selections[family] = {
                "threshold": threshold_choice, "fixed_budget": budget_choice}
            for selection_mode, choice, candidates in (
                    ("threshold", threshold_choice, threshold_candidates),
                    ("fixed_budget", budget_choice, budget_candidates)):
                candidate_summary.extend(append_fields(
                    candidates.to_dict("records"), outer_work=outer_work,
                    family=family, selection_mode=selection_mode))
                selection_rows.append({
                    "outer_work": outer_work, "family": family,
                    "selection_mode": selection_mode,
                    "selection_metric": args.selection_metric,
                    "selection_score": choice["selection_score"],
                    "selected_checkpoint_epoch": int(choice["checkpoint_epoch"]),
                    "selected_contrast_weight": choice["contrast_weight"],
                    "selected_balance_weight": choice["balance_weight"],
                    "selected_value": choice["candidate_value"],
                    "selected_min_separation_qb": choice["min_separation_qb"],
                    "selected_local_max_radius_qb": choice["local_max_radius_qb"],
                    "inner_precision": choice["precision"],
                    "inner_recall": choice["recall"], "inner_f1": choice["f1"],
                    "inner_f0.5": choice["f0.5"],
                })

        base_outer_train = items_for_works(cache, train_works)
        base_test = items_for_works(cache, [outer_work])
        assignment_rows.extend({"outer_work": outer_work, "inner_fold": None,
                                "work": work, "split": "outer_train"}
                               for work in train_works)
        assignment_rows.append({"outer_work": outer_work, "inner_fold": None,
                                "work": outer_work, "split": "outer_test"})
        for family in args.families:
            epoch = int(selections[family]["threshold"]["checkpoint_epoch"])
            seeds = [args.seed] if family == "handcrafted_key_profile" else [
                args.seed, *[value for value in args.stability_seeds if value != args.seed]]
            for evaluation_seed in seeds:
                if family == "handcrafted_key_profile":
                    distance, history, audit = None, [], []
                else:
                    original_epochs = args.checkpoint_epochs
                    args.checkpoint_epochs = [epoch]
                    checkpoints, history, audit = fit_family_models(
                        base_outer_train, family, args, evaluation_seed)
                    args.checkpoint_epochs = original_epochs
                    distance = checkpoints[0][1]
                training_history.extend(append_fields(
                    history, outer_work=outer_work, inner_fold=None,
                    stage="outer_refit", family=family,
                    training_seed=evaluation_seed))
                training_audit.extend(append_fields(
                    audit, outer_work=outer_work, inner_fold=None,
                    stage="outer_refit", family=family,
                    training_seed=evaluation_seed))
                current_train, current_test, affinity_meta, scale = family_items(
                    base_outer_train, base_test, family, distance, args)
                learned_affinity_metadata.extend(append_fields(
                    affinity_meta, outer_work=outer_work, inner_fold=None,
                    stage="outer_refit", family=family,
                    checkpoint_epoch=epoch, training_seed=evaluation_seed,
                    training_affinity_scale=scale))
                args.learning_rate = args.contrast_learning_rate
                contrast_models, current_audit, current_history = fit_contrast_models(
                    current_train, args)
                args.learning_rate = neural_learning_rate
                scorer = contrast_models["supervised_contrast"]
                contrast_audit.extend(append_fields(
                    current_audit, outer_work=outer_work, inner_fold=None,
                    stage="outer_refit", family=family,
                    checkpoint_epoch=epoch, training_seed=evaluation_seed))
                training_history.extend(append_fields(
                    current_history, outer_work=outer_work, inner_fold=None,
                    stage="outer_refit_contrast", family=family,
                    checkpoint_epoch=epoch, training_seed=evaluation_seed))
                rows, diagnostics, salience = evaluate_selected(
                    current_test, family, scorer, selections[family], epoch,
                    evaluation_seed, args, outer_work)
                outer_rows.extend(rows)
                outer_diagnostics.extend(diagnostics)
                salience_rows.extend(salience)
                distance_rows.extend(distance_diagnostics(
                    current_test, distance, family, outer_work,
                    evaluation_seed, args))
                if distance is not None:
                    checkpoint_path = (args.output_dir / "shards" / outer_work /
                                       "checkpoints" /
                                       f"{outer_work}_{family}_seed{evaluation_seed}.npz")
                    _atomic_save_distance(distance, checkpoint_path)
                    model_rows.append({
                        "outer_work": outer_work, "inner_fold": None,
                        "stage": "outer_refit", "family": family,
                        "checkpoint_epoch": epoch, "training_seed": evaluation_seed,
                        "training_affinity_scale": scale,
                        "checkpoint_path": str(checkpoint_path),
                        "model": json.dumps(distance.as_dict(), sort_keys=True),
                    })
        write_shard(
            args.output_dir, outer_work, collections, starts, fingerprint,
            configuration, data_sha256, code_sha256,
            expected_checkpoint_count)
        completed_works.append(outer_work)
        _atomic_json({
            "phase": "running",
            "configuration_fingerprint": fingerprint,
            "requested_outer_works": outer_works,
            "completed_outer_works": completed_works,
            "remaining_outer_works": [
                work for work in works_to_run if work not in completed_works],
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }, args.output_dir / "run_progress.json")
        print(
            f"[outer {outer_number}/{len(works_to_run)}] {outer_work}: "
            "shard committed", flush=True)

    results = pd.DataFrame(outer_rows)
    diagnostics = pd.DataFrame(outer_diagnostics)
    pd.DataFrame(assignment_rows).drop_duplicates().to_csv(
        args.output_dir / "fold_assignments.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(args.output_dir / "inner_validation_per_piece.csv", index=False)
    pd.DataFrame(inner_diagnostics).to_csv(args.output_dir / "inner_tree_diagnostics.csv", index=False)
    pd.DataFrame(candidate_summary).to_csv(args.output_dir / "inner_candidate_summary.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_dir / "outer_selections.csv", index=False)
    pd.DataFrame(training_history).to_csv(args.output_dir / "training_history.csv", index=False)
    pd.DataFrame(training_audit).to_csv(args.output_dir / "neural_training_audit.csv", index=False)
    pd.DataFrame(contrast_audit).to_csv(args.output_dir / "contrast_training_audit.csv", index=False)
    pd.DataFrame(model_rows).to_csv(args.output_dir / "model_parameters.csv", index=False)
    pd.DataFrame(base_affinity_metadata + learned_affinity_metadata).to_csv(
        args.output_dir / "affinity_metadata.csv", index=False)
    results.to_csv(args.output_dir / "held_out_per_piece.csv", index=False)
    diagnostics.to_csv(args.output_dir / "tree_diagnostics.csv", index=False)
    pd.DataFrame(salience_rows).to_csv(args.output_dir / "boundary_salience.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(args.output_dir / "learned_distance_diagnostics.csv", index=False)

    group = ["family", "search", "selection_mode", "calibration_scope",
             "training_seed", "tolerance_qb", "oracle"]
    work_results = results.groupby(["work"] + group, as_index=False)[
        ["tp", "fp", "fn", "precision", "recall", "f1", "f0.5"]].mean()
    work_results.to_csv(args.output_dir / "held_out_per_work.csv", index=False)
    bootstrap_summary(work_results, group, samples=2000, seed=args.seed).to_csv(
        args.output_dir / "work_macro_summary.csv", index=False)
    bootstrap_summary(work_results, group, value="f0.5", samples=2000, seed=args.seed).to_csv(
        args.output_dir / "work_macro_f0_5_summary.csv", index=False)
    micro = micro_summary(results, group)
    micro["f0.5"] = f_beta(micro.precision, micro.recall)
    micro.to_csv(args.output_dir / "micro_summary.csv", index=False)

    comparison = diagnostics.pivot_table(
        index=["outer_work", "piece", "family", "selection_mode", "training_seed"],
        columns="search", values="objective", aggfunc="first").reset_index()
    if {"greedy", "dp"} <= set(comparison):
        comparison["objective_gap"] = comparison.dp - comparison.greedy
        comparison["relative_objective_gap"] = (
            comparison.objective_gap / comparison.dp.abs().replace(0, np.nan))
        if (comparison.objective_gap < -1e-8).any():
            raise AssertionError("exact DP objective is below Greedy")
    comparison.to_csv(args.output_dir / "objective_comparison.csv", index=False)

    primary = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & work_results.training_seed.eq(args.seed)
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)].copy()
    primary["method"] = primary.family + "_" + primary.search
    paired_permutation_tests(
        primary[["work", "method", "f1"]], samples=10000,
        seed=args.seed, unit_column="work").to_csv(
        args.output_dir / "primary_paired_tests.csv", index=False)
    stability = work_results[
        (~work_results.oracle) & work_results.selection_mode.eq("threshold")
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)]
    stability.groupby(["work", "family", "search"], as_index=False).agg(
        seed_mean_f1=("f1", "mean"), seed_std_f1=("f1", "std"),
        seeds=("training_seed", "nunique")).to_csv(
        args.output_dir / "seed_stability.csv", index=False)
    plot_outputs(results, diagnostics, args.output_dir)

    metadata = {
        "primary_metric": "work-level macro DCML local-key Boundary F1",
        "primary_configuration": {
            "bin_size_qb": args.bin_size,
            "selection_tolerance_qb": args.selection_tolerance,
            "selection_mode": "nested-validation shared threshold",
            "training_seed": args.seed,
        },
        "families": args.families,
        "neural_architecture": {
            "name": args.neural_architecture,
            "channels": args.channels,
            "circular_dilations": args.neural_dilations,
            "temporal_dilations": args.temporal_dilations,
            "temporal_dropout": args.temporal_dropout,
            "temporal_affinity_context_bins": args.temporal_affinity_context,
            "input": "masked ordered chroma sequences for harmonic_tcn_v3",
            "distance": "0.5 * Euclidean distance between L2-normalised embeddings",
        },
        "neural_supervision": (
            "nearest-bin DCML local-key boundary positives; tolerance-neighbour "
            "candidates ignored; all remaining negatives retained; work/class-balanced "
            "BCE; optional contrastive-margin and hard-negative ranking ablations"),
        "neural_optimiser": {
            "name": "AdamW with cosine annealing",
            "learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "weight_decay": args.neural_weight_decay,
            "batch_size": args.neural_batch_size,
            "gradient_clip": args.gradient_clip,
            "metric_loss_weight": args.metric_loss_weight,
            "ranking_loss_weight": args.ranking_loss_weight,
            "contrastive_margin": args.contrastive_margin,
            "ranking_margin": args.ranking_margin,
            "device": args.device,
        },
        "distance_usage": "same frozen distance supplies leaf affinity and local boundary contrast",
        "dp_optimality": (
            "exact only for fixed ordered leaves, frozen learned terms and the stated "
            "additive boundary-aware objective"),
        "ground_truth_warning": (
            "DCML local-key annotations are flat boundaries, not hierarchical ground-truth trees"),
        "checkpoint_epochs": args.checkpoint_epochs,
        "contexts": args.contexts, "lambdas": args.lambdas,
        "balance_weights": args.balance_weights, "thresholds": args.thresholds,
        "min_separations_qb": args.min_separations_qb,
        "local_max_radii_qb": args.local_max_radii_qb,
        "tolerances_qb": args.tolerances,
        "ignore_tolerance_qb": args.ignore_tolerance_qb,
        "outer_works": outer_works, "inner_folds": args.inner_folds,
        "seed": args.seed, "stability_seeds": args.stability_seeds,
        "configuration_fingerprint": fingerprint,
        "data_sha256": data_sha256,
        "implementation_sha256": code_sha256,
        "git_revision": git_revision(),
        "shard_schema_version": 1,
        "complete_outer_work_shards": len(outer_works),
        "expected_learned_checkpoints": (
            len(outer_works) * expected_checkpoint_count),
        "resume_enabled": args.resume,
        "aggregate_only": args.aggregate_only,
        "test_used_for_selection": False,
        "reinforcement_learning": "not implemented; reserved for future/exploratory work",
        "successful_movements": int(sum(row["status"] == "success" for row in statuses)),
        "skipped_movements": int(sum(row["status"] == "skipped" for row in statuses)),
        "failed_movements": int(sum(row["status"] == "failed" for row in statuses)),
    }
    _atomic_json(metadata, args.output_dir / "metadata.json")
    _atomic_json({
        "phase": "complete",
        "configuration_fingerprint": fingerprint,
        "requested_outer_works": outer_works,
        "completed_outer_works": outer_works,
        "remaining_outer_works": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }, args.output_dir / "run_progress.json")
    summary = work_results[
        (~work_results.oracle)
        & work_results.selection_mode.eq("threshold")
        & work_results.training_seed.eq(args.seed)
        & np.isclose(work_results.tolerance_qb, args.selection_tolerance)].groupby(
            ["family", "search"], as_index=False).agg(
                works=("work", "nunique"), precision=("precision", "mean"),
                recall=("recall", "mean"), f0_5=("f0.5", "mean"),
                f1=("f1", "mean"), std_f1=("f1", "std"))
    print("\nHELD-OUT WORK SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
