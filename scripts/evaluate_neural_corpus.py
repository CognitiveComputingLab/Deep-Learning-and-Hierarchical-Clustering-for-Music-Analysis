#!/usr/bin/env python
"""Work-blocked out-of-fold evaluation of learned music distances.

This entry point is deliberately metric-learning only.  It evaluates every
complete ABC work exactly once as unseen data while keeping model selection,
boundary-budget selection, and checkpointing inside each outer fold.  The
more expensive RL extension remains in ``train_deep_clustering.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from greedy_clustering import ClusterNode
from greedy_evaluation import (
    boundary_scores, collect_prominent_splits, paired_permutation_tests,
    ted_diagnostics, tree_shape_diagnostics,
)
from neural_clustering import (
    NeuralEmbeddingDistance, boundary_average_precision,
    reference_boundary_indices, state_dict_cpu,
)
from ordered_affinity import (
    affinity_tree_revenue, greedy_adjacent_average_linkage,
    optimal_affinity_tree, pairwise_affinity,
)
from train_deep_clustering import (
    deep_interval_examples, metric_held_out_work_rows, resolve_device,
    summarize_held_out, train_metric,
)
from train_parametric_distance import discover_pairs, load_cache, work_id


ENCODER_LABELS = {
    "mlp": "mlp",
    "harmonic_cnn": "harmonic_cnn",
    "circular_harmonic_cnn": "harmonic_cnn",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/neural_corpus"))
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODER_LABELS),
                        default=["mlp", "harmonic_cnn"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260827, 20260828, 20260829])
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--validation-works", type=int, default=3)
    parser.add_argument("--contexts", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--budgets", nargs="+", type=int,
                        default=[3, 5, 8, 10, 12, 15, 20])
    parser.add_argument("--bin-size", type=float, default=8.0)
    parser.add_argument("--tolerance", type=float, default=8.0)
    parser.add_argument("--max-bins", type=int, default=350)
    parser.add_argument("--metric-epochs", type=int, default=200)
    parser.add_argument("--metric-patience", type=int, default=20)
    parser.add_argument("--metric-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ted-center-bins", type=int, default=20)
    parser.add_argument("--ted-width-bins", type=int, default=10)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def configure_quick(args):
    if not args.quick:
        return
    args.seeds = args.seeds[:1]
    args.outer_folds = 2
    args.validation_works = 1
    args.contexts = [1, 2]
    args.budgets = [3, 5]
    args.metric_epochs = min(args.metric_epochs, 2)
    args.metric_patience = min(args.metric_patience, 1)
    args.permutation_samples = min(args.permutation_samples, 200)


def canonical_encoders(values):
    result = []
    for value in values:
        name = ENCODER_LABELS[value]
        if name not in result:
            result.append(name)
    return result


def grouped_pairs(pairs):
    groups = {}
    for pair in pairs:
        groups.setdefault(work_id(pair[0]), []).append(pair)
    return {work: sorted(items) for work, items in groups.items()}


def make_outer_folds(pairs, args):
    """Return deterministic nested work splits; no annotation is inspected."""
    groups = grouped_pairs(pairs)
    works = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(args.split_seed)
    ordered = list(works[rng.permutation(len(works))])
    if args.quick:
        ordered = ordered[:min(6, len(ordered))]
        groups = {work: groups[work][:1] for work in ordered}
    if args.outer_folds < 2 or args.outer_folds > len(ordered):
        raise ValueError("outer-folds must be between 2 and the number of works")
    chunks = [list(chunk) for chunk in np.array_split(ordered, args.outer_folds)]
    folds = []
    for fold_index, test_works in enumerate(chunks, start=1):
        remaining = [work for work in ordered if work not in test_works]
        validation_rng = np.random.default_rng(args.split_seed + 1000 + fold_index)
        validation_order = list(np.asarray(remaining, dtype=object)[
            validation_rng.permutation(len(remaining))])
        if args.validation_works < 1 or args.validation_works >= len(remaining):
            raise ValueError("validation-works must leave at least one training work")
        validation_works = validation_order[:args.validation_works]
        train_works = [work for work in remaining if work not in validation_works]
        names = {
            "train": train_works,
            "validation": validation_works,
            "test": test_works,
        }
        split_pairs = {
            role: [pair for work in selected for pair in groups[work]]
            for role, selected in names.items()
        }
        folds.append({"fold": fold_index, "works": names, "pairs": split_pairs})
    test_counts = pd.Series([work for fold in folds for work in fold["works"]["test"]]).value_counts()
    if set(test_counts.index) != set(ordered) or not (test_counts == 1).all():
        raise AssertionError("every selected work must be outer-test exactly once")
    return folds


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def annotation_event(rows, fold, event, scope, details=""):
    rows.append({
        "sequence": len(rows) + 1, "fold": fold, "event": event,
        "annotation_scope": scope, "details": details,
    })


def references(item):
    gt = [end for _, end, _ in item["segments"][:-1]]
    return gt, reference_boundary_indices(item["bounds"], gt)


def evaluate_distance_pair(cache, prefix, distance_getter, budgets, args,
                           *, include_ted):
    """Evaluate Greedy and exact DP on the identical leaf-affinity matrix."""
    score_rows, diagnostic_rows, ted_rows = [], [], []
    for piece, item in sorted(cache.items()):
        matrix, bounds = item["matrix"], item["bounds"]
        distance = distance_getter(item)
        affinity_started = time.perf_counter()
        affinity, affinity_info = pairwise_affinity(matrix, distance)
        affinity_seconds = time.perf_counter() - affinity_started

        greedy_started = time.perf_counter()
        greedy = greedy_adjacent_average_linkage(matrix, bounds, affinity, ClusterNode)
        greedy_search_seconds = time.perf_counter() - greedy_started
        greedy_revenue, greedy_normalized = affinity_tree_revenue(greedy, affinity)

        dp_started = time.perf_counter()
        dp, dp_info = optimal_affinity_tree(matrix, bounds, affinity, ClusterNode)
        dp_search_seconds = time.perf_counter() - dp_started
        if dp_info.total_revenue < greedy_revenue - 1e-8:
            raise AssertionError(f"DP revenue below Greedy for {piece} ({prefix})")

        gt, reference_indices = references(item)
        methods = [
            (f"{prefix}_affinity_greedy", greedy, greedy_revenue,
             greedy_normalized, greedy_search_seconds),
            (f"{prefix}_dp", dp, dp_info.total_revenue,
             dp_info.normalized_revenue, dp_search_seconds),
        ]
        for model_name, tree, revenue, normalized, search_seconds in methods:
            runtime = affinity_seconds + search_seconds
            boundary_ap = boundary_average_precision(
                tree, item["bounds"], reference_indices)
            for budget in budgets:
                score = boundary_scores(
                    collect_prominent_splits(tree, budget), gt, args.tolerance)
                score_rows.append({
                    "piece": piece, "work": item["work"], "model": model_name,
                    "budget": int(budget), "boundary_ap": boundary_ap,
                    "runtime_seconds": runtime,
                    **{key: score[key] for key in
                       ("tp", "fp", "fn", "precision", "recall", "f1")},
                })
            diagnostic_rows.append({
                "piece": piece, "work": item["work"], "model": model_name,
                "affinity_seconds": affinity_seconds,
                "search_seconds": search_seconds, "runtime_seconds": runtime,
                "shared_objective_revenue": revenue,
                "shared_normalized_revenue": normalized,
                "affinity_scale": affinity_info["scale"],
                **tree_shape_diagnostics(tree),
            })
            if include_ted:
                ted_rows.append({
                    "piece": piece, "work": item["work"], "model": model_name,
                    **ted_diagnostics(
                        tree, item["segments"], item["total"],
                        args.ted_center_bins, args.ted_width_bins),
                })
    return (pd.DataFrame(score_rows), pd.DataFrame(diagnostic_rows),
            pd.DataFrame(ted_rows))


def select_shared_budget(validation, models, fold, seed, encoder):
    work = (validation[validation.model.isin(models)]
            .groupby(["model", "budget", "work"], as_index=False).f1.mean())
    summary = (work.groupby("budget", as_index=False).f1.mean()
               .sort_values(["f1", "budget"], ascending=[False, True]))
    selected = int(summary.iloc[0].budget)
    rows = []
    for row in summary.itertuples(index=False):
        rows.append({
            "fold": fold, "seed": seed, "encoder": encoder,
            "model_group": "+".join(models), "budget": int(row.budget),
            "validation_work_macro_f1": float(row.f1),
            "selected": int(row.budget) == selected,
        })
    return selected, rows


def add_identity(frame, fold, seed, encoder):
    result = frame.copy()
    result.insert(0, "encoder", encoder)
    result.insert(0, "seed", seed)
    result.insert(0, "fold", fold)
    return result


def selected_rows(frame, budget):
    return frame[frame.budget == budget].copy().reset_index(drop=True)


def ablation_summary(work_frame):
    means = work_frame.groupby(["seed", "model"], as_index=False).f1.mean()
    pivot = means.pivot(index="seed", columns="model", values="f1")
    comparisons = [
        ("harmonic_cnn_dp_minus_mlp_dp", "harmonic_cnn_dp", "mlp_dp"),
        ("harmonic_cnn_greedy_minus_mlp_greedy",
         "harmonic_cnn_affinity_greedy", "mlp_affinity_greedy"),
        ("harmonic_cnn_dp_minus_key_profile_dp",
         "harmonic_cnn_dp", "key_profile_dp"),
        ("mlp_dp_minus_key_profile_dp", "mlp_dp", "key_profile_dp"),
        ("harmonic_cnn_dp_minus_harmonic_cnn_greedy",
         "harmonic_cnn_dp", "harmonic_cnn_affinity_greedy"),
        ("mlp_dp_minus_mlp_greedy", "mlp_dp", "mlp_affinity_greedy"),
        ("key_profile_dp_minus_key_profile_greedy",
         "key_profile_dp", "key_profile_affinity_greedy"),
    ]
    rows = []
    for label, left, right in comparisons:
        if left in pivot and right in pivot:
            differences = (pivot[left] - pivot[right]).dropna()
            rows.append({
                "comparison": label,
                "mean_work_macro_f1_difference": float(differences.mean()),
                "std_across_seeds": (float(differences.std(ddof=1))
                                     if len(differences) > 1 else 0.0),
                "n_seeds": len(differences),
            })
    return pd.DataFrame(rows)


def plot_metric_histories(history, output_dir):
    figure, axis = plt.subplots(figsize=(9, 5))
    for keys, group in history.groupby(["fold", "encoder", "seed"]):
        fold, encoder, seed = keys
        axis.plot(group.epoch, group.validation_work_macro_ap,
                  label=f"f{fold} {encoder} {seed}", alpha=0.8)
    axis.set(title="Nested validation metric-learning AP", xlabel="epoch",
             ylabel="work-macro average precision")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "metric_learning_curves.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    configure_quick(args)
    args.encoders = canonical_encoders(args.encoders)
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(args.data_root)
    if not pairs:
        raise SystemExit(f"No paired ABC annotations found under {args.data_root}")
    folds = make_outer_folds(pairs, args)
    fold_assignment_rows = []
    for fold in folds:
        for role, selected in fold["works"].items():
            for work in selected:
                for piece, _, _ in fold["pairs"][role]:
                    if work_id(piece) == work:
                        fold_assignment_rows.append({
                            "fold": fold["fold"], "role": role,
                            "work": work, "piece": piece,
                        })
    pd.DataFrame(fold_assignment_rows).to_csv(
        output_dir / "fold_assignments.csv", index=False)

    config = {
        "protocol": "nested_work_blocked_out_of_fold",
        "primary_metrics": ["work_macro_precision", "work_macro_recall",
                            "work_macro_f1"],
        "auxiliary_metrics": ["boundary_prominence_average_precision", "TED",
                              "tree_shape", "runtime"],
        "test_access_rule": (
            "Within each outer fold, test annotations are loaded only after all "
            "encoder checkpoints and validation-selected budgets are frozen."),
        "full_corpus_interpretation": (
            "Every selected work is unseen outer-test data exactly once; training "
            "works are never included in their own held-out score."),
        "encoders": args.encoders, "seeds": args.seeds,
        "split_seed": args.split_seed, "outer_folds": args.outer_folds,
        "validation_works": args.validation_works, "contexts": args.contexts,
        "budgets": args.budgets, "bin_size": args.bin_size,
        "tolerance": args.tolerance, "max_bins": args.max_bins,
        "metric_epochs": args.metric_epochs,
        "metric_patience": args.metric_patience, "metric_lr": args.metric_lr,
        "batch_size": args.batch_size, "device": str(device),
        "quick": args.quick,
        "augmentation": (
            "Joint circular pitch-class transposition of left/right intervals; "
            "the same dedicated seeded shift schedule is used by both encoders."),
        "boundary_budget_policy": (
            "For each fold/seed, MLP/CNN and their affinity-Greedy/DP searches "
            "share one validation-selected budget; key-profile Greedy/DP share "
            "a separate validation-selected baseline budget."),
        "deterministic_baseline_seeds": (
            "Key-profile rows are repeated across neural seeds for paired tables; "
            "their across-seed standard deviation is therefore zero."),
        "rl_scope": (
            "Excluded from this 24-training metric ablation; RL is evaluated by "
            "train_deep_clustering.py on the fixed 10/3/3 split."),
    }
    save_json(output_dir / "config.json", config)
    state = {"phase": "running", "completed_folds": [],
             "test_annotations_loaded": False}
    save_json(output_dir / "experiment_state.json", state)

    access_rows, status_rows, history_frames = [], [], []
    budget_rows, held_piece_frames, diagnostic_frames, ted_frames = [], [], [], []
    metric_test_rows = []

    for fold in folds:
        fold_id = fold["fold"]
        print(f"\n=== Outer fold {fold_id}/{len(folds)} ===")
        train_cache, train_status = load_cache(
            fold["pairs"]["train"], args.bin_size, args.max_bins)
        validation_cache, validation_status = load_cache(
            fold["pairs"]["validation"], args.bin_size, args.max_bins)
        for role, statuses in (("train", train_status),
                               ("validation", validation_status)):
            status_rows.extend({"fold": fold_id, "role": role, **row}
                               for row in statuses)
        annotation_event(access_rows, fold_id, "development_annotations_loaded",
                         "train+validation")
        if not train_cache or not validation_cache:
            raise RuntimeError(f"fold {fold_id} has an empty development split")

        train_examples = deep_interval_examples(
            train_cache, args.contexts, args.split_seed + fold_id)
        validation_examples = deep_interval_examples(
            validation_cache, args.contexts, args.split_seed + 100 + fold_id)
        if not train_examples or not validation_examples:
            raise RuntimeError(f"fold {fold_id} produced no metric examples")

        key_validation, _, _ = evaluate_distance_pair(
            validation_cache, "key_profile",
            lambda item: item["distances"]["key_profile"], args.budgets, args,
            include_ted=False)
        key_models = ["key_profile_affinity_greedy", "key_profile_dp"]
        key_budget, rows = select_shared_budget(
            key_validation, key_models, fold_id, "baseline", "key_profile")
        budget_rows.extend(rows)

        trained = []
        for encoder_name in args.encoders:
            for seed in args.seeds:
                print(f"--- {encoder_name}, seed {seed} ---")
                model, history, selection = train_metric(
                    train_examples, validation_examples, args, seed, device,
                    encoder_name=encoder_name)
                history.insert(0, "encoder", encoder_name)
                history.insert(0, "fold", fold_id)
                history_frames.append(history)
                distance = NeuralEmbeddingDistance(model.encoder, device)
                validation, _, _ = evaluate_distance_pair(
                    validation_cache, encoder_name, lambda _item, d=distance: d,
                    args.budgets, args, include_ted=False)
                trained.append({
                    "encoder": encoder_name, "seed": seed, "model": model,
                    "distance": distance, "validation": validation,
                    "selection": selection,
                })

        # One boundary budget is shared across both search algorithms and all
        # encoders for a given fold/seed.  This prevents the MLP/CNN ablation
        # from changing the number of reported predictions together with the
        # representation architecture.
        for seed in args.seeds:
            seed_records = [row for row in trained if row["seed"] == seed]
            validation = pd.concat(
                [row["validation"] for row in seed_records], ignore_index=True)
            model_names = [
                name for row in seed_records
                for name in (f'{row["encoder"]}_affinity_greedy',
                             f'{row["encoder"]}_dp')
            ]
            budget, rows = select_shared_budget(
                validation, model_names, fold_id, seed, "neural_shared")
            budget_rows.extend(rows)
            for record in seed_records:
                encoder_name = record["encoder"]
                selection = record["selection"]
                record["budget"] = budget
                checkpoint_path = (checkpoint_dir /
                                   f"fold_{fold_id}_{encoder_name}_{seed}.pt")
                torch.save({
                    "fold": fold_id, "seed": seed, "encoder": encoder_name,
                    "model_state_dict": state_dict_cpu(record["model"]),
                    "encoder_config": record["model"].encoder.architecture_config(),
                    "best_epoch": selection["best_epoch"],
                    "validation_work_macro_ap": selection[
                        "validation_work_macro_ap"],
                    "validation_selected_budget": budget,
                    "train_works": fold["works"]["train"],
                    "validation_works": fold["works"]["validation"],
                    "outer_test_works": fold["works"]["test"],
                }, checkpoint_path)
                annotation_event(
                    access_rows, fold_id, "metric_checkpoint_frozen", "validation",
                    f"{encoder_name}/{seed}; budget={budget}")

        annotation_event(access_rows, fold_id, "fold_model_selection_frozen",
                         "none", f"{len(trained)} neural checkpoints")
        # This is intentionally the first load_cache call involving outer-test
        # annotation paths in this fold.
        test_cache, test_status = load_cache(
            fold["pairs"]["test"], args.bin_size, args.max_bins)
        status_rows.extend({"fold": fold_id, "role": "test", **row}
                           for row in test_status)
        annotation_event(access_rows, fold_id, "outer_test_annotations_loaded",
                         "test", ",".join(fold["works"]["test"]))
        if not test_cache:
            raise RuntimeError(f"fold {fold_id} has an empty outer-test split")

        key_test, key_diagnostics, key_ted = evaluate_distance_pair(
            test_cache, "key_profile",
            lambda item: item["distances"]["key_profile"], [key_budget], args,
            include_ted=True)
        for seed in args.seeds:
            held_piece_frames.append(add_identity(
                selected_rows(key_test, key_budget), fold_id, seed, "handcrafted"))
            diagnostic_frames.append(add_identity(
                key_diagnostics, fold_id, seed, "handcrafted"))
            ted_frames.append(add_identity(key_ted, fold_id, seed, "handcrafted"))

        test_examples = deep_interval_examples(
            test_cache, args.contexts, args.split_seed + 200 + fold_id)
        for record in trained:
            encoder_name, seed = record["encoder"], record["seed"]
            test_scores, diagnostics, ted = evaluate_distance_pair(
                test_cache, encoder_name,
                lambda _item, d=record["distance"]: d,
                [record["budget"]], args, include_ted=True)
            held_piece_frames.append(add_identity(
                selected_rows(test_scores, record["budget"]),
                fold_id, seed, encoder_name))
            diagnostic_frames.append(add_identity(
                diagnostics, fold_id, seed, encoder_name))
            ted_frames.append(add_identity(ted, fold_id, seed, encoder_name))
            rows = metric_held_out_work_rows(
                record["model"], test_examples, device, seed)
            metric_test_rows.extend({"fold": fold_id, "encoder": encoder_name,
                                     **row} for row in rows)
        annotation_event(access_rows, fold_id, "outer_test_evaluation_complete",
                         "test")
        state["completed_folds"].append(fold_id)
        state["test_annotations_loaded"] = True
        save_json(output_dir / "experiment_state.json", state)

    held_piece = pd.concat(held_piece_frames, ignore_index=True)
    held_work = held_piece.groupby(
        ["fold", "seed", "encoder", "model", "work"], as_index=False).agg(
            f1=("f1", "mean"), precision=("precision", "mean"),
            recall=("recall", "mean"), boundary_ap=("boundary_ap", "mean"),
            runtime_seconds=("runtime_seconds", "mean"),
            n_pieces=("piece", "nunique"))
    per_seed, summary = summarize_held_out(held_work)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    ted = pd.concat(ted_frames, ignore_index=True)
    metric_history = pd.concat(history_frames, ignore_index=True)

    held_piece.to_csv(output_dir / "held_out_per_piece.csv", index=False)
    held_work.to_csv(output_dir / "held_out_per_work.csv", index=False)
    per_seed.to_csv(output_dir / "held_out_per_seed.csv", index=False)
    summary.to_csv(output_dir / "held_out_summary.csv", index=False)
    ablation_summary(held_work).to_csv(
        output_dir / "ablation_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "tree_diagnostics.csv", index=False)
    ted.to_csv(output_dir / "ted_auxiliary_per_piece.csv", index=False)
    metric_history.to_csv(output_dir / "metric_training_history.csv", index=False)
    pd.DataFrame(metric_test_rows).to_csv(
        output_dir / "metric_held_out_per_work.csv", index=False)
    pd.DataFrame(budget_rows).to_csv(
        output_dir / "validation_budget_selection.csv", index=False)
    pd.DataFrame(access_rows).to_csv(output_dir / "access_audit.csv", index=False)
    pd.DataFrame(status_rows).to_csv(output_dir / "run_status.csv", index=False)

    permutation_input = (held_work.groupby(["work", "model"], as_index=False).f1.mean()
                         .rename(columns={"model": "method"}))
    paired_permutation_tests(
        permutation_input, samples=args.permutation_samples,
        seed=args.split_seed, unit_column="work").to_csv(
            output_dir / "paired_work_tests.csv", index=False)
    plot_metric_histories(metric_history, output_dir)

    state.update({"phase": "complete", "test_annotations_loaded": True,
                  "n_outer_test_works": int(held_work.work.nunique())})
    save_json(output_dir / "experiment_state.json", state)
    print("\nComplete. Work-macro held-out summary:")
    print(summary.to_string(index=False))
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
