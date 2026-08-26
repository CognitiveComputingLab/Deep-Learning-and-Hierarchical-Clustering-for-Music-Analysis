#!/usr/bin/env python
"""Train a Siamese music distance and adjacent-merge REINFORCE policies."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
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
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from greedy_clustering import distance_specs, greedy_cluster
from greedy_evaluation import (
    boundary_scores, collect_prominent_splits, dcml_localkey_segments,
    tree_shape_diagnostics,
)
from neural_clustering import (
    AdjacentMergeEnvironment, BoundaryDistanceModel, MergePolicy,
    NeuralEmbeddingDistance, PitchClassEncoder, average_precision,
    boundary_average_precision, project_reference_boundaries,
    reference_boundary_indices, rollout_policy, state_dict_cpu,
    transpose_pitch_classes,
)
from ordered_affinity import (
    affinity_tree_revenue, greedy_adjacent_average_linkage,
    optimal_affinity_tree, pairwise_affinity,
)
from train_parametric_distance import discover_pairs, load_cache, split_works, work_id


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("external/ABC"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/deep_clustering"))
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260827, 20260828, 20260829])
    parser.add_argument("--split-seed", type=int, default=20260727)
    parser.add_argument("--train-works", type=int, default=10)
    parser.add_argument("--validation-works", type=int, default=3)
    parser.add_argument("--test-works", type=int, default=3)
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
    parser.add_argument("--imitation-epochs", type=int, default=20)
    parser.add_argument("--rl-epochs", type=int, default=100)
    parser.add_argument("--rl-patience", type=int, default=20)
    parser.add_argument("--policy-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-rl-lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def configure_quick(args):
    if not args.quick:
        return
    args.seeds = args.seeds[:1]
    args.train_works, args.validation_works, args.test_works = 3, 1, 2
    args.contexts = [1, 2]
    args.budgets = [3, 5]
    args.metric_epochs = min(args.metric_epochs, 4)
    args.metric_patience = min(args.metric_patience, 2)
    args.imitation_epochs = min(args.imitation_epochs, 2)
    args.rl_epochs = min(args.rl_epochs, 3)
    args.rl_patience = min(args.rl_patience, 2)
    args.validation_every = 1


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_interval_examples(cache, contexts, seed):
    """Balanced boundary/non-boundary interval pairs with work identifiers."""
    rng = np.random.default_rng(seed)
    examples = []
    for work in sorted({item["work"] for item in cache.values()}):
        positive, negative = [], []
        for piece, item in sorted(cache.items()):
            if item["work"] != work:
                continue
            matrix = item["matrix"]
            prefix = np.vstack([np.zeros(12), np.cumsum(matrix, axis=0)])
            bounds = np.asarray(item["bounds"], dtype=float)
            edges = np.r_[bounds[0, 0], bounds[:, 1]] if bounds.ndim == 2 else bounds
            gt = [end for _, end, _ in item["segments"][:-1]]
            marked = reference_boundary_indices(edges, gt)
            for split in range(1, len(matrix)):
                destination = positive if split in marked else negative
                for context in contexts:
                    if split - context >= 0 and split + context <= len(matrix):
                        destination.append({
                            "work": work, "piece": piece,
                            "left": prefix[split] - prefix[split - context],
                            "right": prefix[split + context] - prefix[split],
                            "label": int(split in marked), "context": context,
                        })
        count = min(len(positive), len(negative))
        if count:
            for indices, source in ((rng.choice(len(positive), count, replace=False), positive),
                                    (rng.choice(len(negative), count, replace=False), negative)):
                examples.extend(source[index] for index in sorted(indices))
    return examples


def example_arrays(examples):
    return (
        np.asarray([row["left"] for row in examples], dtype=np.float32),
        np.asarray([row["right"] for row in examples], dtype=np.float32),
        np.asarray([row["label"] for row in examples], dtype=np.float32),
        np.asarray([row["work"] for row in examples], dtype=object),
    )


def work_macro_ap(labels, scores, works):
    values = [average_precision(labels[works == work], scores[works == work])
              for work in sorted(set(works))]
    return float(np.mean(values)) if values else 0.0


def grouped_macro_mean(values, groups):
    """Average within groups first, then give every group equal weight."""
    frame = pd.DataFrame({"value": values, "group": groups})
    if frame.empty:
        return 0.0
    return float(frame.groupby("group", sort=True).value.mean().mean())


def metric_validation(model, examples, device):
    left, right, labels, works = example_arrays(examples)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(left).to(device),
                       torch.from_numpy(right).to(device)).cpu().numpy()
    return work_macro_ap(labels.astype(int), logits, works)


def train_metric(train_examples, validation_examples, args, seed, device):
    seed_everything(seed)
    model = BoundaryDistanceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.metric_lr,
                                  weight_decay=1e-4)
    left, right, labels, works = example_arrays(train_examples)
    counts = {work: int(np.sum(works == work)) for work in set(works)}
    weights = np.asarray([1.0 / counts[work] for work in works], dtype=np.float32)
    weights /= weights.mean()
    rng = np.random.default_rng(seed)
    history = []
    best_state, best_ap, best_epoch = None, -np.inf, 0
    stale = 0
    for epoch in range(1, args.metric_epochs + 1):
        model.train()
        order = rng.permutation(len(labels))
        losses = []
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            lt = torch.from_numpy(left[index]).to(device)
            rt = torch.from_numpy(right[index]).to(device)
            yt = torch.from_numpy(labels[index]).to(device)
            wt = torch.from_numpy(weights[index]).to(device)
            shifts = torch.randint(0, 12, (len(index),), device=device)
            lt = transpose_pitch_classes(lt, shifts)
            rt = transpose_pitch_classes(rt, shifts)
            optimizer.zero_grad()
            logits = model(lt, rt)
            loss = (F.binary_cross_entropy_with_logits(logits, yt, reduction="none") * wt).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_ap = metric_validation(model, validation_examples, device)
        history.append({"seed": seed, "epoch": epoch,
                        "loss": float(np.mean(losses)),
                        "validation_work_macro_ap": validation_ap})
        if validation_ap > best_ap + 1e-8:
            best_ap, best_epoch = validation_ap, epoch
            best_state = state_dict_cpu(model)
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or args.quick:
            print(f"[metric {seed} {epoch:03d}] loss={np.mean(losses):.5f} val_AP={validation_ap:.4f}")
        if stale >= args.metric_patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch,
                                           "validation_work_macro_ap": best_ap}


def select_epoch_pieces(cache, rng):
    groups = {}
    for piece, item in cache.items():
        groups.setdefault(item["work"], []).append(piece)
    return [rng.choice(sorted(groups[work])) for work in sorted(groups)]


def item_references(item):
    gt = [end for _, end, _ in item["segments"][:-1]]
    return reference_boundary_indices(item["bounds"], gt)


def boundary_projection_audit(cache, split, tolerance):
    """Describe discretisation error without exposing labels to inference."""
    rows = []
    for piece, item in sorted(cache.items()):
        gt = [end for _, end, _ in item["segments"][:-1]]
        bounds = np.asarray(item["bounds"], dtype=float)
        edges = np.r_[bounds[0, 0], bounds[:, 1]] if bounds.ndim == 2 else bounds
        internal = edges[1:-1]
        nearest = ([int(np.argmin(np.abs(internal - value))) + 1 for value in gt]
                   if len(internal) else [])
        projection = project_reference_boundaries(item["bounds"], gt)
        errors = np.asarray([row["absolute_error_qb"] for row in projection], dtype=float)
        rows.append({
            "split": split, "piece": piece, "work": item["work"],
            "gt_boundary_count": len(gt),
            "candidate_boundary_count": len(internal),
            "independent_nearest_unique_count": len(set(nearest)),
            "independent_nearest_collision_count": len(nearest) - len(set(nearest)),
            "projected_reference_count": len(projection),
            "unrepresented_reference_count": len(gt) - len(projection),
            "mean_projection_error_qb": float(errors.mean()) if len(errors) else 0.0,
            "max_projection_error_qb": float(errors.max()) if len(errors) else 0.0,
            "projected_within_tolerance_count": int(np.sum(errors <= tolerance)),
            "tolerance_qb": tolerance,
        })
    return rows


def train_imitation(encoder, train_cache, args, seed, device):
    seed_everything(seed + 1000)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    policy = MergePolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    rng = np.random.default_rng(seed + 1000)
    history = []
    for epoch in range(1, args.imitation_epochs + 1):
        losses = []
        for piece in select_epoch_pieces(train_cache, rng):
            item = train_cache[piece]
            environment = AdjacentMergeEnvironment(item["matrix"], item["bounds"])
            references = item_references(item)
            episode_losses = []
            while not environment.done:
                features = environment.candidate_features(encoder, device)
                logits = policy(features)
                log_probabilities = torch.log_softmax(logits, dim=0)
                safe = environment.safe_actions(references)
                if safe:
                    target = safe[int(torch.argmin(torch.linalg.vector_norm(
                        features[safe, :16] - features[safe, 16:32], dim=1)).item())]
                else:
                    target = int(torch.argmin(torch.linalg.vector_norm(
                        features[:, :16] - features[:, 16:32], dim=1)).item())
                episode_losses.append(-log_probabilities[target])
                environment.step(target)
            optimizer.zero_grad()
            loss = torch.stack(episode_losses).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"seed": seed, "variant": "imitation", "epoch": epoch,
                        "loss": float(np.mean(losses)), "validation_ap": np.nan})
        print(f"[imitation {seed} {epoch:03d}] loss={np.mean(losses):.5f}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    return policy, pd.DataFrame(history)


def validate_policy(encoder, policy, cache, device):
    encoder.eval(); policy.eval()
    rewards, works = [], []
    with torch.no_grad():
        for item in cache.values():
            rollout = rollout_policy(item["matrix"], item["bounds"], encoder, policy,
                                     deterministic=True, device=device)
            rewards.append(boundary_average_precision(
                rollout.root, item["bounds"], item_references(item)))
            works.append(item["work"])
    return grouped_macro_mean(rewards, works)


def train_reinforce(base_encoder, imitation_policy, train_cache, validation_cache,
                    args, seed, device, joint):
    encoder = copy.deepcopy(base_encoder).to(device)
    policy = copy.deepcopy(imitation_policy).to(device)
    parameters = [{"params": policy.parameters(), "lr": args.policy_lr}]
    if joint:
        parameters.append({"params": encoder.parameters(), "lr": args.encoder_rl_lr})
    else:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(parameters)
    rng = np.random.default_rng(seed + (3000 if joint else 2000))
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed + (3000 if joint else 2000))
    variant = "rl_joint" if joint else "rl_frozen"
    history = []
    best_encoder, best_policy, best_ap, best_epoch = None, None, -np.inf, 0
    stale = 0
    for epoch in range(1, args.rl_epochs + 1):
        encoder.eval(); policy.train()
        losses, rewards, baselines = [], [], []
        for piece in select_epoch_pieces(train_cache, rng):
            item = train_cache[piece]
            sampled = rollout_policy(item["matrix"], item["bounds"], encoder, policy,
                                     deterministic=False, device=device, generator=generator)
            reward = boundary_average_precision(sampled.root, item["bounds"],
                                                item_references(item))
            with torch.no_grad():
                baseline_rollout = rollout_policy(
                    item["matrix"], item["bounds"], encoder, policy,
                    deterministic=True, device=device)
                baseline = boundary_average_precision(
                    baseline_rollout.root, item["bounds"], item_references(item))
            advantage = reward - baseline
            loss = (-advantage * sampled.log_probability
                    - args.entropy_coefficient * sampled.mean_entropy)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            if joint:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            rewards.append(reward); baselines.append(baseline)
        validation_ap = np.nan
        if epoch % args.validation_every == 0 or epoch == args.rl_epochs:
            validation_ap = validate_policy(encoder, policy, validation_cache, device)
            if validation_ap > best_ap + 1e-8:
                best_ap, best_epoch = validation_ap, epoch
                best_encoder, best_policy = state_dict_cpu(encoder), state_dict_cpu(policy)
                stale = 0
            else:
                stale += args.validation_every
        history.append({
            "seed": seed, "variant": variant, "epoch": epoch,
            "loss": float(np.mean(losses)), "sampled_reward": float(np.mean(rewards)),
            "greedy_baseline_reward": float(np.mean(baselines)),
            "validation_ap": validation_ap,
        })
        if epoch == 1 or epoch % args.validation_every == 0:
            print(f"[{variant} {seed} {epoch:03d}] reward={np.mean(rewards):.4f} "
                  f"baseline={np.mean(baselines):.4f} val_AP={validation_ap:.4f}")
        if stale >= args.rl_patience:
            break
    if best_policy is None:
        best_encoder, best_policy = state_dict_cpu(encoder), state_dict_cpu(policy)
        best_ap, best_epoch = validate_policy(encoder, policy, validation_cache, device), epoch
    encoder.load_state_dict(best_encoder); policy.load_state_dict(best_policy)
    return encoder, policy, pd.DataFrame(history), {
        "best_epoch": best_epoch, "validation_work_macro_ap": best_ap,
    }


def build_tree(model, item, neural_distance, rl_models, device):
    matrix, bounds = item["matrix"], item["bounds"]
    started = time.perf_counter()
    if model == "euclidean_greedy":
        distance = distance_specs()[0]["euclidean"]
        tree = greedy_cluster(matrix, bounds, distance)
    elif model == "key_profile_greedy":
        distance = distance_specs()[0]["key_profile"]
        tree = greedy_cluster(matrix, bounds, distance)
    elif model == "key_profile_affinity_greedy":
        distance = distance_specs()[0]["key_profile"]
        affinity, affinity_info = pairwise_affinity(matrix, distance)
        tree = greedy_adjacent_average_linkage(matrix, bounds, affinity, ClusterNode)
        revenue, normalized = affinity_tree_revenue(tree, affinity)
        tree.shared_objective_revenue = revenue
        tree.shared_normalized_revenue = normalized
        tree.affinity_scale = affinity_info["scale"]
    elif model == "key_profile_dp":
        distance = distance_specs()[0]["key_profile"]
        affinity, affinity_info = pairwise_affinity(matrix, distance)
        tree, information = optimal_affinity_tree(matrix, bounds, affinity, ClusterNode)
        tree.shared_objective_revenue = information.total_revenue
        tree.shared_normalized_revenue = information.normalized_revenue
        tree.affinity_scale = affinity_info["scale"]
    elif model == "siamese_greedy":
        tree = greedy_cluster(matrix, bounds, neural_distance)
    elif model == "siamese_affinity_greedy":
        affinity, affinity_info = pairwise_affinity(matrix, neural_distance)
        tree = greedy_adjacent_average_linkage(matrix, bounds, affinity, ClusterNode)
        revenue, normalized = affinity_tree_revenue(tree, affinity)
        tree.shared_objective_revenue = revenue
        tree.shared_normalized_revenue = normalized
        tree.affinity_scale = affinity_info["scale"]
    elif model == "siamese_dp":
        affinity, affinity_info = pairwise_affinity(matrix, neural_distance)
        tree, information = optimal_affinity_tree(matrix, bounds, affinity, ClusterNode)
        tree.shared_objective_revenue = information.total_revenue
        tree.shared_normalized_revenue = information.normalized_revenue
        tree.affinity_scale = affinity_info["scale"]
    elif model in rl_models:
        encoder, policy = rl_models[model]
        with torch.no_grad():
            tree = rollout_policy(matrix, bounds, encoder, policy,
                                  deterministic=True, device=device).root
    else:
        raise ValueError(f"unknown model {model}")
    return tree, time.perf_counter() - started


# Imported here to keep the model-building cases above easy to scan.
from greedy_clustering import ClusterNode


MODELS = [
    "euclidean_greedy", "key_profile_greedy", "key_profile_affinity_greedy",
    "key_profile_dp", "siamese_greedy", "siamese_affinity_greedy",
    "siamese_dp", "rl_frozen", "rl_joint",
]

SHARED_BUDGET_GROUPS = [
    ("key_profile_affinity_greedy", "key_profile_dp"),
    ("siamese_affinity_greedy", "siamese_dp"),
]


def evaluate_cache(cache, models, neural_distance, rl_models, budgets, tolerance, device):
    rows, diagnostics, trajectories = [], [], []
    for piece, item in sorted(cache.items()):
        gt = [end for _, end, _ in item["segments"][:-1]]
        references = item_references(item)
        piece_objectives = {}
        for model in models:
            tree, runtime = build_tree(model, item, neural_distance, rl_models, device)
            ap = boundary_average_precision(tree, item["bounds"], references)
            shape = tree_shape_diagnostics(tree)
            objective = float(getattr(tree, "shared_objective_revenue", np.nan))
            normalized = float(getattr(tree, "shared_normalized_revenue", np.nan))
            if np.isfinite(objective):
                piece_objectives[model] = objective
            diagnostics.append({
                "piece": piece, "work": item["work"], "model": model,
                "runtime_seconds": runtime,
                "shared_objective_revenue": objective,
                "shared_normalized_revenue": normalized,
                "affinity_scale": float(getattr(tree, "affinity_scale", np.nan)),
                **shape,
            })
            for budget in budgets:
                score = boundary_scores(collect_prominent_splits(tree, budget), gt, tolerance)
                rows.append({
                    "piece": piece, "work": item["work"], "model": model,
                    "budget": budget, "boundary_ap": ap, "runtime_seconds": runtime,
                    **{key: score[key] for key in ["tp", "fp", "fn", "precision", "recall", "f1"]},
                })
            if model in rl_models:
                encoder, policy = rl_models[model]
                with torch.no_grad():
                    rollout = rollout_policy(item["matrix"], item["bounds"], encoder, policy,
                                             deterministic=True, device=device)
                trajectories.extend({"piece": piece, "work": item["work"], "model": model,
                                     **row} for row in rollout.trajectory)
        for prefix in ("key_profile", "siamese"):
            greedy_name, dp_name = f"{prefix}_affinity_greedy", f"{prefix}_dp"
            if ({greedy_name, dp_name} <= set(piece_objectives)
                    and piece_objectives[dp_name] < piece_objectives[greedy_name] - 1e-8):
                raise AssertionError(
                    f"{dp_name} revenue is below {greedy_name} for {piece}")
    return pd.DataFrame(rows), pd.DataFrame(diagnostics), pd.DataFrame(trajectories)


def select_budgets(validation):
    work = validation.groupby(["model", "budget", "work"], as_index=False).f1.mean()
    summary = work.groupby(["model", "budget"], as_index=False).f1.mean()
    selected = {}
    for group in SHARED_BUDGET_GROUPS:
        available = [model for model in group if model in set(work.model)]
        if len(available) != len(group):
            continue
        shared = (work[work.model.isin(group)]
                  .groupby("budget", as_index=False).f1.mean()
                  .sort_values(["f1", "budget"], ascending=[False, True]))
        budget = int(shared.iloc[0].budget)
        selected.update({model: budget for model in group})
    for model, group in summary.groupby("model"):
        if model in selected:
            continue
        best = group.sort_values(["f1", "budget"], ascending=[False, True]).iloc[0]
        selected[model] = int(best.budget)
    return selected


def aggregate_selected(frame, selected, seed):
    selected_frame = pd.concat([
        frame[(frame.model == model) & (frame.budget == budget)]
        for model, budget in selected.items()
    ], ignore_index=True)
    selected_frame.insert(0, "seed", seed)
    work = selected_frame.groupby(["seed", "model", "work"], as_index=False).agg(
        f1=("f1", "mean"), precision=("precision", "mean"), recall=("recall", "mean"),
        boundary_ap=("boundary_ap", "mean"), runtime_seconds=("runtime_seconds", "mean"))
    return selected_frame, work


def summarize_held_out(work_frame):
    """Return per-seed work macros and mean/std across independent seeds."""
    per_seed = work_frame.groupby(["seed", "model"], as_index=False).agg(
        n_works=("work", "nunique"), work_macro_f1=("f1", "mean"),
        work_macro_precision=("precision", "mean"),
        work_macro_recall=("recall", "mean"),
        work_macro_boundary_ap=("boundary_ap", "mean"),
        mean_runtime_seconds=("runtime_seconds", "mean"))
    summary = per_seed.groupby("model", as_index=False).agg(
        n_seeds=("seed", "nunique"), n_works=("n_works", "min"),
        mean_f1=("work_macro_f1", "mean"), std_f1=("work_macro_f1", "std"),
        mean_precision=("work_macro_precision", "mean"),
        mean_recall=("work_macro_recall", "mean"),
        mean_boundary_ap=("work_macro_boundary_ap", "mean"),
        std_boundary_ap=("work_macro_boundary_ap", "std"),
        mean_runtime_seconds=("mean_runtime_seconds", "mean"))
    return per_seed, summary


def make_ablation_summary(work_frame):
    means = work_frame.groupby(["seed", "model"], as_index=False).f1.mean()
    pivot = means.pivot(index="seed", columns="model", values="f1")
    comparisons = [
        ("siamese_vs_key_profile_aggregate_greedy", "siamese_greedy", "key_profile_greedy"),
        ("siamese_vs_key_profile_affinity_greedy", "siamese_affinity_greedy", "key_profile_affinity_greedy"),
        ("siamese_dp_vs_key_profile_dp", "siamese_dp", "key_profile_dp"),
        ("siamese_dp_vs_affinity_greedy", "siamese_dp", "siamese_affinity_greedy"),
        ("rl_frozen_vs_siamese_affinity_greedy", "rl_frozen", "siamese_affinity_greedy"),
        ("rl_joint_vs_siamese_dp", "rl_joint", "siamese_dp"),
        ("rl_joint_vs_frozen", "rl_joint", "rl_frozen"),
    ]
    rows = []
    for label, left, right in comparisons:
        if left in pivot and right in pivot:
            values = pivot[left] - pivot[right]
            rows.append({"comparison": label, "mean_f1_difference": values.mean(),
                         "std_across_seeds": values.std(ddof=1) if len(values) > 1 else 0.0,
                         "n_seeds": len(values)})
    return pd.DataFrame(rows)


def metric_held_out_work_rows(model, examples, device, seed):
    left, right, labels, works = example_arrays(examples)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(left).to(device),
                       torch.from_numpy(right).to(device)).cpu().numpy()
    return [{"seed": seed, "work": work,
             "boundary_classification_ap": average_precision(
                 labels[works == work].astype(int), logits[works == work])}
            for work in sorted(set(works))]


def plot_histories(metric_history, rl_history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for seed, group in metric_history.groupby("seed"):
        axes[0].plot(group.epoch, group.validation_work_macro_ap, label=str(seed))
    axes[0].set(title="Siamese validation AP", xlabel="epoch", ylabel="work-macro AP")
    for (seed, variant), group in rl_history[rl_history.variant != "imitation"].groupby(["seed", "variant"]):
        axes[1].plot(group.epoch, group.validation_ap, label=f"{variant}-{seed}")
    axes[1].set(title="RL validation AP", xlabel="epoch", ylabel="work-macro AP")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    args = parse_args(); configure_quick(args)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    access_events = []
    completed_seeds = []

    def write_experiment_state(phase):
        state = {
            "phase": phase, "completed_seeds": completed_seeds,
            "requested_seeds": args.seeds, "device": str(device),
            "quick": args.quick,
        }
        (args.output_dir / "experiment_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8")

    def record_access(event, *, split="", seed=""):
        access_events.append({"sequence": len(access_events) + 1, "event": event,
                              "split": split, "seed": seed})

    write_experiment_state("initialising")
    pairs = discover_pairs(args.data_root)
    if args.quick:
        selected = sorted({work_id(piece) for piece, _, _ in pairs})[:6]
        pairs = [min((row for row in pairs if work_id(row[0]) == work), key=lambda row: row[0])
                 for work in selected]
    splits, split_names = split_works(
        pairs, args.split_seed, args.train_works, args.validation_works, args.test_works)
    caches, status_rows = {}, []
    projection_rows = []
    # Load only development annotations before model and budget selection.  The
    # test harmonies are deliberately not parsed until every seed is frozen.
    for split in ("train", "validation"):
        split_pairs = splits[split]
        caches[split], current = load_cache(split_pairs, args.bin_size, args.max_bins)
        record_access("annotations_loaded", split=split)
        status_rows.extend({"split": split, **row} for row in current)
        projection_rows.extend(boundary_projection_audit(
            caches[split], split, args.tolerance))
    pd.DataFrame(status_rows).to_csv(args.output_dir / "run_status.csv", index=False)
    assignment = [{"split": split, "work": work_id(piece), "piece": piece}
                  for split, rows in splits.items() for piece, _, _ in rows]
    pd.DataFrame(assignment).to_csv(args.output_dir / "data_split.csv", index=False)
    if any(not caches[name] for name in ("train", "validation")):
        raise SystemExit("A development split is empty; inspect run_status.csv")
    write_experiment_state("development_data_loaded")

    train_examples = deep_interval_examples(caches["train"], args.contexts, args.split_seed)
    validation_examples = deep_interval_examples(caches["validation"], args.contexts,
                                                 args.split_seed + 1)
    if not train_examples or not validation_examples:
        raise SystemExit("Metric-learning examples could not be constructed")

    metric_histories, rl_histories = [], []
    test_rows, test_work_rows, diagnostic_rows, trajectory_rows = [], [], [], []
    metric_test_rows = []
    model_records = {}
    trained_runs = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} on {device} ===")
        metric, metric_history, metric_info = train_metric(
            train_examples, validation_examples, args, seed, device)
        pretrained_encoder = copy.deepcopy(metric.encoder).to(device)
        imitation_policy, imitation_history = train_imitation(
            pretrained_encoder, caches["train"], args, seed, device)
        frozen_encoder, frozen_policy, frozen_history, frozen_info = train_reinforce(
            pretrained_encoder, imitation_policy, caches["train"], caches["validation"],
            args, seed, device, joint=False)
        joint_encoder, joint_policy, joint_history, joint_info = train_reinforce(
            pretrained_encoder, imitation_policy, caches["train"], caches["validation"],
            args, seed, device, joint=True)
        metric_histories.append(metric_history)
        rl_histories.extend([imitation_history, frozen_history, joint_history])

        neural_distance = NeuralEmbeddingDistance(metric.encoder, device)
        rl_models = {
            "rl_frozen": (frozen_encoder, frozen_policy),
            "rl_joint": (joint_encoder, joint_policy),
        }
        validation, _, _ = evaluate_cache(
            caches["validation"], MODELS, neural_distance, rl_models,
            args.budgets, args.tolerance, device)
        budgets = select_budgets(validation)
        checkpoint = {
            "seed": seed, "metric": state_dict_cpu(metric),
            "rl_frozen_encoder": state_dict_cpu(frozen_encoder),
            "rl_frozen_policy": state_dict_cpu(frozen_policy),
            "rl_joint_encoder": state_dict_cpu(joint_encoder),
            "rl_joint_policy": state_dict_cpu(joint_policy),
            "selected_budgets": budgets,
        }
        torch.save(checkpoint, args.output_dir / f"checkpoint_seed_{seed}.pt")
        record_access("checkpoint_and_budgets_frozen", seed=seed)
        model_records[str(seed)] = {
            "metric": metric_info, "rl_frozen": frozen_info,
            "rl_joint": joint_info, "selected_budgets": budgets,
        }
        trained_runs.append({
            "seed": seed, "metric": metric, "neural_distance": neural_distance,
            "rl_models": rl_models, "budgets": budgets,
        })
        completed_seeds.append(seed)
        write_experiment_state("training")

    # This is the single held-out phase. Every seed checkpoint and every
    # validation-selected boundary budget has already been frozen.
    record_access("all_checkpoints_and_budgets_frozen")
    write_experiment_state("all_checkpoints_and_budgets_frozen")
    record_access("held_out_phase_started", split="test")
    caches["test"], current = load_cache(
        splits["test"], args.bin_size, args.max_bins)
    record_access("annotations_loaded", split="test")
    status_rows.extend({"split": "test", **row} for row in current)
    pd.DataFrame(status_rows).to_csv(args.output_dir / "run_status.csv", index=False)
    if not caches["test"]:
        raise SystemExit("The held-out split is empty; inspect run_status.csv")
    write_experiment_state("held_out_data_loaded")
    projection_rows.extend(boundary_projection_audit(
        caches["test"], "test", args.tolerance))
    test_examples = deep_interval_examples(caches["test"], args.contexts,
                                           args.split_seed + 2)
    for run in trained_runs:
        seed = run["seed"]
        held_out, diagnostics, trajectories = evaluate_cache(
            caches["test"], MODELS, run["neural_distance"], run["rl_models"],
            args.budgets, args.tolerance, device)
        selected, work = aggregate_selected(held_out, run["budgets"], seed)
        test_rows.append(selected); test_work_rows.append(work)
        diagnostics.insert(0, "seed", seed); diagnostic_rows.append(diagnostics)
        if not trajectories.empty:
            trajectories.insert(0, "seed", seed); trajectory_rows.append(trajectories)
        metric_test_rows.extend(metric_held_out_work_rows(
            run["metric"], test_examples, device, seed))

    metric_history = pd.concat(metric_histories, ignore_index=True)
    rl_history = pd.concat(rl_histories, ignore_index=True)
    held_out = pd.concat(test_rows, ignore_index=True)
    held_out_work = pd.concat(test_work_rows, ignore_index=True)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    trajectories = (pd.concat(trajectory_rows, ignore_index=True)
                    if trajectory_rows else pd.DataFrame())
    metric_held_out = pd.DataFrame(metric_test_rows)
    per_seed, summary = summarize_held_out(held_out_work)
    ablation = make_ablation_summary(held_out_work)
    metric_history.to_csv(args.output_dir / "metric_training_history.csv", index=False)
    rl_history.to_csv(args.output_dir / "rl_training_history.csv", index=False)
    held_out.to_csv(args.output_dir / "held_out_per_piece.csv", index=False)
    held_out_work.to_csv(args.output_dir / "held_out_per_work.csv", index=False)
    per_seed.to_csv(args.output_dir / "held_out_per_seed.csv", index=False)
    summary.to_csv(args.output_dir / "held_out_summary.csv", index=False)
    ablation.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    diagnostics.to_csv(args.output_dir / "tree_diagnostics.csv", index=False)
    trajectories.to_csv(args.output_dir / "action_trajectories.csv", index=False)
    metric_held_out.to_csv(args.output_dir / "metric_held_out_per_work.csv", index=False)
    pd.DataFrame(projection_rows).to_csv(
        args.output_dir / "boundary_projection_audit.csv", index=False)
    pd.DataFrame(access_events).to_csv(
        args.output_dir / "access_audit.csv", index=False)
    plot_histories(metric_history, rl_history, args.output_dir)
    config = {
        "method_scope": "Advanced extension: learned proxy, not a true expert hierarchy",
        "test_access_policy": "Test annotations loaded only after all seed checkpoints and budgets were frozen",
        "rl_validation_aggregation": "movement AP averaged within work, then macro-averaged across works",
        "held_out_seed_aggregation": "work-macro metrics computed per seed before mean/std across seeds",
        "boundary_projection": "minimum-error order-preserving unique projection to internal bin edges",
        "shared_budget_groups": SHARED_BUDGET_GROUPS,
        "split_seed": args.split_seed, "splits": split_names,
        "model_seeds": args.seeds, "device": str(device),
        "contexts": args.contexts, "budgets": args.budgets,
        "bin_size_qb": args.bin_size, "tolerance_qb": args.tolerance,
        "training_examples": len(train_examples), "models": model_records,
        "quick": args.quick,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_experiment_state("complete")
    print("\nHELD-OUT SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
