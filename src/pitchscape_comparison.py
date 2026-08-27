"""Comparable Pitch Scapes for handcrafted Greedy/DP and learned-metric DP.

The inference functions in this module receive notes/features only.  DCML
annotations are loaded separately by the command-line renderer, after a formal
checkpoint has been selected, and are used only in the companion comparison
figure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from dp_clustering import assert_valid_ordered_binary_tree
from dp_pitchscape import collect_tree_geometry
from greedy_clustering import ClusterNode, distance_specs, load_pc_bins, to_float_qb
from greedy_evaluation import tree_shape_diagnostics
from neural_clustering import (
    BoundaryDistanceModel,
    NeuralEmbeddingDistance,
    build_pitch_class_encoder,
)
from ordered_affinity import (
    affinity_tree_revenue,
    greedy_adjacent_average_linkage,
    optimal_affinity_tree,
    pairwise_affinity,
)


@dataclass(frozen=True)
class SelectedPiece:
    piece_id: str
    directory_name: str
    display_name: str
    rationale: str


SELECTED_PIECES: tuple[SelectedPiece, ...] = (
    SelectedPiece(
        "n01op18-1_01", "Beethoven_Op18_No1_Movement1",
        "Beethoven, String Quartet Op. 18 No. 1, Movement 1",
        "Early-period large first movement",
    ),
    SelectedPiece(
        "n07op59-1_01", "Beethoven_Op59_No1_Movement1",
        "Beethoven, String Quartet Op. 59 No. 1, Movement 1",
        "Expansive middle-period first movement",
    ),
    SelectedPiece(
        "n11op95_01", "Beethoven_Op95_Movement1",
        "Beethoven, String Quartet Op. 95, Movement 1",
        "Compact F-minor middle/late-period transition",
    ),
    SelectedPiece(
        "n14op131_01", "Beethoven_Op131_Movement1",
        "Beethoven, String Quartet Op. 131, Movement 1",
        "Late C-sharp-minor fugue",
    ),
    SelectedPiece(
        "n16op135_04", "Beethoven_Op135_Movement4",
        "Beethoven, String Quartet Op. 135, Movement 4",
        "Late finale with contrasting tonal regions",
    ),
)

METHOD_LABELS = {
    "greedy": "Key-profile affinity + adjacent Greedy",
    "dp": "Key-profile affinity + exact ordered DP",
    "dl": "Harmonic CNN distance + exact ordered DP",
}


@dataclass(frozen=True)
class FormalMetricCheckpoint:
    seed: int
    validation_ap: float
    path: Path
    sha256: str
    encoder_config: dict[str, Any]
    distance: NeuralEmbeddingDistance


def resolve_device(value: str | torch.device) -> torch.device:
    if str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required formal-run file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_formal_metric_checkpoint(
    checkpoint_dir: str | Path,
    device: str | torch.device = "auto",
) -> FormalMetricCheckpoint:
    """Select the formal harmonic-CNN seed using validation AP only."""
    checkpoint_dir = Path(checkpoint_dir).resolve()
    state = _read_json(checkpoint_dir / "experiment_state.json")
    config = _read_json(checkpoint_dir / "config.json")
    if state.get("phase") != "complete":
        raise RuntimeError(
            "formal deep experiment is not complete: "
            f"phase={state.get('phase')!r}")
    if bool(state.get("quick")) or bool(config.get("quick")):
        raise RuntimeError("quick/smoke checkpoints cannot produce formal figures")
    requested = [int(value) for value in state.get("requested_seeds", [])]
    completed = [int(value) for value in state.get("completed_seeds", [])]
    configured = [int(value) for value in config.get("model_seeds", [])]
    if len(requested) != 3 or sorted(set(requested)) != sorted(set(completed)):
        raise RuntimeError("three requested seeds must all be complete")
    if configured and sorted(set(configured)) != sorted(set(requested)):
        raise RuntimeError("experiment state and config disagree about model seeds")

    selected: tuple[float, int, Path, dict[str, Any]] | None = None
    torch_device = resolve_device(device)
    for seed in sorted(requested):
        path = checkpoint_dir / f"checkpoint_seed_{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"formal checkpoint is missing: {path}")
        saved = torch.load(path, map_location=torch_device, weights_only=False)
        encoder_config = dict(saved.get("encoder_config", {}))
        architecture = encoder_config.get("name", "")
        if architecture != "circular_harmonic_cnn":
            raise RuntimeError(
                f"checkpoint {path.name} is not the formal harmonic CNN: "
                f"{architecture!r}")
        try:
            validation_ap = float(
                saved["selection"]["metric"]["validation_work_macro_ap"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"checkpoint {path.name} has no metric validation AP") from error
        if not np.isfinite(validation_ap):
            raise RuntimeError(f"checkpoint {path.name} has non-finite validation AP")
        candidate = (validation_ap, -seed, path, saved)
        if selected is None or candidate[:2] > selected[:2]:
            selected = candidate

    assert selected is not None
    validation_ap, negative_seed, path, saved = selected
    seed = -negative_seed
    encoder_config = dict(saved["encoder_config"])
    metric = BoundaryDistanceModel(
        build_pitch_class_encoder(encoder_config["name"])).to(torch_device)
    metric.load_state_dict(saved["metric"])
    metric.eval()
    return FormalMetricCheckpoint(
        seed=seed,
        validation_ap=validation_ap,
        path=path,
        sha256=file_sha256(path),
        encoder_config=encoder_config,
        distance=NeuralEmbeddingDistance(metric.encoder, torch_device),
    )


def notes_tsv_to_pitchscape_values(
    notes_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Chordify a DCML notes TSV on its exact notated quarter-beat timeline."""
    notes_path = Path(notes_path)
    notes = pd.read_csv(notes_path, sep="\t").dropna(
        subset=["quarterbeats", "duration_qb", "midi"])
    if notes.empty:
        raise ValueError(f"no timed notes in {notes_path}")

    changes: dict[float, np.ndarray] = defaultdict(
        lambda: np.zeros(12, dtype=float))
    maximum = 0.0
    for row in notes.itertuples():
        start = float(to_float_qb(row.quarterbeats))
        duration = float(row.duration_qb)
        end = start + duration
        if not np.isfinite(start) or not np.isfinite(duration) or start < 0:
            raise ValueError(f"invalid note timing in {notes_path}")
        # DCML occasionally retains zero-duration grace-note placeholders.
        # They contribute no temporal pitch mass to clustering or Pitch Scapes.
        if duration == 0:
            continue
        if duration < 0 or end <= start:
            raise ValueError(f"invalid note timing in {notes_path}")
        pitch_class = int(row.midi) % 12
        changes[start][pitch_class] += 1.0
        changes[end][pitch_class] -= 1.0
        maximum = max(maximum, end)
    if maximum <= 0:
        raise ValueError(f"no positive-duration timed notes in {notes_path}")
    changes[0.0] += 0.0
    changes[maximum] += 0.0
    times = np.asarray(sorted(changes), dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"could not construct a timeline from {notes_path}")

    current = np.zeros(12, dtype=float)
    values = np.zeros((len(times) - 1, 12), dtype=float)
    for index, start in enumerate(times[:-1]):
        current = current + changes[float(start)]
        current[np.abs(current) < 1e-12] = 0.0
        if np.any(current < 0):
            raise ValueError(f"negative active-note count at {start:g} in {notes_path}")
        values[index] = current
    if not np.any(values):
        raise ValueError(f"all pitch-class intervals are empty in {notes_path}")
    return values, times


def make_pitchscape(notes_path: str | Path) -> Any:
    """Build a PitchScape without MIDI/MuseScore conversion."""
    from pitchscapes.scapes import PitchScape

    values, times = notes_tsv_to_pitchscape_values(notes_path)
    return PitchScape(values=values, times=times)


def select_resolution(
    notes_path: str | Path,
    candidates: Sequence[float] = (8.0, 16.0, 32.0),
    max_leaves: int = 192,
) -> tuple[np.ndarray, list[tuple[float, float]], float]:
    if max_leaves < 2:
        raise ValueError("max_leaves must be at least two")
    candidates = tuple(sorted(float(value) for value in candidates))
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("resolution candidates must be positive")
    counts: list[tuple[float, int]] = []
    for bin_size in candidates:
        matrix, bounds = load_pc_bins(notes_path, bin_size)
        counts.append((bin_size, len(matrix)))
        if len(matrix) <= max_leaves:
            return matrix, bounds, bin_size
    description = ", ".join(f"{size:g}qb={count}" for size, count in counts)
    raise ValueError(
        f"no registered resolution has <= {max_leaves} leaves ({description})")


def build_comparison_trees(
    matrix: np.ndarray,
    bounds: Sequence[tuple[float, float]],
    neural_distance: NeuralEmbeddingDistance,
    *,
    max_leaves: int = 192,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the three pre-registered trees from identical leaf features."""
    matrix = np.asarray(matrix, dtype=float)
    if len(matrix) > max_leaves:
        raise ValueError(f"{len(matrix)} leaves exceeds max_leaves={max_leaves}")
    handcrafted = distance_specs(matrix.sum(axis=0))[0]["key_profile"]
    handcrafted_affinity, handcrafted_info = pairwise_affinity(matrix, handcrafted)
    neural_affinity, neural_info = pairwise_affinity(matrix, neural_distance)

    started = time.perf_counter()
    greedy = greedy_adjacent_average_linkage(
        matrix, bounds, handcrafted_affinity, ClusterNode)
    greedy_seconds = time.perf_counter() - started
    greedy_revenue = affinity_tree_revenue(greedy, handcrafted_affinity)

    dp, dp_info = optimal_affinity_tree(
        matrix, bounds, handcrafted_affinity, ClusterNode,
        tie_break="midpoint", max_bins=max_leaves)
    learned, learned_info = optimal_affinity_tree(
        matrix, bounds, neural_affinity, ClusterNode,
        tie_break="midpoint", max_bins=max_leaves)
    trees = {"greedy": greedy, "dp": dp, "dl": learned}
    for tree in trees.values():
        assert_valid_ordered_binary_tree(tree, bounds)

    diagnostics = {
        "greedy": {
            "runtime_seconds": greedy_seconds,
            "objective_revenue": greedy_revenue[0],
            "normalized_objective_revenue": greedy_revenue[1],
            "affinity": handcrafted_info,
            **tree_shape_diagnostics(greedy),
        },
        "dp": {
            "runtime_seconds": dp_info.elapsed_seconds,
            "objective_revenue": dp_info.total_revenue,
            "normalized_objective_revenue": dp_info.normalized_revenue,
            "affinity": handcrafted_info,
            **tree_shape_diagnostics(dp),
        },
        "dl": {
            "runtime_seconds": learned_info.elapsed_seconds,
            "objective_revenue": learned_info.total_revenue,
            "normalized_objective_revenue": learned_info.normalized_revenue,
            "affinity": neural_info,
            **tree_shape_diagnostics(learned),
        },
    }
    return trees, diagnostics


def tree_node_rows(root: Any, method: str) -> list[dict[str, Any]]:
    """Serialize every tree node with stable preorder identifiers."""
    rows: list[dict[str, Any]] = []
    counter = [0]

    def walk(node: Any, parent_id: int | None, depth: int) -> int:
        node_id = counter[0]
        counter[0] += 1
        row = {
            "method": method,
            "node_id": node_id,
            "parent_id": "" if parent_id is None else parent_id,
            "depth": depth,
            "start_qb": float(node.start),
            "end_qb": float(node.end),
            "center_qb": (float(node.start) + float(node.end)) / 2.0,
            "span_qb": float(node.end) - float(node.start),
            "merge_order": int(getattr(node, "merge_order", -1)),
            "is_leaf": not bool(list(getattr(node, "children", []) or [])),
        }
        rows.append(row)
        for child in list(getattr(node, "children", []) or []):
            walk(child, node_id, depth + 1)
        return node_id

    walk(root, None, 0)
    total = float(root.end) - float(root.start)
    for row in rows:
        row["center_normalized"] = (
            (float(row["center_qb"]) - float(root.start)) / total)
        row["span_normalized"] = float(row["span_qb"]) / total
    return rows


def draw_tree_on_pitchscape(
    axis: Any,
    scape: Any,
    root: Any,
    *,
    method: str,
    max_depth: int = 6,
    n_samples: int = 200,
    expert_boundaries_qb: Iterable[float] = (),
) -> tuple[int, int]:
    import pitchscapes.plotting as plotting

    plotting.key_scape_plot(scape=scape, n_samples=n_samples, ax=axis)
    total = float(root.end) - float(root.start)
    points, edges = collect_tree_geometry(root, max_depth)
    for parent, child in edges:
        x1 = (parent[0] - float(root.start)) / total
        y1 = parent[1] / total
        x2 = (child[0] - float(root.start)) / total
        y2 = child[1] / total
        axis.plot([x1, x2], [y1, y2], color="white", linewidth=1.1,
                  alpha=0.82, zorder=5)
    for center, width, _ in points:
        x = (center - float(root.start)) / total
        y = width / total
        axis.plot(x, y, "o", color="black", markersize=4.5,
                  markeredgecolor="white", markeredgewidth=0.7, zorder=10)
    for boundary in sorted(set(float(value) for value in expert_boundaries_qb)):
        if float(root.start) < boundary < float(root.end):
            x = (boundary - float(root.start)) / total
            axis.axvline(x, color="#ffd43b", linestyle="--", linewidth=0.8,
                        alpha=0.55, zorder=4)
    axis.set_title(METHOD_LABELS[method])
    return len(points), len(edges)
