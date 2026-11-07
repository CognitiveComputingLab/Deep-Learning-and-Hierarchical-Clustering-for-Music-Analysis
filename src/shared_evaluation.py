"""Adapters from shared annotations to existing boundary/tree evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from shared_annotation_format import SharedAnnotation, SharedPiece, iter_boundaries


@dataclass
class ReferenceTreeNode:
    label: str
    annotation_id: str
    start: float | tuple[int, float]
    end: float | tuple[int, float]
    children: list["ReferenceTreeNode"] = field(default_factory=list)


def boundary_reference(piece: SharedPiece, *, layers: Iterable[str] | None = None,
                       max_level: int | None = None) -> list[float]:
    """Quarterbeat boundaries compatible with current Boundary F1 code."""
    if piece.coordinate_system not in {"quarterbeat", "mixed"}:
        raise ValueError(
            f"{piece.piece_id} has {piece.coordinate_system} coordinates; "
            "align it to a symbolic score before pitch-based Boundary F1")
    return iter_boundaries(piece, layers=layers, max_level=max_level)


def _node_position(item: SharedAnnotation, end: bool = False):
    qb = item.end_qb if end else item.start_qb
    if qb is not None:
        return float(qb)
    measure = item.end_measure if end else item.start_measure
    beat = item.end_beat if end else item.start_beat
    return (int(measure), float(beat or 0.0))


def reference_forest(piece: SharedPiece, tree_id: str) -> list[ReferenceTreeNode]:
    """Build a forest only from explicit hierarchy claims in the source."""
    items = [item for item in piece.annotations
             if item.kind == "hierarchical_segment" and item.tree_id == tree_id]
    nodes = {
        item.annotation_id: ReferenceTreeNode(
            label=item.label, annotation_id=item.annotation_id,
            start=_node_position(item), end=_node_position(item, end=True))
        for item in items
    }
    roots = []
    for item in items:
        node = nodes[item.annotation_id]
        if item.parent_id and item.parent_id in nodes:
            nodes[item.parent_id].children.append(node)
        else:
            roots.append(node)
    for node in nodes.values():
        node.children.sort(key=lambda child: child.start)
    return sorted(roots, key=lambda node: node.start)


def algorithm_readiness(piece: SharedPiece) -> dict[str, object]:
    """Explain which current evaluations can use a converted piece."""
    quarterbeat = piece.coordinate_system in {"quarterbeat", "mixed"}
    score = bool(piece.score_path)
    trees = sorted({item.tree_id for item in piece.annotations
                    if item.kind == "hierarchical_segment" and item.tree_id})
    boundary_layers = sorted({item.layer for item in piece.annotations
                              if item.kind in {"partition_segment", "hierarchical_segment"}
                              and item.start_qb is not None})
    return {
        "pitch_clustering_ready": quarterbeat and score,
        "boundary_evaluation_ready": quarterbeat and bool(boundary_layers),
        "hierarchical_evaluation_ready": bool(trees),
        "tree_ids": trees,
        "quarterbeat_boundary_layers": boundary_layers,
        "blocking_reason": "" if quarterbeat and score else (
            "matching score/features missing" if quarterbeat else
            "measure/beat-to-quarterbeat alignment missing"),
    }
