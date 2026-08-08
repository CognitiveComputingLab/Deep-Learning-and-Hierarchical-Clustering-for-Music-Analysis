'''Evaluation helpers for greedy-versus-DP experiments.

Boundary F1 is the primary metric because DCML local-key annotations provide
boundaries, not hierarchical ground-truth trees. TED is auxiliary and compares
the predicted hierarchy with the flat tree induced by those segments.
'''

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


@dataclass
class EvaluationResult:
    piece: str
    work: str
    search: str
    method: str
    bin_size_qb: float
    tolerance_qb: float
    depth: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    predicted_boundary_count: int
    gt_boundary_count: int
    raw_ted: float
    pruned_ted: float
    normalized_raw_ted: float
    normalized_pruned_ted: float
    node_count_scaled_raw_ted: float
    node_count_scaled_pruned_ted: float
    predicted_node_count: int
    gt_node_count: int
    pruned_node_count: int
    pruned_leaf_count: int
    target_segment_count: int
    selected_pruning_depth: int
    objective_cost: float
    runtime_seconds: float
    evaluated_splits: int = 0
    tie_count: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def work_id(piece: str) -> str:
    return piece.rsplit('_', 1)[0] if '_' in piece else piece


def evaluate_tree(*, piece: str, search: str, method: str, tree: Any,
                  segments: Sequence[tuple[float, float, str]], total: float,
                  bin_size_qb: float, tolerance_qb: float, depth: int,
                  objective_cost: float, runtime_seconds: float,
                  evaluated_splits: int = 0, tie_count: int = 0,
                  center_bins: int = 10, width_bins: int = 5) -> EvaluationResult:
    from greedy_evaluation import boundary_scores, collect_top_splits, ted_diagnostics

    reference = [end for _, end, _ in segments[:-1]]
    predicted = collect_top_splits(tree, depth)
    score = boundary_scores(predicted, reference, tolerance_qb)
    ted = ted_diagnostics(tree, segments, total, center_bins, width_bins)
    return EvaluationResult(
        piece=piece, work=work_id(piece), search=search, method=method,
        bin_size_qb=float(bin_size_qb), tolerance_qb=float(tolerance_qb),
        depth=int(depth), tp=int(score['tp']), fp=int(score['fp']), fn=int(score['fn']),
        precision=float(score['precision']), recall=float(score['recall']),
        f1=float(score['f1']),
        predicted_boundary_count=int(score['predicted_boundary_count']),
        gt_boundary_count=int(score['gt_boundary_count']), raw_ted=float(ted['raw_ted']),
        pruned_ted=float(ted['pruned_ted']),
        normalized_raw_ted=float(ted['normalized_raw_ted']),
        normalized_pruned_ted=float(ted['normalized_pruned_ted']),
        node_count_scaled_raw_ted=float(ted['node_count_scaled_raw_ted']),
        node_count_scaled_pruned_ted=float(ted['node_count_scaled_pruned_ted']),
        predicted_node_count=int(ted['predicted_node_count']),
        gt_node_count=int(ted['gt_node_count']),
        pruned_node_count=int(ted['pruned_node_count']),
        pruned_leaf_count=int(ted['pruned_leaf_count']),
        target_segment_count=int(ted['target_segment_count']),
        selected_pruning_depth=int(ted['selected_pruning_depth']),
        objective_cost=float(objective_cost), runtime_seconds=float(runtime_seconds),
        evaluated_splits=int(evaluated_splits), tie_count=int(tie_count))


def print_result_table(results: Sequence[EvaluationResult], title: str) -> None:
    print('\n' + title)
    print(f"{'search':9s} {'method':22s} {'P':>7s} {'R':>7s} {'F1':>7s} {'nTED':>7s} {'objective':>11s} {'sec':>7s}")
    for result in sorted(results, key=lambda item: (item.method, item.search)):
        print(f'{result.search:9s} {result.method:22s} {result.precision:7.3f} '
              f'{result.recall:7.3f} {result.f1:7.3f} '
              f'{result.normalized_pruned_ted:7.3f} {result.objective_cost:11.5f} '
              f'{result.runtime_seconds:7.3f}')
