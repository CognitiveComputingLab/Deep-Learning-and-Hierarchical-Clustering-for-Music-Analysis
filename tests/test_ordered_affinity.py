from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from greedy_clustering import ClusterNode,distance_specs
from greedy_evaluation import (boundary_prominence,collect_prominent_splits,
                               tree_shape_diagnostics)
from ordered_affinity import (affinity_tree_revenue,
    greedy_adjacent_average_linkage,optimal_affinity_tree,pairwise_affinity)


def brute_force_revenue(affinity):
    n=len(affinity)
    @lru_cache(None)
    def solve(i,j):
        if j==i+1: return 0.0
        return max(solve(i,k)+solve(k,j)+(n-(j-i))*affinity[i:k,k:j].sum()
                   for k in range(i+1,j))
    return solve(0,n)


def test_exact_affinity_dp_matches_brute_force():
    rng=np.random.default_rng(812)
    for n in range(2,8):
        raw=rng.random((n,n)); affinity=(raw+raw.T)/2
        np.fill_diagonal(affinity,1)
        matrix=rng.random((n,4))
        root,diagnostics=optimal_affinity_tree(
            matrix,np.arange(n+1),affinity,ClusterNode)
        recomputed,normalized=affinity_tree_revenue(root,affinity)
        assert np.isclose(diagnostics.total_revenue,brute_force_revenue(affinity))
        assert np.isclose(recomputed,diagnostics.total_revenue)
        assert np.isclose(normalized,diagnostics.normalized_revenue)


def test_dp_revenue_is_not_lower_than_adjacent_greedy():
    rng=np.random.default_rng(44)
    matrix=rng.random((9,12)); bounds=np.arange(10,dtype=float)
    spec=distance_specs(matrix.sum(axis=0))[0]['key_profile']
    affinity,_=pairwise_affinity(matrix,spec,context_radius=1)
    greedy=greedy_adjacent_average_linkage(matrix,bounds,affinity,ClusterNode)
    _,greedy_normalized=affinity_tree_revenue(greedy,affinity)
    _,diagnostics=optimal_affinity_tree(matrix,bounds,affinity,ClusterNode)
    assert diagnostics.normalized_revenue>=greedy_normalized-1e-12
    assert all(np.isfinite(row['child_mean_affinity'])
               for row in boundary_prominence(greedy))


def test_context_affinity_is_symmetric_and_bounded():
    rng=np.random.default_rng(91)
    matrix=rng.random((11,12))
    spec=distance_specs(matrix.sum(axis=0))[0]['euclidean']
    affinity,metadata=pairwise_affinity(matrix,spec,context_radius=2)
    assert np.allclose(affinity,affinity.T)
    assert np.all((affinity>=0)&(affinity<=1))
    assert np.allclose(np.diag(affinity),1)
    assert metadata['context_radius']==2 and metadata['scale']>0


def test_constant_affinity_ties_use_midpoint_without_changing_optimum():
    matrix=np.ones((6,12)); affinity=np.ones((6,6)); bounds=np.arange(7)
    midpoint,diagnostics=optimal_affinity_tree(
        matrix,bounds,affinity,ClusterNode,tie_break='midpoint')
    earliest,other=optimal_affinity_tree(
        matrix,bounds,affinity,ClusterNode,tie_break='earliest')
    assert diagnostics.root_split==3
    assert other.root_split==1
    assert diagnostics.tie_count>0
    assert np.isclose(diagnostics.total_revenue,other.total_revenue)
    assert tree_shape_diagnostics(midpoint)['root_split_ratio']==1
    assert tree_shape_diagnostics(earliest)['root_split_ratio']<1


def test_prominence_returns_each_boundary_once_and_equal_budget():
    matrix=np.eye(4); bounds=np.arange(5,dtype=float)
    # Explicit balanced tree: ((0,1),(2,3)).
    leaves=[ClusterNode(i,i+1,matrix[i]) for i in range(4)]
    left=ClusterNode(0,2,matrix[:2].sum(0),leaves[:2])
    right=ClusterNode(2,4,matrix[2:].sum(0),leaves[2:])
    root=ClusterNode(0,4,matrix.sum(0),[left,right])
    ranked=boundary_prominence(root)
    assert [row['boundary_qb'] for row in ranked]==[2,1,3]
    assert collect_prominent_splits(root,2)==[1,2]
    shape=tree_shape_diagnostics(root)
    assert shape['leaf_count']==4 and shape['max_depth']==2
    assert shape['root_split_ratio']==1
