'''Integration tests for project DP and parametric distance interfaces.'''

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))

from dp_clustering import (
    additive_tree_cost,
    assert_valid_ordered_binary_tree,
    node_vector,
    optimal_adjacent_binary_tree,
)
from greedy_clustering import ClusterNode, distance_specs
from greedy_clustering import greedy_cluster, load_pc_bins
from greedy_evaluation import tree_shape_diagnostics
from ordered_affinity import pairwise_affinity,optimal_boundary_aware_tree
from parametric_distances import (
    DiagonalMahalanobisDistance,
    WeightedDistanceMixture,
    estimate_distance_scales,
)
from scripts.train_parametric_distance import split_works, work_id


def test_project_bounds_and_feature_node_are_supported():
    matrix=np.asarray([[1,0,0],[0,1,0],[0,0,1]],dtype=float)
    bounds=np.asarray([[0,2],[2,4],[4,7]],dtype=float)
    distance=lambda left,right:float(np.linalg.norm(left/left.sum()-right/right.sum()))
    root,diagnostics=optimal_adjacent_binary_tree(matrix,bounds,distance,ClusterNode)
    assert_valid_ordered_binary_tree(root,bounds)
    assert np.allclose(node_vector(root),matrix.sum(axis=0))
    assert np.isclose(root.subtree_objective,diagnostics.total_cost)
    assert root.merge_cost>=0
    assert all(leaf.subtree_objective==0 for child in root.children
               for leaf in ([child] if not child.children else child.children))


def test_non_contiguous_bounds_are_rejected():
    matrix=np.ones((2,12))
    with pytest.raises(ValueError,match='contiguous'):
        optimal_adjacent_binary_tree(matrix,[(0,1),(2,3)],
                                     lambda left,right:0.0,ClusterNode)


def test_ties_choose_smallest_split_and_single_leaf_metadata():
    zero=lambda left,right:0.0
    matrix=np.ones((4,12))
    root,diagnostics=optimal_adjacent_binary_tree(matrix,np.arange(5),zero,ClusterNode)
    assert diagnostics.root_split==1
    assert diagnostics.tie_count>0
    leaf,one=optimal_adjacent_binary_tree(matrix[:1],np.arange(2),zero,ClusterNode)
    assert one.total_cost==0 and one.root_split==-1
    assert leaf.merge_cost==0 and leaf.subtree_objective==0


def test_scalar_and_batch_distance_specs_agree():
    rng=np.random.default_rng(17)
    matrix=rng.random((8,12))
    specs,_=distance_specs(matrix.sum(axis=0))
    for spec in specs.values():
        scalar=np.asarray([spec(left,right) for left,right in zip(matrix[:4],matrix[4:])])
        assert np.allclose(scalar,spec.batch_distance(matrix[:4],matrix[4:]))
        assert np.allclose(spec.batch_distance(matrix[:4],matrix[4:]),
                           spec.batch_distance(matrix[4:],matrix[:4]))


def test_calibrated_mixture_extremes_and_training_only_scales():
    rng=np.random.default_rng(3)
    matrix=rng.random((6,12))
    specs,_=distance_specs(matrix.sum(axis=0))
    base={name:specs[name] for name in ['euclidean','circle_of_fifths','key_profile']}
    pairs=list(zip(matrix[:3],matrix[3:]))
    scales=estimate_distance_scales(base,pairs)
    scale_vector=np.asarray([scales[name] for name in base])
    for index,name in enumerate(base):
        weights=np.zeros(3); weights[index]=1
        mixture=WeightedDistanceMixture.from_weights(base,weights,scale_vector)
        assert np.isclose(mixture(matrix[0],matrix[3]),
                          base[name](matrix[0],matrix[3])/scales[name])
    changed=estimate_distance_scales(base,pairs+[(np.ones(12),np.arange(12))])
    assert any(not np.isclose(scales[name],changed[name]) for name in scales)


def test_diagonal_numpy_batch_matches_scalar():
    rng=np.random.default_rng(9)
    left=rng.random((5,12)); right=rng.random((5,12))
    model=DiagonalMahalanobisDistance.from_logits(np.linspace(-1,1,12))
    scalar=np.asarray([model(a,b) for a,b in zip(left,right)])
    assert np.all(model.weights>=0)
    assert np.isclose(model.weights.sum(),1)
    assert np.allclose(scalar,model.batch_distance(left,right))


def test_quartet_level_split_is_deterministic_and_disjoint(tmp_path):
    pairs=[]
    for work in range(16):
        for movement in range(1,4):
            piece=f'n{work:02d}op1-{work}_{movement:02d}'
            pairs.append((piece,tmp_path/f'{piece}.notes.tsv',
                          tmp_path/f'{piece}.harmonies.tsv'))
    first,names1=split_works(pairs,42,10,3,3)
    second,names2=split_works(pairs,42,10,3,3)
    assert names1==names2
    sets={split:{work_id(piece) for piece,_,_ in values}
          for split,values in first.items()}
    assert len(sets['train'])==10 and len(sets['validation'])==3 and len(sets['test'])==3
    assert sets['train'].isdisjoint(sets['validation'])
    assert sets['train'].isdisjoint(sets['test'])
    assert sets['validation'].isdisjoint(sets['test'])


def test_vectorized_mixture_dp_matches_recomputed_objective():
    rng=np.random.default_rng(21)
    matrix=rng.random((7,12)); bounds=np.arange(8,dtype=float)
    specs,_=distance_specs(matrix.sum(axis=0))
    base={name:specs[name] for name in ['euclidean','circle_of_fifths','key_profile']}
    mixture=WeightedDistanceMixture.from_weights(base,[0.2,0.3,0.5],[1,2,3])
    root,diagnostics=optimal_adjacent_binary_tree(matrix,bounds,mixture,ClusterNode)
    assert np.isclose(diagnostics.total_cost,additive_tree_cost(root,mixture))


def test_real_op95_four_distances_dp_not_worse_than_greedy():
    notes=ROOT/'external'/'ABC'/'notes'/'n11op95_01.notes.tsv'
    if not notes.exists():
        pytest.skip('DCML corpus is not installed')
    matrix,bounds=load_pc_bins(notes,8.0)
    specs,_=distance_specs(matrix.sum(axis=0))
    for name in ['euclidean','circle_of_fifths','key_profile','tonic_weighted']:
        distance=specs[name]
        greedy=greedy_cluster(matrix,bounds,distance)
        _,diagnostics=optimal_adjacent_binary_tree(
            matrix,bounds,distance,ClusterNode,max_bins=350)
        assert diagnostics.total_cost<=additive_tree_cost(greedy,distance)+1e-8


def test_regularized_op95_pitchscape_dp_avoids_extreme_comb_trees():
    notes=ROOT/'external'/'ABC'/'notes'/'n11op95_01.notes.tsv'
    if not notes.exists():
        pytest.skip('DCML corpus is not installed')
    matrix,bounds=load_pc_bins(notes,8.0)
    specs,_=distance_specs(matrix.sum(axis=0))
    contrast=np.zeros(len(matrix)-1)
    for name in ['circle_of_fifths','key_profile','tonic_weighted']:
        affinity,_=pairwise_affinity(matrix,specs[name])
        tree,diagnostics=optimal_boundary_aware_tree(
            matrix,bounds,affinity,contrast,ClusterNode,
            contrast_weight=0.0,balance_weight=0.6,max_bins=350)
        shape=tree_shape_diagnostics(tree)
        assert diagnostics.balance_weight==0.6
        assert shape['max_depth']<=8
        assert shape['root_split_ratio']>=0.6
