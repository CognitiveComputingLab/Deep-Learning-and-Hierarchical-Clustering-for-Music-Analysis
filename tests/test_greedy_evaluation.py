import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from greedy_clustering import (KS_MAJOR,ClusterNode,load_pc_bins,_norm,estimate_global_key,
    tonic_relative_weights,d_euclidean,d_circle_of_fifths,d_key_profile,
    random_adjacent_merge_tree,collapse_empty_bins)
from greedy_evaluation import (boundary_scores,tree_size,count_leaves,prune_tree,
    find_prune_depth_matching,flat_tree_from_segments,clone_and_label,ted,normalized_ted,
    collect_k_splits,paired_permutation_tests)

def test_note_duration_is_split_across_bins(tmp_path):
    path=tmp_path/'notes.tsv'
    pd.DataFrame({'quarterbeats':[1.0],'duration_qb':[4.0],'midi':[60]}).to_csv(path,sep='\t',index=False)
    matrix,bounds=load_pc_bins(path,2.0)
    assert bounds==[(0.0,2.0),(2.0,4.0),(4.0,5.0)]
    assert matrix[:,0].tolist()==pytest.approx([1,2,1])

def test_empty_vector_and_distances_are_finite_and_symmetric():
    empty=np.zeros(12); note=np.eye(12)[7]
    assert _norm(empty).sum()==pytest.approx(1)
    for distance in (d_euclidean,d_circle_of_fifths,d_key_profile):
        assert np.isfinite(distance(empty,note))
        assert distance(empty,note)==pytest.approx(distance(note,empty))

def test_global_key_and_profile_rotation():
    tonic,mode,score=estimate_global_key(np.roll(KS_MAJOR,7))
    assert (tonic,mode)==(7,'major'); assert score>0.99
    assert np.argmax(tonic_relative_weights(7,'major'))==7

def test_boundary_matching_is_one_to_one_and_deterministic():
    result=boundary_scores([9,11],[10],2)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,0)
    assert result['matches'][0][0]==9
    assert boundary_scores([],[],2)['f1']==0
    assert boundary_scores([], [1],2)['fn']==1
    assert boundary_scores([1],[],2)['fp']==1

def test_boundary_matching_maximises_true_positives_before_error():
    result=boundary_scores([0,4],[3,5],3)
    assert result['tp']==2
    assert result['matches']==[(0.0,3.0,3.0),(4.0,5.0,1.0)]

def test_empty_bins_are_collapsed_into_contiguous_regions():
    matrix=np.array([[1]+[0]*11,[0]*12,[0]*12,[0,1]+[0]*10],dtype=float)
    collapsed,bounds=collapse_empty_bins(matrix,[(0,2),(2,4),(4,6),(6,8)])
    assert collapsed.shape==(2,12)
    assert bounds==[(0.0,4.0),(4.0,8.0)]

def sample_tree():
    leaves=[ClusterNode(i,i+1,np.ones(12)) for i in range(4)]
    left=ClusterNode(0,2,np.ones(12),leaves[:2]); right=ClusterNode(2,4,np.ones(12),leaves[2:])
    return ClusterNode(0,4,np.ones(12),[left,right])

def test_tree_counts_pruning_labels_and_normalized_ted():
    root=sample_tree(); assert tree_size(root)==7 and count_leaves(root)==4
    assert count_leaves(prune_tree(root,1))==2
    assert find_prune_depth_matching(root,3)==1
    reference=flat_tree_from_segments([(0,2,'a'),(2,4,'b')])
    left=clone_and_label(root,4,10,5); right=clone_and_label(reference,4,10,5)
    distance=ted(left,right)
    assert 0<=normalized_ted(distance,left,right)<=1
    assert left.label=='c5w4'

def test_random_baseline_reproducibility():
    matrix=np.eye(12)[:5]; bounds=[(i,i+1) for i in range(5)]
    first=random_adjacent_merge_tree(matrix,bounds,np.random.default_rng(42))
    second=random_adjacent_merge_tree(matrix,bounds,np.random.default_rng(42))
    def signature(node): return (node.start,node.end,tuple(signature(x) for x in node.children))
    assert signature(first)==signature(second)
    assert len(collect_k_splits(first,2))==2

def test_empty_paired_test_is_a_valid_table():
    result=paired_permutation_tests(pd.DataFrame(columns=['piece','method','f1']))
    assert result.empty and 'p_holm' in result.columns
