'''Contiguous greedy clustering and centralised pitch-class distances.'''
from fractions import Fraction
import numpy as np
import pandas as pd

KS_MAJOR=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_MINOR=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
PITCH_CLASS_NAMES=('C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B')
FIFTHS_ORDER=(0,7,2,9,4,11,6,1,8,3,10,5)
_PC_TO_FIFTH={pc:i for i,pc in enumerate(FIFTHS_ORDER)}
_ANGLES=np.array([2*np.pi*_PC_TO_FIFTH[pc]/12 for pc in range(12)])

def to_float_qb(value):
    return float(Fraction(value)) if isinstance(value,str) and '/' in value else float(value)

def load_pc_bins(notes_tsv,bin_size_qb=4.0,collapse_empty=True):
    '''Return duration-weighted pitch-class bins and exact time bounds.'''
    if bin_size_qb<=0: raise ValueError('bin_size_qb must be positive')
    notes=pd.read_csv(notes_tsv,sep='\t').dropna(subset=['quarterbeats'])
    if notes.empty: raise ValueError(f'no timed notes in {notes_tsv}')
    notes['qb']=notes['quarterbeats'].map(to_float_qb)
    notes['dur']=notes['duration_qb'].astype(float)
    notes['pc']=notes['midi'].astype(int)%12
    total=float((notes.qb+notes.dur).max())
    n_bins=int(np.ceil(total/bin_size_qb)); matrix=np.zeros((n_bins,12))
    for note in notes.itertuples():
        start,end,pc=float(note.qb),float(note.qb+note.dur),int(note.pc)
        b0=max(0,int(start//bin_size_qb))
        b1=min(n_bins-1,int(min(end,total-1e-9)//bin_size_qb))
        for b in range(b0,b1+1):
            lo,hi=b*bin_size_qb,min((b+1)*bin_size_qb,total)
            matrix[b,pc]+=max(0.0,min(end,hi)-max(start,lo))
    bounds=[(i*bin_size_qb,min((i+1)*bin_size_qb,total)) for i in range(n_bins)]
    return collapse_empty_bins(matrix,bounds) if collapse_empty else (matrix,bounds)

def collapse_empty_bins(matrix,bounds):
    '''Assign silent time to the nearest sounding bin instead of uniform chroma.'''
    active=np.flatnonzero(np.asarray(matrix).sum(axis=1)>0)
    if len(active)==0: raise ValueError('all pitch-class bins are empty')
    if len(active)==len(bounds): return matrix,bounds
    centers=np.array([(bounds[i][0]+bounds[i][1])/2 for i in active])
    cuts=(centers[:-1]+centers[1:])/2
    starts=np.r_[bounds[0][0],cuts]; ends=np.r_[cuts,bounds[-1][1]]
    new_bounds=[(float(start),float(end)) for start,end in zip(starts,ends)]
    return np.asarray(matrix)[active].copy(),new_bounds

def _norm(vector):
    vector=np.asarray(vector,dtype=float); total=vector.sum()
    return vector/total if total>0 else np.full(12,1/12)

def _corr(a,b):
    return 0.0 if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])

def estimate_global_key(pc_vector):
    '''Estimate tonic/mode by maximum correlation with 24 KS profiles.'''
    vector=_norm(pc_vector); candidates=[]
    for tonic in range(12):
        candidates += [(_corr(vector,np.roll(KS_MAJOR,tonic)),tonic,'major'),
                       (_corr(vector,np.roll(KS_MINOR,tonic)),tonic,'minor')]
    score,tonic,mode=max(candidates,key=lambda x:(x[0],-x[1],x[2]=='major'))
    return tonic,mode,score

def tonic_relative_weights(tonic,mode):
    weights=np.roll(KS_MAJOR if mode=='major' else KS_MINOR,int(tonic)%12).astype(float)
    return weights/weights.sum()

def d_euclidean(a,b): return float(np.linalg.norm(_norm(a)-_norm(b)))

def circle_of_fifths_embedding(vector):
    vector=_norm(vector)
    return np.array([np.sum(vector*np.cos(_ANGLES)),np.sum(vector*np.sin(_ANGLES))])

def d_circle_of_fifths(a,b):
    return float(np.linalg.norm(circle_of_fifths_embedding(a)-circle_of_fifths_embedding(b)))

def make_weighted_distance(weights):
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(12,) or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('weights must be a non-negative 12-vector')
    weights=weights/weights.sum()
    return lambda a,b: float(np.sqrt(np.sum(weights*(_norm(a)-_norm(b))**2)))

def key_activation(vector):
    vector=_norm(vector); values=[]
    for tonic in range(12):
        values += [_corr(vector,np.roll(KS_MAJOR,tonic)),_corr(vector,np.roll(KS_MINOR,tonic))]
    return np.asarray(values)

def d_key_profile(a,b): return float(np.linalg.norm(key_activation(a)-key_activation(b)))

FIXED_C_MAJOR_WEIGHTS=KS_MAJOR/KS_MAJOR.sum()
d_fixed_c_major_weighted=make_weighted_distance(FIXED_C_MAJOR_WEIGHTS)

def distance_functions(piece_pc_vector=None,include_ablation=False):
    distances={'euclidean':d_euclidean,'circle_of_fifths':d_circle_of_fifths,
               'key_profile':d_key_profile}; key=None
    if piece_pc_vector is not None:
        tonic,mode,score=estimate_global_key(piece_pc_vector)
        distances['tonic_weighted']=make_weighted_distance(tonic_relative_weights(tonic,mode))
        key={'tonic_pc':tonic,'tonic':PITCH_CLASS_NAMES[tonic],
             'mode':mode,'correlation':score}
    if include_ablation:
        distances['fixed_c_major_weighted_ablation']=d_fixed_c_major_weighted
    return distances,key

class ClusterNode:
    def __init__(self,start,end,feature,children=None,merge_order=-1):
        self.start,self.end=float(start),float(end)
        self.feature=feature; self.children=list(children or []); self.label=''
        self.merge_order=merge_order

def _leaves(pc_mat,boundaries):
    if len(boundaries)==0: raise ValueError('cannot build a tree from an empty sequence')
    return [ClusterNode(s,e,pc_mat[i].copy()) for i,(s,e) in enumerate(boundaries)]

def _merge(left,right,merge_order=-1):
    return ClusterNode(left.start,right.end,left.feature+right.feature,[left,right],merge_order)

def greedy_cluster(pc_mat,boundaries,dist_fn):
    clusters=_leaves(pc_mat,boundaries)
    distances=[dist_fn(clusters[j].feature,clusters[j+1].feature)
               for j in range(len(clusters)-1)]
    merge_order=0
    while len(clusters)>1:
        i=int(np.argmin(distances))
        clusters[i:i+2]=[_merge(clusters[i],clusters[i+1],merge_order)]
        merge_order+=1
        distances.pop(i)
        if i>0:
            distances[i-1]=dist_fn(clusters[i-1].feature,clusters[i].feature)
        if i<len(distances):
            distances[i]=dist_fn(clusters[i].feature,clusters[i+1].feature)
    return clusters[0]

def balanced_tree(pc_mat,boundaries):
    merge_order=[0]
    def build(nodes):
        if len(nodes)==1: return nodes[0]
        middle=len(nodes)//2
        left,right=build(nodes[:middle]),build(nodes[middle:])
        node=_merge(left,right,merge_order[0]); merge_order[0]+=1
        return node
    return build(_leaves(pc_mat,boundaries))

def random_adjacent_merge_tree(pc_mat,boundaries,rng):
    clusters=_leaves(pc_mat,boundaries)
    merge_order=0
    while len(clusters)>1:
        i=int(rng.integers(0,len(clusters)-1))
        clusters[i:i+2]=[_merge(clusters[i],clusters[i+1],merge_order)]
        merge_order+=1
    return clusters[0]

# Compatibility aliases for old plots; new evaluations use accurate names.
d_tonnetz=d_circle_of_fifths
d_weighted=d_fixed_c_major_weighted
DISTANCES={'euclidean':d_euclidean,'circle_of_fifths':d_circle_of_fifths,
           'key_profile':d_key_profile,
           'fixed_c_major_weighted_ablation':d_fixed_c_major_weighted,
           'tonnetz':d_circle_of_fifths,'weighted':d_fixed_c_major_weighted,
           'keyprofile':d_key_profile}
