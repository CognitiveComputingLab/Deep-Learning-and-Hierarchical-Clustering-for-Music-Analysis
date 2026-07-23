'''Evaluation utilities for DCML local-key boundaries and auxiliary flat-tree TED.'''
from fractions import Fraction
import numpy as np
import pandas as pd
import zss
from greedy_clustering import ClusterNode

def _float(value):
    return float(Fraction(value)) if isinstance(value,str) and '/' in value else float(value)

def dcml_localkey_segments(path):
    data=pd.read_csv(path,sep='\t').dropna(subset=['quarterbeats','localkey']).copy()
    if data.empty: raise ValueError(f'no timed local-key annotations in {path}')
    data['qb']=data['quarterbeats'].map(_float)
    data['dur']=data['duration_qb'].map(_float)
    data=data.sort_values('qb',kind='stable')
    total=float((data.qb+data.dur).max()); segments=[]
    key=data.iloc[0].localkey; start=float(data.iloc[0].qb)
    for row in data.itertuples():
        if row.localkey!=key:
            segments.append((start,float(row.qb),key)); start=float(row.qb); key=row.localkey
    segments.append((start,total,key))
    return segments,total

def flat_tree_from_segments(segments):
    children=[ClusterNode(s,e,None) for s,e,_ in segments]
    return ClusterNode(segments[0][0],segments[-1][1],None,children)

def clone_and_label(node,total,center_bins=10,width_bins=5):
    if total<=0: raise ValueError('total duration must be positive')
    copy=ClusterNode(node.start,node.end,node.feature)
    center=(node.start+node.end)/2; width=node.end-node.start
    c=min(center_bins-1,max(0,int(center_bins*center/total)))
    w=min(width_bins-1,max(0,int(width_bins*width/total)))
    copy.label=f'c{c}w{w}'
    copy.children=[clone_and_label(x,total,center_bins,width_bins) for x in node.children]
    return copy

def tree_size(node): return 1+sum(tree_size(child) for child in node.children)
def count_leaves(node): return 1 if not node.children else sum(count_leaves(x) for x in node.children)

def prune_tree(node,max_depth,depth=0):
    children=[] if depth>=max_depth else [prune_tree(x,max_depth,depth+1) for x in node.children]
    return ClusterNode(node.start,node.end,node.feature,children)

def find_prune_depth_matching(root,target_leaves):
    candidates=[]
    for depth in range(1,tree_size(root)+1):
        leaves=count_leaves(prune_tree(root,depth)); candidates.append((abs(leaves-target_leaves),depth))
        if leaves>=target_leaves: break
    return min(candidates)[1]

def ted(left,right):
    return float(zss.simple_distance(left,right,lambda n:n.children,
                                     lambda n:n.label,lambda a,b:0 if a==b else 1))

def normalized_ted(distance,left,right):
    return float(distance)/max(tree_size(left),tree_size(right))

def collect_top_splits(node,max_depth):
    splits=set()
    def walk(current,depth):
        if depth>=max_depth or not current.children: return
        splits.update(child.end for child in current.children[:-1])
        for child in current.children: walk(child,depth+1)
    walk(node,0)
    return sorted(splits)

def collect_k_splits(root,budget):
    '''Return a dendrogram cut with at most budget boundaries.'''
    budget=max(0,int(budget)); frontier=[root]; splits=[]
    while len(splits)<budget:
        candidates=[(index,node) for index,node in enumerate(frontier) if node.children]
        if not candidates: break
        index,node=max(candidates,key=lambda item:(
            getattr(item[1],'merge_order',-1),item[1].end-item[1].start,-item[0]))
        frontier[index:index+1]=node.children
        splits.extend(child.end for child in node.children[:-1])
    return sorted(set(splits))[:budget]

def boundary_scores(predicted,reference,tolerance):
    '''Order-preserving matching: maximise TP, then minimise total timing error.'''
    predicted=sorted(set(map(float,predicted))); reference=sorted(map(float,reference))
    n,m=len(predicted),len(reference)
    dp=[[(0,0.0,()) for _ in range(m+1)] for _ in range(n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):
            candidates=[dp[i-1][j],dp[i][j-1]]
            distance=abs(predicted[i-1]-reference[j-1])
            if distance<=tolerance:
                matched,error,pairs=dp[i-1][j-1]
                pair=(predicted[i-1],reference[j-1],distance)
                candidates.append((matched+1,error+distance,pairs+(pair,)))
            dp[i][j]=min(candidates,key=lambda x:(-x[0],x[1],x[2]))
    matches=list(dp[n][m][2])
    tp=len(matches); fp=len(predicted)-tp; fn=len(reference)-tp
    precision=tp/len(predicted) if predicted else 0.0
    recall=tp/len(reference) if reference else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
            'predicted_boundary_count':len(predicted),'gt_boundary_count':len(reference),
            'matches':matches,
            'mean_match_error_qb':sum(x[2] for x in matches)/tp if tp else np.nan}

def ted_diagnostics(root,segments,total,center_bins,width_bins):
    reference=flat_tree_from_segments(segments)
    depth=find_prune_depth_matching(root,len(segments)); pruned=prune_tree(root,depth)
    pred_lab=clone_and_label(root,total,center_bins,width_bins)
    prune_lab=clone_and_label(pruned,total,center_bins,width_bins)
    ref_lab=clone_and_label(reference,total,center_bins,width_bins)
    raw=ted(pred_lab,ref_lab); pruned_distance=ted(prune_lab,ref_lab)
    return {'raw_ted':raw,'pruned_ted':pruned_distance,
            'normalized_raw_ted':normalized_ted(raw,pred_lab,ref_lab),
            'normalized_pruned_ted':normalized_ted(pruned_distance,prune_lab,ref_lab),
            'node_count_scaled_raw_ted':normalized_ted(raw,pred_lab,ref_lab),
            'node_count_scaled_pruned_ted':normalized_ted(pruned_distance,prune_lab,ref_lab),
            'predicted_node_count':tree_size(root),'gt_node_count':tree_size(reference),
            'pruned_node_count':tree_size(pruned),'pruned_leaf_count':count_leaves(pruned),
            'target_segment_count':len(segments),'selected_pruning_depth':depth,
            'center_bins':center_bins,'width_bins':width_bins}

def bootstrap_summary(frame,group_columns,value='f1',samples=2000,seed=0):
    rows=[]; rng=np.random.default_rng(seed)
    for keys,group in frame.groupby(group_columns,dropna=False):
        values=group[value].dropna().to_numpy(float); keys=keys if isinstance(keys,tuple) else (keys,)
        means=np.mean(rng.choice(values,(samples,len(values)),replace=True),axis=1) if len(values) else np.array([np.nan])
        row=dict(zip(group_columns,keys)); row.update({'n_pieces':len(values),'mean':np.mean(values),
            'std':np.std(values,ddof=1) if len(values)>1 else 0.0,
            'ci95_low':np.quantile(means,.025),'ci95_high':np.quantile(means,.975)})
        rows.append(row)
    return pd.DataFrame(rows)

def micro_summary(frame,group_columns):
    grouped=frame.groupby(group_columns,dropna=False)[['tp','fp','fn']].sum().reset_index()
    grouped['precision']=grouped.tp/(grouped.tp+grouped.fp).replace(0,np.nan)
    grouped['recall']=grouped.tp/(grouped.tp+grouped.fn).replace(0,np.nan)
    grouped[['precision','recall']]=grouped[['precision','recall']].fillna(0)
    grouped['f1']=2*grouped.precision*grouped.recall/(grouped.precision+grouped.recall).replace(0,np.nan)
    grouped['f1']=grouped.f1.fillna(0); return grouped

def paired_permutation_tests(frame,samples=10000,seed=0,unit_column='piece'):
    '''Pairwise two-sided sign-flip tests on per-piece F1, with Holm correction.'''
    columns=['method_a','method_b','n_pairs','mean_f1_difference','p_value','p_holm']
    if frame.empty or 'method' not in frame or frame.method.nunique()<2:
        return pd.DataFrame(columns=columns)
    rng=np.random.default_rng(seed); methods=sorted(frame.method.unique()); rows=[]
    pivot=frame.pivot_table(index=unit_column,columns='method',values='f1',aggfunc='mean')
    for i,left in enumerate(methods):
        for right in methods[i+1:]:
            paired=pivot[[left,right]].dropna(); differences=(paired[left]-paired[right]).to_numpy()
            observed=abs(differences.mean()) if len(differences) else np.nan
            if len(differences):
                null=np.abs((rng.choice([-1,1],(samples,len(differences)))*differences).mean(axis=1))
                p=(1+np.sum(null>=observed))/(samples+1)
            else: p=np.nan
            rows.append({'method_a':left,'method_b':right,'n_pairs':len(differences),
                         'mean_f1_difference':differences.mean() if len(differences) else np.nan,
                         'p_value':p})
    result=pd.DataFrame(rows); valid=result.p_value.notna(); ordered=result.loc[valid].sort_values('p_value')
    adjusted=[]; running=0.0; count=len(ordered)
    for rank,p in enumerate(ordered.p_value):
        running=max(running,min(1.0,(count-rank)*p)); adjusted.append(running)
    result['p_holm']=np.nan; result.loc[ordered.index,'p_holm']=adjusted
    return result[columns]
