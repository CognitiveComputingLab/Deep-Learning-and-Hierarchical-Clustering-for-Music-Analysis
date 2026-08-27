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

def boundary_prominence(root):
    '''Return every tree boundary ranked by its LCA temporal span.

    A complete binary tree eventually contains every leaf boundary, so merely
    collecting all splits is uninformative. The LCA span is a depth-independent
    measure that permits equal-budget comparisons across different tree shapes.
    '''
    rows=[]
    def walk(node):
        children=list(getattr(node,'children',[]) or [])
        if not children: return 1
        if len(children)!=2:
            raise ValueError('boundary prominence requires a binary tree')
        left_leaves=walk(children[0]); right_leaves=walk(children[1])
        rows.append({'boundary_qb':float(children[0].end),
                     'lca_span_qb':float(node.end-node.start),
                     'lca_leaf_count':left_leaves+right_leaves,
                     'child_mean_affinity':float(getattr(node,'child_mean_affinity',np.nan)),
                     'merge_order':int(getattr(node,'merge_order',-1))})
        return left_leaves+right_leaves
    walk(root)
    total_span=float(root.end-root.start)
    if not np.isfinite(total_span) or total_span<=0:
        raise ValueError('tree must have a finite positive temporal span')
    for row in rows:
        row['prominence']=row['lca_span_qb']/total_span
    return sorted(rows,key=lambda row:(-row['prominence'],-row['lca_leaf_count'],
        row['child_mean_affinity'] if np.isfinite(row['child_mean_affinity']) else np.inf,
        -row['merge_order'],row['boundary_qb']))

def boundary_prominence_scores(root,bounds):
    '''Return one canonical time-span prominence score per internal bin edge.

    This is the single ranking used by fixed-budget evaluation, average
    precision, and the RL terminal reward.  Matching is by the exact temporal
    edge represented by each ordered-tree split; the nearest-edge fallback is
    retained only for small floating-point conversion errors.
    '''
    bounds=np.asarray(bounds,dtype=float)
    edges=np.r_[bounds[0,0],bounds[:,1]] if bounds.ndim==2 else bounds
    if edges.ndim!=1 or len(edges)<2 or np.any(np.diff(edges)<=0):
        raise ValueError('bounds must define strictly increasing bin edges')
    internal=edges[1:-1]
    scores=np.zeros(len(internal),dtype=float)
    for row in boundary_prominence(root):
        if not len(internal): break
        index=int(np.argmin(np.abs(internal-row['boundary_qb'])))
        tolerance=1e-7*max(1.0,abs(float(row['boundary_qb'])))
        if abs(float(internal[index])-float(row['boundary_qb']))>tolerance:
            raise ValueError('tree boundary does not coincide with a supplied bin edge')
        scores[index]=float(row['prominence'])
    return scores

def collect_prominent_splits(root,budget):
    '''Select an equal number of the tree's most structurally prominent splits.'''
    budget=max(0,int(budget))
    return sorted(row['boundary_qb'] for row in boundary_prominence(root)[:budget])

def boundary_salience(root,contrast=None):
    '''Return calibrated boundary scores coupled to normalized LCA leaf span.

    Contrast follows leaf order and has one entry per candidate boundary.
    Passing None produces the structure-only baseline, for which every local
    contrast equals one and salience is solely normalized LCA span.
    '''
    n=count_leaves(root)
    values=np.ones(max(0,n-1),dtype=float) if contrast is None else np.asarray(contrast,dtype=float)
    if values.shape!=(max(0,n-1),) or not np.all(np.isfinite(values)):
        raise ValueError('contrast must contain one finite value per leaf boundary')
    if np.any(values<0) or np.any(values>1):
        raise ValueError('contrast values must lie in [0,1]')
    rows=[]; cursor=[0]
    def walk(node):
        children=list(getattr(node,'children',[]) or [])
        if not children:
            index=cursor[0]; cursor[0]+=1
            return index,index+1
        if len(children)!=2:
            raise ValueError('boundary salience requires a binary tree')
        start,split=walk(children[0]); right_start,end=walk(children[1])
        if split!=right_start: raise ValueError('tree leaves are not temporally ordered')
        span=end-start
        span_weight=float(np.log(span)/np.log(n)) if n>1 and span>=2 else 0.0
        local=float(values[split-1])
        rows.append({'boundary_qb':float(children[0].end),'boundary_index':split,
                     'lca_leaf_count':span,'lca_span_qb':float(node.end-node.start),
                     'span_weight':span_weight,'boundary_contrast':local,
                     'salience':span_weight*local})
        return start,end
    walk(root)
    return sorted(rows,key=lambda row:(-row['salience'],-row['lca_leaf_count'],
                                       row['boundary_qb'],row['boundary_index']))

def collect_salient_splits(root,contrast=None,*,threshold=None,budget=None):
    '''Select variable-count threshold boundaries or an equal fixed budget.'''
    if (threshold is None)==(budget is None):
        raise ValueError('specify exactly one of threshold or budget')
    rows=boundary_salience(root,contrast)
    if threshold is not None:
        threshold=float(threshold)
        if not 0<=threshold<=1: raise ValueError('threshold must lie in [0,1]')
        selected=[row for row in rows if row['salience']>=threshold]
    else:
        selected=rows[:max(0,int(budget))]
    return sorted(row['boundary_qb'] for row in selected)

def tree_shape_diagnostics(root):
    '''Quantify imbalance so objective gains are not confused with comb trees.'''
    leaf_depths=[]; singleton_children=0; internal_nodes=0; colless=0
    def walk(node,depth=0):
        nonlocal singleton_children,internal_nodes,colless
        children=list(getattr(node,'children',[]) or [])
        if not children:
            leaf_depths.append(depth); return 1
        if len(children)!=2: raise ValueError('tree shape diagnostics require a binary tree')
        internal_nodes+=1
        left=walk(children[0],depth+1); right=walk(children[1],depth+1)
        colless+=abs(left-right)
        singleton_children+=int(left==1)+int(right==1)
        return left+right
    leaves=walk(root)
    root_children=list(getattr(root,'children',[]) or [])
    if root_children:
        def count(node):
            children=list(getattr(node,'children',[]) or [])
            return 1 if not children else sum(count(child) for child in children)
        root_left=count(root_children[0]); root_right=count(root_children[1])
        root_ratio=min(root_left,root_right)/max(root_left,root_right)
    else:
        root_left,root_right,root_ratio=1,0,1.0
    maximum=max(leaf_depths) if leaf_depths else 0
    return {'leaf_count':leaves,'max_depth':maximum,
            'normalized_max_depth':maximum/max(1,leaves-1),
            'sackin_index':sum(leaf_depths),
            'normalized_sackin_index':sum(leaf_depths)/max(1,leaves*max(1,leaves-1)),
            'colless_index':colless,
            'normalized_colless_index':colless/max(1,(leaves-1)*(leaves-2)/2),
            'singleton_child_ratio':singleton_children/max(1,2*internal_nodes),
            'root_left_leaves':root_left,'root_right_leaves':root_right,
            'root_split_ratio':root_ratio}

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
