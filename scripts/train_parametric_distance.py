#!/usr/bin/env python
'''Train calibrated mixture and diagonal Mahalanobis music distances.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'src'))

from dp_clustering import additive_tree_cost, optimal_adjacent_binary_tree
from parametric_distances import (
    DiagonalMahalanobisDistance,
    WeightedDistanceMixture,
    estimate_distance_scales,
)

BASE_METHODS=['euclidean','circle_of_fifths','key_profile']


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('external/ABC'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/dp_parametric'))
    parser.add_argument('--model',choices=['mixture','diagonal','both'],default='both')
    parser.add_argument('--train-works',type=int,default=10)
    parser.add_argument('--validation-works',type=int,default=3)
    parser.add_argument('--test-works',type=int,default=3)
    parser.add_argument('--contexts',nargs='+',type=int,default=[1,2,4])
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--checkpoint-every',type=int,default=25)
    parser.add_argument('--learning-rate',type=float,default=0.05)
    parser.add_argument('--step',type=float,default=0.1)
    parser.add_argument('--seed',type=int,default=20260727)
    parser.add_argument('--bin-size',type=float,default=8.0)
    parser.add_argument('--tolerance',type=float,default=8.0)
    parser.add_argument('--depth',type=int,default=4)
    parser.add_argument('--max-bins',type=int,default=350)
    parser.add_argument('--quick',action='store_true')
    return parser.parse_args()


def work_id(piece):
    return piece.rsplit('_',1)[0] if '_' in piece else piece


def discover_pairs(root):
    notes={p.name.replace('.notes.tsv',''):p for p in (root/'notes').glob('*.notes.tsv')}
    harmonies={p.name.replace('.harmonies.tsv',''):p for p in (root/'harmonies').glob('*.harmonies.tsv')}
    return [(stem,notes[stem],harmonies[stem]) for stem in sorted(set(notes)&set(harmonies))]


def split_works(pairs,seed,train_works=10,validation_works=3,test_works=3):
    groups={}
    for pair in pairs:
        groups.setdefault(work_id(pair[0]),[]).append(pair)
    required=train_works+validation_works+test_works
    if len(groups)<required:
        raise ValueError(f'{required} complete works requested but only {len(groups)} are available')
    rng=np.random.default_rng(seed)
    works=np.asarray(sorted(groups),dtype=object)
    ordered=list(works[rng.permutation(len(works))])[:required]
    names={
        'train':ordered[:train_works],
        'validation':ordered[train_works:train_works+validation_works],
        'test':ordered[train_works+validation_works:],
    }
    return {split:[pair for work in selected for pair in sorted(groups[work])]
            for split,selected in names.items()},names


def simplex_grid(dimension,step):
    denominator=round(1/step)
    if not 0<step<=1 or not np.isclose(step*denominator,1):
        raise ValueError('step must be the reciprocal of an integer')
    def compositions(total,parts,prefix=()):
        if parts==1:
            yield prefix+(total,)
        else:
            for value in range(total+1):
                yield from compositions(total-value,parts-1,prefix+(value,))
    for counts in compositions(denominator,dimension):
        yield np.asarray(counts,dtype=float)/denominator


def load_cache(pairs,bin_size,max_bins):
    from greedy_clustering import distance_functions, load_pc_bins
    from greedy_evaluation import dcml_localkey_segments
    cache={}; statuses=[]
    for piece,notes,harmonies in pairs:
        try:
            segments,total=dcml_localkey_segments(harmonies)
            matrix,bounds=load_pc_bins(notes,bin_size)
            if len(matrix)>max_bins:
                raise ValueError(f'{len(matrix)} bins exceeds max_bins={max_bins}')
            distances,key=distance_functions(matrix.sum(axis=0),include_ablation=False)
            cache[piece]={'work':work_id(piece),'matrix':matrix,'bounds':bounds,
                          'segments':segments,'total':total,
                          'distances':{name:distances[name] for name in BASE_METHODS},
                          'estimated_key':key}
            statuses.append({'piece':piece,'work':work_id(piece),'status':'success','message':''})
        except Exception as error:
            statuses.append({'piece':piece,'work':work_id(piece),'status':'failed','message':repr(error)})
    return cache,statuses


def interval_examples(cache,contexts,seed):
    '''Create work-balanced supervised adjacent-interval pairs.'''
    rng=np.random.default_rng(seed); examples=[]
    for work in sorted({item['work'] for item in cache.values()}):
        positive=[]; negative=[]
        for piece,item in sorted(cache.items()):
            if item['work']!=work:
                continue
            matrix=item['matrix']; n=len(matrix); prefix=np.vstack([np.zeros(12),np.cumsum(matrix,axis=0)])
            bounds=np.asarray(item['bounds'],dtype=float)
            edges=np.r_[bounds[0,0],bounds[:,1]] if bounds.ndim==2 else bounds
            gt=np.asarray([end for _,end,_ in item['segments'][:-1]],dtype=float)
            marked={int(np.argmin(np.abs(edges[1:-1]-boundary)))+1 for boundary in gt} if len(gt) else set()
            for split in range(1,n):
                destination=positive if split in marked else negative
                for context in contexts:
                    if split-context>=0 and split+context<=n:
                        destination.append((prefix[split]-prefix[split-context],
                                            prefix[split+context]-prefix[split],int(split in marked)))
        count=min(len(positive),len(negative))
        if count:
            p=rng.choice(len(positive),count,replace=False)
            q=rng.choice(len(negative),count,replace=False)
            examples.extend(positive[i] for i in sorted(p))
            examples.extend(negative[i] for i in sorted(q))
    return examples


def evaluate(cache,distance_builder,searches,args,model):
    from greedy_clustering import ClusterNode, greedy_cluster
    from greedy_evaluation import boundary_scores, collect_top_splits
    rows=[]
    for piece,item in cache.items():
        distance=distance_builder(item)
        for search in searches:
            started=time.perf_counter()
            if search=='dp':
                tree,diagnostics=optimal_adjacent_binary_tree(
                    item['matrix'],item['bounds'],distance,ClusterNode,max_bins=args.max_bins)
                objective=diagnostics.total_cost
                evaluated=diagnostics.evaluated_splits
            else:
                tree=greedy_cluster(item['matrix'],item['bounds'],distance)
                objective=additive_tree_cost(tree,distance)
                evaluated=0
            elapsed=time.perf_counter()-started
            predicted=collect_top_splits(tree,args.depth)
            gt=[end for _,end,_ in item['segments'][:-1]]
            score=boundary_scores(predicted,gt,args.tolerance)
            rows.append({'piece':piece,'work':item['work'],'model':model,'search':search,
                         **{key:score[key] for key in ['tp','fp','fn','precision','recall','f1']},
                         'objective':objective,'runtime_seconds':elapsed,
                         'evaluated_splits':evaluated})
    return pd.DataFrame(rows)


def work_metrics(frame):
    return frame.groupby(['work','model','search'],as_index=False).agg(
        f1=('f1','mean'),recall=('recall','mean'),precision=('precision','mean'),
        objective=('objective','mean'),runtime_seconds=('runtime_seconds','mean'))


def train_mixture(train_cache,val_cache,examples,args):
    pairs=[(left,right) for left,right,_ in examples]
    representative=next(iter(train_cache.values()))['distances']
    scales_map=estimate_distance_scales(representative,pairs)
    scales=np.asarray([scales_map[name] for name in BASE_METHODS])
    rows=[]; best=None; best_key=None
    for index,weights in enumerate(simplex_grid(3,args.step),1):
        builder=lambda item,w=weights: WeightedDistanceMixture.from_weights(
            item['distances'],w,scales)
        result=evaluate(train_cache,builder,['dp'],args,'mixture')
        work=work_metrics(result)
        f1=float(work.f1.mean()); recall=float(work.recall.mean()); objective=float(work.objective.mean())
        rows.append({**{f'weight_{name}':float(value) for name,value in zip(BASE_METHODS,weights)},
                     'train_f1':f1,'train_recall':recall,'train_objective':objective})
        key=(f1,recall,-objective)
        if best is None or key>best_key or (key==best_key and tuple(weights)<tuple(best)):
            best=weights.copy(); best_key=key
        print(f'[mixture {index:02d}] train F1={f1:.4f}, weights={weights.tolist()}')
    builder=lambda item: WeightedDistanceMixture.from_weights(item['distances'],best,scales)
    validation=evaluate(val_cache,builder,['dp'],args,'mixture')
    return best,scales_map,pd.DataFrame(rows),validation


def train_diagonal(examples,val_cache,args):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch is required for --model diagonal') from error
    torch.manual_seed(args.seed)
    left=np.asarray([x[0] for x in examples],dtype=np.float32)
    right=np.asarray([x[1] for x in examples],dtype=np.float32)
    labels=np.asarray([x[2] for x in examples],dtype=np.float32)
    left/=np.maximum(left.sum(axis=1,keepdims=True),1e-12)
    right/=np.maximum(right.sum(axis=1,keepdims=True),1e-12)
    unweighted=np.sqrt(np.mean((left-right)**2,axis=1))
    margin=float(np.median(unweighted[labels==1])) if np.any(labels==1) else float(np.median(unweighted))
    lt=torch.tensor(left); rt=torch.tensor(right); yt=torch.tensor(labels)
    logits=torch.zeros(12,requires_grad=True)
    optimizer=torch.optim.Adam([logits],lr=args.learning_rate)
    history=[]; best_weights=None; best_f1=-np.inf; best_epoch=-1
    for epoch in range(1,args.epochs+1):
        optimizer.zero_grad(); weights=torch.softmax(logits,dim=0)
        distance=torch.sqrt(torch.sum(weights*(lt-rt)**2,dim=1)+1e-12)
        loss=torch.mean((1-yt)*distance**2+yt*torch.relu(margin-distance)**2)
        loss.backward(); optimizer.step()
        row={'epoch':epoch,'loss':float(loss.detach()),'validation_f1':np.nan}
        if epoch%args.checkpoint_every==0 or epoch==args.epochs:
            candidate=DiagonalMahalanobisDistance.from_weights(
                torch.softmax(logits.detach(),dim=0).cpu().numpy())
            validation=evaluate(val_cache,lambda item,d=candidate:d,['dp'],args,'diagonal')
            f1=float(work_metrics(validation).f1.mean())
            row['validation_f1']=f1
            if f1>best_f1+1e-12:
                best_f1=f1; best_epoch=epoch; best_weights=candidate.weights.copy()
            print(f'[diagonal {epoch:03d}] loss={row["loss"]:.6f}, validation F1={f1:.4f}')
        history.append(row)
    model=DiagonalMahalanobisDistance.from_weights(best_weights)
    return model,pd.DataFrame(history),{'margin':margin,'best_epoch':best_epoch,
                                        'validation_f1':best_f1}


def main():
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=discover_pairs(args.data_root)
    if args.quick:
        selected=sorted({work_id(piece) for piece,_,_ in pairs})[:6]
        pairs=[pair for pair in pairs if work_id(pair[0]) in selected]
        pairs=[min((pair for pair in pairs if work_id(pair[0])==work),key=lambda item:item[0])
               for work in selected]
        args.train_works,args.validation_works,args.test_works=3,1,2
        args.step=max(args.step,0.5); args.epochs=min(args.epochs,25)
        args.checkpoint_every=min(args.checkpoint_every,args.epochs)
    splits,work_names=split_works(pairs,args.seed,args.train_works,
                                  args.validation_works,args.test_works)
    caches={}; statuses=[]
    for split,pieces in splits.items():
        caches[split],current=load_cache(pieces,args.bin_size,args.max_bins)
        for row in current:
            row['split']=split
        statuses.extend(current)
    pd.DataFrame(statuses).to_csv(args.output_dir/'run_status.csv',index=False)
    assignment=[{'work':work,'piece':piece,'split':split}
                for split,items in splits.items() for piece,_,_ in items
                for work in [work_id(piece)]]
    pd.DataFrame(assignment).to_csv(args.output_dir/'data_split.csv',index=False)
    examples=interval_examples(caches['train'],args.contexts,args.seed)
    if not examples:
        raise SystemExit('No balanced training interval pairs could be constructed')

    learned={}; model_info={}
    if args.model in {'mixture','both'}:
        weights,scales,search,validation=train_mixture(
            caches['train'],caches['validation'],examples,args)
        search.to_csv(args.output_dir/'mixture_weight_search.csv',index=False)
        validation.to_csv(args.output_dir/'mixture_validation.csv',index=False)
        scale_vector=np.asarray([scales[name] for name in BASE_METHODS])
        learned['learned_mixture']=lambda item,w=weights,s=scale_vector: (
            WeightedDistanceMixture.from_weights(item['distances'],w,s))
        model_info['mixture']={'weights':dict(zip(BASE_METHODS,map(float,weights))),
                               'training_scales':scales,
                               'validation_work_f1':float(work_metrics(validation).f1.mean())}
    if args.model in {'diagonal','both'}:
        diagonal,history,details=train_diagonal(examples,caches['validation'],args)
        history.to_csv(args.output_dir/'diagonal_training_history.csv',index=False)
        learned['learned_diagonal']=lambda item,d=diagonal:d
        model_info['diagonal']={'weights':diagonal.weights.tolist(),**details}

    test_frames=[]
    for name in BASE_METHODS:
        test_frames.append(evaluate(caches['test'],lambda item,n=name:item['distances'][n],
                                    ['greedy','dp'],args,f'handcrafted_{name}'))
    for name,builder in learned.items():
        test_frames.append(evaluate(caches['test'],builder,['greedy','dp'],args,name))
    test=pd.concat(test_frames,ignore_index=True)
    test.to_csv(args.output_dir/'held_out_test_per_piece.csv',index=False)
    test_work=work_metrics(test)
    test_work.to_csv(args.output_dir/'held_out_test_per_work.csv',index=False)
    summary=test_work.groupby(['model','search'],as_index=False).agg(
        works=('work','nunique'),mean_f1=('f1','mean'),std_f1=('f1','std'),
        mean_recall=('recall','mean'),mean_precision=('precision','mean'),
        mean_objective=('objective','mean'),mean_runtime_seconds=('runtime_seconds','mean'))
    summary.to_csv(args.output_dir/'held_out_test_summary.csv',index=False)
    objective=test.pivot_table(index=['piece','work','model'],columns='search',
                               values='objective',aggfunc='mean').reset_index()
    if {'greedy','dp'} <= set(objective):
        objective['objective_gap']=objective.greedy-objective.dp
        objective['relative_objective_gap']=(
            objective.objective_gap/objective.greedy.abs().replace(0,np.nan))
    objective.to_csv(args.output_dir/'held_out_objective_comparison.csv',index=False)
    scale_rows=[]; weight_rows=[]
    if 'mixture' in model_info:
        scale_rows=[{'distance':name,'training_scale':value}
                    for name,value in model_info['mixture']['training_scales'].items()]
        weight_rows.extend({'model':'mixture','feature':name,'weight':value}
                           for name,value in model_info['mixture']['weights'].items())
    if 'diagonal' in model_info:
        weight_rows.extend({'model':'diagonal','feature':f'pitch_class_{index}',
                            'weight':value}
                           for index,value in enumerate(model_info['diagonal']['weights']))
    pd.DataFrame(scale_rows).to_csv(args.output_dir/'distance_scales.csv',index=False)
    pd.DataFrame(weight_rows).to_csv(args.output_dir/'learned_weights.csv',index=False)
    metadata={'seed':args.seed,'splits':work_names,'bin_size_qb':args.bin_size,
              'tolerance_qb':args.tolerance,'depth':args.depth,'contexts':args.contexts,
              'training_examples':len(examples),'models':model_info,
              'test_used_for_selection':False,
              'optimality_scope':'Fixed ordered leaves and additive child-distance objective only.'}
    (args.output_dir/'models.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\nHELD-OUT TEST SUMMARY')
    print(summary.to_string(index=False,float_format=lambda value:f'{value:.4f}'))


if __name__=='__main__':
    main()
