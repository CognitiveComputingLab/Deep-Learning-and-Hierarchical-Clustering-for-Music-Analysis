#!/usr/bin/env python
'''Evaluate accuracy-oriented greedy and exact DP under one shared objective.

This is the recommended intermediate-stage entry point.  It preserves the
supervisor's adjacent bottom-up greedy baseline and compares it with an exact
ordered interval DP.  DCML labels are used only after trees have been built.
'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from greedy_clustering import ClusterNode,distance_specs,load_pc_bins
from greedy_evaluation import (bootstrap_summary,boundary_scores,
    collect_prominent_splits,collect_top_splits,dcml_localkey_segments,
    paired_permutation_tests,ted_diagnostics,tree_shape_diagnostics)
from dp_clustering import assert_valid_ordered_binary_tree
from ordered_affinity import (affinity_tree_revenue,
    greedy_adjacent_average_linkage,optimal_affinity_tree,pairwise_affinity)


DEFAULT_METHODS=['euclidean','circle_of_fifths','key_profile','tonic_weighted']


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('external/ABC'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/optimized_stage'))
    parser.add_argument('--piece',action='append')
    parser.add_argument('--methods',nargs='+',default=DEFAULT_METHODS)
    parser.add_argument('--bin-size',type=float,default=8.0)
    parser.add_argument('--tolerances',nargs='+',type=float,default=[2,4,8,12,24])
    parser.add_argument('--depths',nargs='+',type=int,default=[1,2,3,4,5,6])
    parser.add_argument('--boundary-budgets',nargs='+',type=int,
                        default=[3,5,8,10,12,15,20])
    parser.add_argument('--context-radii',nargs='+',type=int,default=[0,2])
    parser.add_argument('--tie-break',choices=['earliest','midpoint','latest'],
                        default='midpoint')
    parser.add_argument('--max-bins',type=int,default=350)
    parser.add_argument('--seed',type=int,default=20260721)
    parser.add_argument('--quick',action='store_true')
    return parser.parse_args()


def discover_pairs(root,selected=None):
    notes={p.name.replace('.notes.tsv',''):p for p in (root/'notes').glob('*.notes.tsv')}
    harmonies={p.name.replace('.harmonies.tsv',''):p for p in (root/'harmonies').glob('*.harmonies.tsv')}
    stems=sorted(set(notes)&set(harmonies))
    if selected: stems=[stem for stem in stems if stem in set(selected)]
    return [(stem,notes[stem],harmonies[stem]) for stem in stems]


def work_id(piece):
    return piece.rsplit('_',1)[0] if '_' in piece else piece


def score_tree(tree,reference,tolerances,depths,budgets):
    depth_rows=[]; prominence_rows=[]
    for depth in depths:
        predicted=collect_top_splits(tree,depth)
        for tolerance in tolerances:
            score=boundary_scores(predicted,reference,tolerance); score.pop('matches')
            score.update({'depth':depth,'tolerance_qb':tolerance})
            depth_rows.append(score)
    specifications=[(str(value),int(value),False) for value in sorted(set(budgets))]
    specifications.append(('oracle',len(reference),True))
    for label,budget,oracle in specifications:
        predicted=collect_prominent_splits(tree,budget)
        for tolerance in tolerances:
            score=boundary_scores(predicted,reference,tolerance); score.pop('matches')
            score.update({'budget_label':label,'boundary_budget':budget,
                          'oracle_budget':oracle,'tolerance_qb':tolerance})
            prominence_rows.append(score)
    return depth_rows,prominence_rows


def leave_one_work_out_selection(prominence,tolerance=8.0):
    '''Choose method/context/budget without observing the held-out work.'''
    fixed=prominence[(~prominence.oracle_budget)&
                     np.isclose(prominence.tolerance_qb,tolerance)].copy()
    if fixed.empty: return pd.DataFrame(),pd.DataFrame()
    dimensions=['method','context_radius','boundary_budget']
    work=(fixed.groupby(['work','search']+dimensions,as_index=False)
          [['precision','recall','f1']].mean())
    selections=[]; results=[]
    works=sorted(work.work.unique())
    if len(works)<2: return pd.DataFrame(),pd.DataFrame()
    for held_out in works:
        training=work[work.work!=held_out]
        # One common configuration for both searches isolates the search
        # algorithm.  Search is averaged only on training works.
        candidates=(training.groupby(dimensions,as_index=False)
                    [['precision','recall','f1']].mean()
                    .sort_values(['f1','recall','precision']+dimensions,
                                 ascending=[False,False,False,True,True,True],
                                 kind='stable'))
        chosen=candidates.iloc[0]
        selection={'held_out_work':held_out,'training_works':len(works)-1,
                   **{name:chosen[name] for name in dimensions},
                   'training_f1':chosen.f1,'training_recall':chosen.recall}
        selections.append(selection)
        mask=work.work.eq(held_out)
        for name in dimensions: mask &= work[name].eq(chosen[name])
        tested=work[mask].copy()
        tested['selected_method']=chosen.method
        tested['selected_context_radius']=int(chosen.context_radius)
        tested['selected_boundary_budget']=int(chosen.boundary_budget)
        results.extend(tested.to_dict('records'))
    return pd.DataFrame(selections),pd.DataFrame(results)


def main():
    args=parse_args()
    if args.quick:
        args.methods=['key_profile']; args.context_radii=[2]
        args.tolerances=[8]; args.depths=[4]; args.boundary_budgets=[5,10]
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=discover_pairs(args.data_root,args.piece)
    if args.quick and not args.piece:
        preferred=[pair for pair in pairs if pair[0]=='n11op95_01']
        pairs=preferred or pairs[:1]
    if not pairs: raise SystemExit('No paired ABC notes/harmonies files found')

    depth_all=[]; prominence_all=[]; diagnostic_all=[]; ted_all=[]; statuses=[]
    affinity_metadata=[]
    for number,(piece,notes,harmonies) in enumerate(pairs,1):
        piece_depth=[]; piece_prominence=[]; piece_diagnostics=[]; piece_ted=[]
        try:
            segments,total=dcml_localkey_segments(harmonies)
            reference=[end for _,end,_ in segments[:-1]]
            matrix,bounds=load_pc_bins(notes,args.bin_size)
            if len(matrix)>args.max_bins:
                raise ValueError(f'{len(matrix)} bins exceeds max_bins={args.max_bins}')
            specs,estimated_key=distance_specs(matrix.sum(axis=0))
            missing=[method for method in args.methods if method not in specs]
            if missing: raise KeyError(f'unknown methods: {missing}')
            for method in args.methods:
                for radius in args.context_radii:
                    affinity,affinity_info=pairwise_affinity(
                        matrix,specs[method],context_radius=radius)
                    affinity_metadata.append({'piece':piece,'method':method,
                                              **affinity_info})
                    started=time.perf_counter()
                    greedy=greedy_adjacent_average_linkage(
                        matrix,bounds,affinity,ClusterNode)
                    greedy_seconds=time.perf_counter()-started
                    greedy_revenue,greedy_normalized=affinity_tree_revenue(greedy,affinity)
                    dp,dp_info=optimal_affinity_tree(
                        matrix,bounds,affinity,ClusterNode,tie_break=args.tie_break,
                        max_bins=args.max_bins)
                    assert_valid_ordered_binary_tree(greedy,bounds)
                    assert_valid_ordered_binary_tree(dp,bounds)
                    if dp_info.total_revenue<greedy_revenue-1e-8:
                        raise AssertionError('exact DP revenue is below greedy revenue')
                    trees=[('greedy',greedy,greedy_revenue,greedy_normalized,
                            greedy_seconds,0,0),
                           ('dp',dp,dp_info.total_revenue,dp_info.normalized_revenue,
                            dp_info.elapsed_seconds,dp_info.evaluated_splits,dp_info.tie_count)]
                    for search,tree,revenue,normalized,seconds,evaluated,ties in trees:
                        common={'piece':piece,'work':work_id(piece),'search':search,
                                'method':method,'context_radius':radius,
                                'bin_size_qb':args.bin_size}
                        depth_rows,prominence_rows=score_tree(
                            tree,reference,args.tolerances,args.depths,args.boundary_budgets)
                        for row in depth_rows: piece_depth.append({**common,**row})
                        for row in prominence_rows: piece_prominence.append({**common,**row})
                        shape=tree_shape_diagnostics(tree)
                        piece_diagnostics.append({**common,**shape,
                            'objective_name':'ordered_similarity_revenue',
                            'objective_revenue':revenue,'normalized_revenue':normalized,
                            'runtime_seconds':seconds,'evaluated_splits':evaluated,
                            'tie_count':ties})
                        ted=ted_diagnostics(tree,segments,total,20,10)
                        piece_ted.append({**common,**ted})
            depth_all.extend(piece_depth); prominence_all.extend(piece_prominence)
            diagnostic_all.extend(piece_diagnostics); ted_all.extend(piece_ted)
            statuses.append({'piece':piece,'status':'success','message':'',
                             **{f'estimated_{key}':value for key,value in estimated_key.items()}})
            print(f'[{number}/{len(pairs)}] {piece}: success ({len(matrix)} bins)',flush=True)
        except Exception as error:
            state='skipped' if 'exceeds max_bins=' in str(error) else 'failed'
            statuses.append({'piece':piece,'status':state,'message':repr(error)})
            print(f'[{number}/{len(pairs)}] {piece}: {state}: {error}',file=sys.stderr,flush=True)

    depth=pd.DataFrame(depth_all); prominence=pd.DataFrame(prominence_all)
    diagnostics=pd.DataFrame(diagnostic_all); ted=pd.DataFrame(ted_all)
    pd.DataFrame(statuses).to_csv(args.output_dir/'run_status.csv',index=False)
    pd.DataFrame(affinity_metadata).to_csv(args.output_dir/'affinity_metadata.csv',index=False)
    if depth.empty: raise SystemExit('All pieces failed; inspect run_status.csv')
    depth.to_csv(args.output_dir/'depth_boundary_per_piece.csv',index=False)
    prominence.to_csv(args.output_dir/'prominence_boundary_per_piece.csv',index=False)
    diagnostics.to_csv(args.output_dir/'tree_diagnostics.csv',index=False)
    ted.to_csv(args.output_dir/'ted_auxiliary.csv',index=False)

    depth_groups=['search','method','context_radius','bin_size_qb','tolerance_qb','depth']
    work_depth=(depth.groupby(['work']+depth_groups,as_index=False)
                [['precision','recall','f1']].mean())
    bootstrap_summary(work_depth,depth_groups,samples=2000,seed=args.seed).to_csv(
        args.output_dir/'depth_boundary_summary.csv',index=False)
    prominence_groups=['search','method','context_radius','bin_size_qb',
                       'tolerance_qb','budget_label','boundary_budget','oracle_budget']
    work_prominence=(prominence.groupby(['work']+prominence_groups,as_index=False)
                     [['precision','recall','f1']].mean())
    bootstrap_summary(work_prominence,prominence_groups,samples=2000,seed=args.seed).to_csv(
        args.output_dir/'prominence_boundary_summary.csv',index=False)

    selections,selected=leave_one_work_out_selection(prominence,8.0)
    selections.to_csv(args.output_dir/'loowcv_selections.csv',index=False)
    selected.to_csv(args.output_dir/'loowcv_per_work.csv',index=False)
    if not selected.empty:
        selected.groupby('search',as_index=False).agg(
            works=('work','nunique'),precision=('precision','mean'),
            recall=('recall','mean'),f1=('f1','mean'),std_f1=('f1','std')).to_csv(
                args.output_dir/'loowcv_summary.csv',index=False)
        tests=paired_permutation_tests(
            selected[['work','search','f1']].rename(columns={'search':'method'}),
            samples=10000,seed=args.seed,unit_column='work')
        tests.to_csv(args.output_dir/'loowcv_paired_test.csv',index=False)

    comparison=(diagnostics.pivot_table(
        index=['piece','work','method','context_radius'],columns='search',
        values='objective_revenue',aggfunc='first').reset_index())
    if {'greedy','dp'}<=set(comparison):
        comparison['objective_gap']=comparison.dp-comparison.greedy
        comparison['relative_objective_gap']=comparison.objective_gap/comparison.dp.abs().replace(0,np.nan)
    comparison.to_csv(args.output_dir/'objective_comparison.csv',index=False)

    status=pd.DataFrame(statuses)
    metadata={'recommended_entry_point':'scripts/evaluate_optimized_stage.py',
      'objective':'Similarity revenue dual to Dasgupta cost, restricted to contiguous ordered binary trees.',
      'greedy':'Adjacent bottom-up average linkage on the same leaf affinity.',
      'dp_optimality':'Exact only for fixed leaves, fixed affinity, temporal contiguity and the stated objective.',
      'boundary_primary':'DCML local-key Boundary F1; labels are never used to construct a tree.',
      'configuration_selection':'Leave-one-work-out; one common method/context/budget is selected for both searches from other works only.',
      'prominence':'Boundary rank is the temporal span of the adjacent leaves LCA.',
      'ted_interpretation':'Auxiliary discrepancy from a flat DCML-segment-induced tree, not hierarchical ground truth.',
      'bin_size_qb':args.bin_size,'tolerances_qb':args.tolerances,
      'depths':args.depths,'boundary_budgets':args.boundary_budgets,
      'context_radii':args.context_radii,'tie_break':args.tie_break,
      'methods':args.methods,'random_seed':args.seed,
      'successful_movements':int((status.status=='success').sum()),
      'skipped_movements':int((status.status=='skipped').sum()),
      'failed_movements':int((status.status=='failed').sum())}
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    if not selected.empty:
        print('\nLEAVE-ONE-WORK-OUT SUMMARY')
        print(pd.read_csv(args.output_dir/'loowcv_summary.csv').to_string(index=False))
    print(json.dumps({key:metadata[key] for key in
          ['successful_movements','skipped_movements','failed_movements']},indent=2))


if __name__=='__main__': main()
