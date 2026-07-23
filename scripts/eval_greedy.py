'''Corpus-level greedy evaluation. Boundary F1 is primary; flat-tree TED is auxiliary.'''
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from greedy_clustering import (load_pc_bins,distance_functions,greedy_cluster,
    balanced_tree,random_adjacent_merge_tree)
from greedy_evaluation import (dcml_localkey_segments,collect_top_splits,boundary_scores,
    collect_k_splits,ted_diagnostics,bootstrap_summary,micro_summary,paired_permutation_tests)

DEFAULT_METHODS=['euclidean','tonic_weighted','circle_of_fifths','key_profile',
                 'balanced_tree','random_adjacent_merge']

def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('external/ABC'))
    parser.add_argument('--output-dir',type=Path,default=Path('results/greedy_eval'))
    parser.add_argument('--piece',action='append',help='piece stem; repeat to select several')
    parser.add_argument('--methods',nargs='+',default=DEFAULT_METHODS)
    parser.add_argument('--bin-sizes',nargs='+',type=float,default=[2,4,8,16])
    parser.add_argument('--tolerances',nargs='+',type=float,default=[2,4,8,12,24])
    parser.add_argument('--depths',nargs='+',type=int,default=[1,2,3,4,5,6])
    parser.add_argument('--boundary-budgets',nargs='+',type=int,default=[5,10,20])
    parser.add_argument('--center-bins',nargs='+',type=int,default=[10,20,40])
    parser.add_argument('--width-bins',nargs='+',type=int,default=[5,10,20])
    parser.add_argument('--random-repeats',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20260721)
    parser.add_argument('--include-ablation',action='store_true')
    parser.add_argument('--quick',action='store_true',help='small deterministic smoke run')
    return parser.parse_args()

def discover_pairs(root,selected=None):
    notes={p.name.replace('.notes.tsv',''):p for p in (root/'notes').glob('*.notes.tsv')}
    harmonies={p.name.replace('.harmonies.tsv',''):p for p in (root/'harmonies').glob('*.harmonies.tsv')}
    stems=sorted(set(notes)&set(harmonies))
    if selected: stems=[stem for stem in stems if stem in set(selected)]
    return [(stem,notes[stem],harmonies[stem]) for stem in stems]

def stable_rng(seed,piece,bin_size,replicate):
    text=f'{seed}|{piece}|{bin_size}|{replicate}'.encode('utf-8')
    value=int.from_bytes(hashlib.sha256(text).digest()[:8],'little')
    return np.random.default_rng(value)

def trees_for_piece(matrix,bounds,methods,include_ablation,seed,piece,bin_size,repeats):
    distances,key=distance_functions(matrix.sum(axis=0),include_ablation)
    trees={}
    for method in methods:
        if method in distances: trees[method]=[greedy_cluster(matrix,bounds,distances[method])]
        elif method=='balanced_tree': trees[method]=[balanced_tree(matrix,bounds)]
        elif method=='random_adjacent_merge':
            trees[method]=[random_adjacent_merge_tree(
                matrix,bounds,stable_rng(seed,piece,bin_size,replicate))
                for replicate in range(repeats)]
        else: raise ValueError(f'unknown method: {method}')
    return trees,key

def evaluate_tree(tree,segments,total,tolerances,depths,budgets,center_bins,width_bins):
    boundaries=[end for _,end,_ in segments[:-1]]; boundary_rows=[]; budget_rows=[]; ted_rows=[]
    for depth in depths:
        predicted=collect_top_splits(tree,depth)
        for tolerance in tolerances:
            score=boundary_scores(predicted,boundaries,tolerance); score.pop('matches')
            score.update({'depth':depth,'tolerance_qb':tolerance}); boundary_rows.append(score)
    budget_specs=[(str(budget),budget,False) for budget in sorted(set(budgets))]
    budget_specs.append(('oracle',len(boundaries),True))
    for budget_label,budget,is_oracle in budget_specs:
        predicted=collect_k_splits(tree,budget)
        for tolerance in tolerances:
            score=boundary_scores(predicted,boundaries,tolerance); score.pop('matches')
            score.update({'budget_label':budget_label,'boundary_budget':budget,
                          'oracle_budget':is_oracle,
                          'tolerance_qb':tolerance}); budget_rows.append(score)
    for centers in center_bins:
        for widths in width_bins:
            ted_rows.append(ted_diagnostics(tree,segments,total,centers,widths))
    return boundary_rows,budget_rows,ted_rows

def average_replicates(rows,key_columns,value_columns):
    frame=pd.DataFrame(rows)
    return frame.groupby(key_columns,as_index=False)[value_columns].mean() if len(frame) else frame

def plot_results(boundary,ted,out):
    macro=boundary.groupby(['method','bin_size_qb','tolerance_qb','depth'],as_index=False).f1.mean()
    bin_value=8 if 8 in set(macro.bin_size_qb) else sorted(macro.bin_size_qb)[0]
    depth_value=4 if 4 in set(macro.depth) else sorted(macro.depth)[0]
    tolerance_value=8 if 8 in set(macro.tolerance_qb) else sorted(macro.tolerance_qb)[0]
    main=macro[(macro.bin_size_qb==bin_value)&(macro.depth==depth_value)]
    fig,ax=plt.subplots();
    for method,group in main.groupby('method'): ax.plot(group.tolerance_qb,group.f1,marker='o',label=method)
    ax.set(xlabel='Tolerance (quarterbeats)',ylabel='Macro F1',title='Boundary F1 vs tolerance'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'f1_vs_tolerance.png',dpi=160); plt.close(fig)
    main=macro[(macro.bin_size_qb==bin_value)&(macro.tolerance_qb==tolerance_value)]
    fig,ax=plt.subplots()
    for method,group in main.groupby('method'): ax.plot(group.depth,group.f1,marker='o',label=method)
    ax.set(xlabel='Tree depth',ylabel='Macro F1',title='Boundary F1 vs depth'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'f1_vs_depth.png',dpi=160); plt.close(fig)
    pr=boundary[(boundary.bin_size_qb==bin_value)].groupby(
        ['method','tolerance_qb','depth'],as_index=False)[['precision','recall']].mean()
    fig,ax=plt.subplots()
    for method,group in pr.groupby('method'): ax.plot(group.recall,group.precision,marker='.',linestyle='none',label=method)
    ax.set(xlabel='Recall',ylabel='Precision',title='Tolerance/depth operating points'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'precision_recall_operating_points.png',dpi=160); plt.close(fig)
    sensitivity=ted.groupby(['method','center_bins','width_bins'],as_index=False).normalized_pruned_ted.mean()
    fig,axes=plt.subplots(1,len(sensitivity.method.unique()),figsize=(4*len(sensitivity.method.unique()),3),squeeze=False)
    for ax,(method,group) in zip(axes[0],sensitivity.groupby('method')):
        table=group.pivot(index='width_bins',columns='center_bins',values='normalized_pruned_ted')
        image=ax.imshow(table.values,aspect='auto'); ax.set_title(method); ax.set_xticks(range(len(table.columns)),table.columns)
        ax.set_yticks(range(len(table.index)),table.index); ax.set(xlabel='Center bins',ylabel='Width bins'); fig.colorbar(image,ax=ax)
    fig.tight_layout(); fig.savefig(out/'ted_sensitivity.png',dpi=160); plt.close(fig)

def plot_budget_results(frame,out):
    fixed=frame[~frame.oracle_budget].copy()
    if fixed.empty: return
    bin_value=8 if 8 in set(fixed.bin_size_qb) else sorted(fixed.bin_size_qb)[0]
    tolerance=8 if 8 in set(fixed.tolerance_qb) else sorted(fixed.tolerance_qb)[0]
    fixed=fixed[(fixed.bin_size_qb==bin_value)&(fixed.tolerance_qb==tolerance)]
    macro=fixed.groupby(['method','boundary_budget'],as_index=False).f1.mean()
    fig,ax=plt.subplots()
    for method,group in macro.groupby('method'):
        ax.plot(group.boundary_budget,group.f1,marker='o',label=method)
    ax.set(xlabel='Predicted boundary budget',ylabel='Macro F1',
           title='Boundary F1 at equal prediction budgets'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'f1_vs_boundary_budget.png',dpi=160); plt.close(fig)

def main():
    args=parse_args()
    if args.include_ablation and 'fixed_c_major_weighted_ablation' not in args.methods:
        args.methods.append('fixed_c_major_weighted_ablation')
    if args.random_repeats<1: raise SystemExit('--random-repeats must be positive')
    if args.quick:
        args.bin_sizes=[8]; args.tolerances=[8]; args.depths=[4]
        args.boundary_budgets=[5]
        args.center_bins=[10]; args.width_bins=[5]; args.random_repeats=min(2,args.random_repeats)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=discover_pairs(args.data_root,args.piece)
    if not pairs: raise SystemExit('No paired notes/harmonies files matched the selection.')
    boundary_rows=[]; budget_rows=[]; ted_rows=[]; key_rows=[]; status=[]
    for number,(piece,notes,harmonies) in enumerate(pairs,1):
        piece_boundary=[]; piece_budget=[]; piece_ted=[]; piece_keys=[]
        try:
            segments,total=dcml_localkey_segments(harmonies)
            for bin_size in args.bin_sizes:
                matrix,bounds=load_pc_bins(notes,bin_size)
                trees,key=trees_for_piece(matrix,bounds,args.methods,args.include_ablation,
                    args.seed,piece,bin_size,args.random_repeats)
                piece_keys.append({'piece':piece,'bin_size_qb':bin_size,**key})
                for method,replicates in trees.items():
                    method_boundary=[]; method_budget=[]; method_ted=[]
                    for replicate,tree in enumerate(replicates):
                        b_rows,k_rows,t_rows=evaluate_tree(tree,segments,total,args.tolerances,
                            args.depths,args.boundary_budgets,args.center_bins,args.width_bins)
                        for row in b_rows: row['replicate']=replicate
                        for row in k_rows: row['replicate']=replicate
                        for row in t_rows: row['replicate']=replicate
                        method_boundary.extend(b_rows); method_budget.extend(k_rows); method_ted.extend(t_rows)
                    b=average_replicates(method_boundary,['depth','tolerance_qb'],
                        ['tp','fp','fn','precision','recall','f1','mean_match_error_qb',
                         'predicted_boundary_count','gt_boundary_count'])
                    k=average_replicates(method_budget,
                        ['budget_label','boundary_budget','oracle_budget','tolerance_qb'],
                        ['tp','fp','fn','precision','recall','f1','mean_match_error_qb',
                         'predicted_boundary_count','gt_boundary_count'])
                    t=average_replicates(method_ted,['center_bins','width_bins'],
                        ['raw_ted','pruned_ted','normalized_raw_ted','normalized_pruned_ted',
                         'node_count_scaled_raw_ted','node_count_scaled_pruned_ted',
                         'predicted_node_count','gt_node_count','pruned_node_count','pruned_leaf_count',
                         'target_segment_count','selected_pruning_depth'])
                    for frame,collector in ((b,piece_boundary),(k,piece_budget),(t,piece_ted)):
                        frame.insert(0,'bin_size_qb',bin_size); frame.insert(0,'method',method); frame.insert(0,'piece',piece)
                        collector.extend(frame.to_dict('records'))
            boundary_rows.extend(piece_boundary); budget_rows.extend(piece_budget)
            ted_rows.extend(piece_ted); key_rows.extend(piece_keys)
            status.append({'piece':piece,'status':'success','message':''})
            print(f'[{number}/{len(pairs)}] {piece}: success',flush=True)
        except Exception as error:
            status.append({'piece':piece,'status':'failed','message':repr(error)})
            print(f'[{number}/{len(pairs)}] {piece}: failed: {error}',file=sys.stderr,flush=True)
    boundary=pd.DataFrame(boundary_rows); budget_frame=pd.DataFrame(budget_rows)
    ted_frame=pd.DataFrame(ted_rows)
    pd.DataFrame(status).to_csv(args.output_dir/'run_status.csv',index=False)
    pd.DataFrame(key_rows).to_csv(args.output_dir/'estimated_global_keys.csv',index=False)
    if boundary.empty: raise SystemExit('All pieces failed; inspect run_status.csv.')
    boundary.to_csv(args.output_dir/'boundary_per_piece.csv',index=False)
    budget_frame.to_csv(args.output_dir/'boundary_fixed_budget_per_piece.csv',index=False)
    ted_frame.to_csv(args.output_dir/'ted_auxiliary_per_piece.csv',index=False)
    groups=['method','bin_size_qb','tolerance_qb','depth']
    boundary['work']=boundary.piece.str.rsplit('_',n=1).str[0]
    work_boundary=boundary.groupby(['work']+groups,as_index=False)[
        ['precision','recall','f1','mean_match_error_qb']].mean()
    work_macro=bootstrap_summary(work_boundary,groups,samples=2000,seed=args.seed)
    work_macro=work_macro.rename(columns={'n_pieces':'n_works'})
    work_macro.to_csv(args.output_dir/'boundary_macro_summary.csv',index=False)
    boundary.groupby(groups,as_index=False).agg(
        n_movements=('piece','nunique'),mean_f1=('f1','mean'),std_f1=('f1','std'),
        mean_precision=('precision','mean'),mean_recall=('recall','mean')).to_csv(
        args.output_dir/'boundary_movement_descriptive.csv',index=False)
    micro_summary(boundary,groups).to_csv(args.output_dir/'boundary_micro_summary.csv',index=False)
    budget_groups=['method','bin_size_qb','tolerance_qb','budget_label','oracle_budget']
    budget_frame['work']=budget_frame.piece.str.rsplit('_',n=1).str[0]
    work_budget=budget_frame.groupby(['work']+budget_groups,as_index=False).f1.mean()
    budget_macro=bootstrap_summary(work_budget,budget_groups,samples=2000,seed=args.seed)
    budget_macro=budget_macro.rename(columns={'n_pieces':'n_works'})
    budget_macro.to_csv(args.output_dir/'boundary_fixed_budget_summary.csv',index=False)
    primary=work_boundary[(work_boundary.bin_size_qb==8)&
        (work_boundary.tolerance_qb==8)&(work_boundary.depth==4)]
    paired_permutation_tests(primary,samples=10000,seed=args.seed,
        unit_column='work').to_csv(args.output_dir/'primary_paired_tests.csv',index=False)
    plot_results(boundary,ted_frame,args.output_dir)
    plot_budget_results(budget_frame,args.output_dir)
    metadata={'primary_metric':'boundary F1','primary_configuration':{'bin_size_qb':8,'tolerance_qb':8,'depth':4},
      'ted_interpretation':'Structural discrepancy from a flat tree induced by DCML local-key segments; not hierarchical ground truth.',
      'node_count_scaled_ted':'TED / max(predicted node count, reference node count); not bounded by one.',
      'boundary_matching':'Order-preserving dynamic programming; maximise matches, then minimise timing error.',
      'inference_unit':'Quartet/work; movement-level summaries are descriptive.',
      'empty_bin_policy':'Silent fixed bins are collapsed into nearest sounding-bin time regions.',
      'distance_definitions':{'euclidean':'Euclidean distance between L1-normalised chromagrams.',
       'tonic_weighted':'Weighted Euclidean distance using a KS major/minor profile rotated to a global key estimated from whole-piece notes.',
       'circle_of_fifths':'Euclidean distance between 2D pitch-class centroids on the circle of fifths; Tonnetz-inspired, not a full Tonnetz.',
       'key_profile':'Euclidean distance between correlations with 24 rotated KS profiles.',
       'fixed_c_major_weighted_ablation':'Fixed absolute C-major KS weights; ablation only.'},
      'ks_profiles':'Krumhansl-Schmuckler major/minor key profiles',
      'normalisation':'Non-empty pitch-class duration vectors are L1-normalised.',
      'random_seed':args.seed,'random_repeats':args.random_repeats,'paired_files':len(pairs),
      'successful':sum(x['status']=='success' for x in status),'failed':sum(x['status']=='failed' for x in status)}
    (args.output_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({k:metadata[k] for k in ['paired_files','successful','failed']},indent=2))

if __name__=='__main__': main()
