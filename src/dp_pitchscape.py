'''Pitch Scapes visualization for exact ordered DP clustering trees.'''

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from dp_clustering import assert_valid_ordered_binary_tree, optimal_adjacent_binary_tree
from greedy_clustering import ClusterNode, distance_specs, load_pc_bins
from greedy_evaluation import tree_shape_diagnostics
from ordered_affinity import (
    optimal_affinity_tree,
    optimal_boundary_aware_tree,
    pairwise_affinity,
)


PROJECT_ROOT=Path(__file__).resolve().parents[1]
DEFAULT_MIDI=PROJECT_ROOT/'n11op95_01.mid'
DEFAULT_NOTES=PROJECT_ROOT/'external'/'ABC'/'notes'/'n11op95_01.notes.tsv'

METHOD_LABELS={
    'key_profile':'key-profile distance',
    'circle_of_fifths':'circle-of-fifths embedding distance',
    'tonic_weighted':'tonic-relative weighted distance',
}


def collect_tree_geometry(root: Any, max_depth: int) -> tuple[list[tuple],list[tuple]]:
    '''Collect node and edge geometry through an inclusive maximum depth.'''
    if max_depth<0:
        raise ValueError('max_depth must be non-negative')
    points=[]; edges=[]
    def walk(node,parent=None,depth=0):
        if depth>max_depth:
            return
        current=((float(node.start)+float(node.end))/2,
                 float(node.end)-float(node.start),depth)
        points.append(current)
        if parent is not None:
            edges.append((parent,current))
        for child in list(getattr(node,'children',[]) or []):
            walk(child,current,depth+1)
    walk(root)
    return points,edges


def render_dp_pitchscape(*, method: str, midi_path: Path, notes_path: Path,
                         output_path: Path, bin_size_qb: float=8.0,
                         max_depth: int=8, max_bins: int=350,
                         n_samples: int=200, dpi: int=140,
                         show: bool=False,
                         objective: str='regularized_affinity',
                         balance_weight: float=0.6,
                         affinity_context_radius: int=0,
                         tie_break: str='midpoint') -> dict:
    '''Build the exact DP tree, overlay it on a Pitch Scape, and save a PNG.'''
    midi_path=Path(midi_path).resolve()
    notes_path=Path(notes_path).resolve()
    output_path=Path(output_path).resolve()
    if not midi_path.is_file():
        raise FileNotFoundError(f'MIDI file not found: {midi_path}')
    if not notes_path.is_file():
        raise FileNotFoundError(f'notes TSV not found: {notes_path}')
    if method not in METHOD_LABELS:
        raise ValueError(f'unsupported visualization method: {method}')
    if objective not in {'regularized_affinity','affinity','additive'}:
        raise ValueError(f'unsupported DP objective: {objective}')
    if balance_weight<0:
        raise ValueError('balance_weight must be non-negative')

    if not show:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pitchscapes.plotting as plotting
    import pitchscapes.reader as reader

    matrix,bounds=load_pc_bins(notes_path,bin_size_qb)
    distances,estimated_key=distance_specs(matrix.sum(axis=0))
    affinity_metadata=None
    if objective=='additive':
        tree,diagnostics=optimal_adjacent_binary_tree(
            matrix,bounds,distances[method],ClusterNode,
            balance_lambda=balance_weight,max_bins=max_bins)
        objective_value=float(diagnostics.total_cost)
        objective_name='additive_child_distance_plus_raw_balance_penalty'
        objective_direction='minimise'
    else:
        affinity,affinity_metadata=pairwise_affinity(
            matrix,distances[method],context_radius=affinity_context_radius)
        if objective=='affinity':
            tree,diagnostics=optimal_affinity_tree(
                matrix,bounds,affinity,ClusterNode,tie_break=tie_break,
                max_bins=max_bins)
            objective_value=float(diagnostics.normalized_revenue)
            objective_name='ordered_similarity_revenue'
        else:
            contrast=np.zeros(max(0,len(matrix)-1),dtype=float)
            tree,diagnostics=optimal_boundary_aware_tree(
                matrix,bounds,affinity,contrast,ClusterNode,
                contrast_weight=0.0,balance_weight=balance_weight,
                tie_break=tie_break,max_bins=max_bins)
            objective_value=float(diagnostics.total_objective)
            objective_name='normalized_ordered_affinity_minus_balance_penalty'
        objective_direction='maximise'
    assert_valid_ordered_binary_tree(tree,bounds)
    shape=tree_shape_diagnostics(tree)
    total=float(tree.end-tree.start)
    if total<=0:
        raise ValueError('tree duration must be positive')
    points,edges=collect_tree_geometry(tree,max_depth)

    scape=reader.get_pitch_scape(str(midi_path))
    figure,axis=plt.subplots(figsize=(14,8))
    plotting.key_scape_plot(scape=scape,n_samples=n_samples,ax=axis)
    for parent,child in edges:
        x1=(parent[0]-tree.start)/total; y1=parent[1]/total
        x2=(child[0]-tree.start)/total; y2=child[1]/total
        axis.plot([x1,x2],[y1,y2],'w-',linewidth=1.2,alpha=0.8,zorder=5)
    for center,width,_ in points:
        x=(center-tree.start)/total; y=width/total
        axis.plot(x,y,'o',color='black',markersize=5,
                  markeredgecolor='white',markeredgewidth=0.8,zorder=10)

    piece=notes_path.name.replace('.notes.tsv','')
    objective_label={
        'regularized_affinity':f'balance-regularized ordered affinity, beta={balance_weight:g}',
        'affinity':'ordered affinity, beta=0',
        'additive':f'legacy additive distance, raw beta={balance_weight:g}',
    }[objective]
    axis.set_title(
        f'{piece} - exact DP tree through depth {max_depth} '
        f'({METHOD_LABELS[method]}; {objective_label})')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output_path,dpi=dpi,bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(figure)

    result={
        'piece':piece,'method':method,'leaf_count':len(matrix),
        'plotted_node_count':len(points),'plotted_edge_count':len(edges),
        'objective':objective_value,'objective_name':objective_name,
        'objective_direction':objective_direction,
        'balance_weight':balance_weight,
        'root_split':diagnostics.root_split,
        'evaluated_splits':diagnostics.evaluated_splits,
        'tie_count':diagnostics.tie_count,
        'runtime_seconds':diagnostics.elapsed_seconds,
        'estimated_key':estimated_key,'affinity_metadata':affinity_metadata,
        'tree_shape':shape,'output_path':str(output_path),
    }
    print(f'Piece: {piece}')
    print(f'Method: {method}')
    print(f'Input: {len(matrix)} leaves at {bin_size_qb:g} qb per bin')
    if method=='tonic_weighted':
        print(f'Estimated global key: {estimated_key}')
    print(f'DP objective ({objective_direction} {objective_name}): {objective_value:.6f}')
    print(f'Root split: {diagnostics.root_split}; evaluated splits: '
          f'{diagnostics.evaluated_splits}; tied states: {diagnostics.tie_count}')
    print(f'Tree shape: max_depth={shape["max_depth"]}, '
          f'normalized Colless={shape["normalized_colless_index"]:.4f}, '
          f'root split ratio={shape["root_split_ratio"]:.4f}')
    print(f'Runtime: {diagnostics.elapsed_seconds:.3f} s')
    print(f'Plotted: {len(points)} nodes and {len(edges)} edges')
    print(f'Saved figure: {output_path}')
    return result


def visualization_cli(method: str, output_suffix: str) -> None:
    parser=argparse.ArgumentParser(
        description=f'Plot an exact DP {METHOD_LABELS[method]} tree on a Pitch Scape.')
    parser.add_argument('--midi',type=Path,default=DEFAULT_MIDI)
    parser.add_argument('--notes',type=Path,default=DEFAULT_NOTES)
    parser.add_argument('--output-dir',type=Path,
                        default=PROJECT_ROOT/'results'/'dp_stage'/'figures')
    parser.add_argument('--output',type=Path,
                        help='Explicit PNG path; overrides --output-dir.')
    parser.add_argument('--bin-size',type=float,default=8.0)
    parser.add_argument('--max-depth',type=int,default=8)
    parser.add_argument('--max-bins',type=int,default=350)
    parser.add_argument('--samples',type=int,default=200)
    parser.add_argument('--dpi',type=int,default=140)
    parser.add_argument('--objective',
                        choices=['regularized_affinity','affinity','additive'],
                        default='regularized_affinity',
                        help=('regularized_affinity is the recommended exact DP; '
                              'additive reproduces the old comb-prone figures.'))
    parser.add_argument('--balance-weight',type=float,default=0.6,
                        help=('Normalised balance weight for regularized_affinity; '
                              'raw legacy penalty for additive. Use 0 for no prior.'))
    parser.add_argument('--affinity-context-radius',type=int,default=0)
    parser.add_argument('--tie-break',choices=['earliest','midpoint','latest'],
                        default='midpoint')
    parser.add_argument('--show',action='store_true')
    args=parser.parse_args()
    piece=args.notes.name.replace('.notes.tsv','')
    output=args.output or args.output_dir/f'{piece}_{output_suffix}'
    render_dp_pitchscape(
        method=method,midi_path=args.midi,notes_path=args.notes,
        output_path=output,bin_size_qb=args.bin_size,
        max_depth=args.max_depth,max_bins=args.max_bins,
        n_samples=args.samples,dpi=args.dpi,show=args.show,
        objective=args.objective,balance_weight=args.balance_weight,
        affinity_context_radius=args.affinity_context_radius,
        tie_break=args.tie_break)
